"""
Standalone Qwen3-VL data preprocessing: tokenize, process media, pack, save as Arrow.

Zero ms-swift dependency. Output is directly loadable by ms-swift's training pipeline
via ``--cached_dataset ./output/train --cached_val_dataset ./output/val --packing true``.

Usage:
    python cache_data.py \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --datasets 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
                   'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
                   'swift/VideoChatGPT:Generic#2000' \
        --max_length 4096 --packing --output_dir ./cached_data

Dependencies:
    pip install transformers torch datasets numpy tqdm qwen-vl-utils binpacking
    # Optional: pip install modelscope  (fallback dataset loader)
"""
import argparse
import json
import os
import traceback

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _parse_dataset_syntax(ds_str: str):
    """Parse ``name:subset#sample`` syntax."""
    sample = None
    if '#' in ds_str:
        ds_str, sample = ds_str.rsplit('#', 1)
        sample = int(sample)
    subset = None
    if ':' in ds_str and not os.path.exists(ds_str):
        ds_str, subset = ds_str.split(':', 1)
    return ds_str, subset, sample


def _sample_dataset(dataset, n, seed=42):
    if n is None or n >= len(dataset):
        return dataset
    idx = np.random.RandomState(seed).permutation(len(dataset))[:n]
    return dataset.select(idx)


def load_single_dataset(ds_str: str, seed=42):
    name, subset, sample = _parse_dataset_syntax(ds_str)
    print(f'  Loading: {name}, subset={subset}, sample={sample}')
    kwargs = {'split': 'train'}
    if subset:
        kwargs['name'] = subset
    try:
        dataset = hf_load_dataset(name, **kwargs, trust_remote_code=True)
    except Exception:
        from modelscope import MsDataset
        dataset = MsDataset.load(name, subset_name=subset, split='train')
        if hasattr(dataset, '_hf_ds'):
            dataset = dataset._hf_ds
    if sample is not None:
        dataset = _sample_dataset(dataset, sample, seed=seed)
    return dataset


# ---------------------------------------------------------------------------
# Qwen3-VL Encoder
# ---------------------------------------------------------------------------

class Qwen3VLEncoder:
    """ChatML tokenizer + Qwen3-VL media processor.  No ms-swift dependency."""

    IMAGE_TOKEN = '<|image_pad|>'
    VIDEO_TOKEN = '<|video_pad|>'
    VISION_START = '<|vision_start|>'
    VISION_END = '<|vision_end|>'

    def __init__(self, processor, max_length=4096, system='You are a helpful assistant.'):
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        self.max_length = max_length
        self.system = system
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN)
        self.merge_size = getattr(processor.image_processor, 'merge_size', 2)

    # -- template --------------------------------------------------------

    def _build_chatml(self, messages, include_response=True):
        parts = []
        if self.system:
            parts.append(f'<|im_start|>system\n{self.system}<|im_end|>\n')
        for i, msg in enumerate(messages):
            role, content = msg['role'], msg.get('content', '')
            if role == 'assistant' and not include_response and i == len(messages) - 1:
                parts.append('<|im_start|>assistant\n')
                break
            parts.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
        return ''.join(parts)

    # -- media token expansion ------------------------------------------

    def _expand_media_tokens(self, encoded, media_token_id, grid_thw,
                             video_input_ids=None):
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        merge_length = self.merge_size ** 2
        idx_list = [i for i, t in enumerate(input_ids) if t == media_token_id]

        if video_input_ids is not None:
            split_token = self.tokenizer.encode('\n', add_special_tokens=False)[0]
            splited, current = [], []
            for t in video_input_ids[0].tolist():
                if t == split_token:
                    if current:
                        splited.append(current)
                    current = []
                else:
                    current.append(t)
            if current:
                splited.append(current)

        added = 0
        for i, idx in enumerate(idx_list):
            if video_input_ids is not None and i < len(splited):
                new_tokens = splited[i]
            else:
                new_tokens = [media_token_id] * int(grid_thw[i].prod() // merge_length)
            pos = idx + added
            input_ids = input_ids[:pos] + new_tokens + input_ids[pos + 1:]
            labels = labels[:pos] + [-100] * len(new_tokens) + labels[pos + 1:]
            added += len(new_tokens) - 1

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels

    # -- main encode -----------------------------------------------------

    def encode(self, row):
        messages = row.get('messages')
        if not messages:
            raise ValueError('No messages in row')

        images = row.get('images') or []
        videos = row.get('videos') or []
        if images and not isinstance(images, list):
            images = [images]
        if videos and not isinstance(videos, list):
            videos = [videos]

        user_content = messages[0].get('content', '') if messages else ''
        if images and '<image>' not in user_content:
            messages[0]['content'] = '<image>' * len(images) + user_content
        if videos and '<video>' not in user_content:
            messages[0]['content'] = '<video>' * len(videos) + messages[0].get('content', '')

        for msg in messages:
            c = msg.get('content', '') or ''
            c = c.replace('<image>', f'{self.VISION_START}{self.IMAGE_TOKEN}{self.VISION_END}')
            c = c.replace('<video>', f'{self.VISION_START}{self.VIDEO_TOKEN}{self.VISION_END}')
            msg['content'] = c

        full_ids = self.tokenizer.encode(self._build_chatml(messages, True), add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(self._build_chatml(messages, False), add_special_tokens=False)
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        encoded = {'input_ids': full_ids, 'labels': labels}

        if images:
            from qwen_vl_utils import fetch_image
            loaded = [fetch_image({'image': img}, image_patch_size=16) for img in images]
            media_inputs = self.processor.image_processor(
                images=loaded, return_tensors='pt', do_resize=False)
            encoded.update(media_inputs)
            self._expand_media_tokens(encoded, self.image_token_id,
                                      media_inputs['image_grid_thw'])

        if videos:
            from qwen_vl_utils import fetch_video
            loaded = []
            for v in videos:
                video, _ = fetch_video(
                    {'video': v}, return_video_sample_fps=True,
                    return_video_metadata=True, image_patch_size=16)
                if isinstance(video, tuple):
                    video = video[0]
                if isinstance(video, torch.Tensor):
                    video = video.to(torch.uint8)
                loaded.append(video)
            video_text = '\n'.join(
                [f'{self.VISION_START}{self.VIDEO_TOKEN}{self.VISION_END}'] * len(loaded))
            media_inputs = self.processor(
                text=[video_text], videos=loaded,
                return_tensors='pt', do_resize=False, do_sample_frames=False)
            for k in ['pixel_values_videos', 'video_grid_thw']:
                if k in media_inputs:
                    encoded[k] = media_inputs[k]
            self._expand_media_tokens(
                encoded, self.video_token_id,
                media_inputs.get('video_grid_thw'),
                video_input_ids=media_inputs.get('input_ids'))

        if len(encoded['input_ids']) > self.max_length:
            encoded['input_ids'] = encoded['input_ids'][:self.max_length]
            encoded['labels'] = encoded['labels'][:self.max_length]
            encoded['labels'][0] = -100

        return encoded


# ---------------------------------------------------------------------------
# Serialization  (encode result -> flat Arrow-friendly dict)
# ---------------------------------------------------------------------------

_TENSOR_FIELDS = [
    ('pixel_values', 'pixel_values_bytes', 'pixel_values_shape', 'image_grid_thw'),
    ('pixel_values_videos', 'pixel_values_videos_bytes', 'pixel_values_videos_shape', 'video_grid_thw'),
]


def _serialize_tensor(tensor):
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None, None
    fp16 = tensor.to(torch.float16)
    return fp16.numpy().tobytes(), list(fp16.shape)


def serialize_encoded(encoded):
    """Convert an encoder result to a flat dict suitable for Arrow storage.

    Column names match ms-swift's ``FullEncodePreprocessor`` output so that
    ``CachedEncodedDataset`` can load the data with zero modification.
    """
    length = len(encoded['input_ids'])
    result = {
        'input_ids': encoded['input_ids'],
        'labels': encoded['labels'],
        'lengths': [length],
        'loss_scale': None,
        'pixel_values_bytes': None,
        'pixel_values_shape': None,
        'image_grid_thw': None,
        'pixel_values_videos_bytes': None,
        'pixel_values_videos_shape': None,
        'video_grid_thw': None,
    }

    for pv_key, bytes_key, shape_key, thw_key in _TENSOR_FIELDS:
        val = encoded.get(pv_key)
        if val is not None:
            result[bytes_key], result[shape_key] = _serialize_tensor(val)
        thw = encoded.get(thw_key)
        if thw is not None:
            result[thw_key] = thw.tolist() if isinstance(thw, torch.Tensor) else thw

    return result


# ---------------------------------------------------------------------------
# Packing  (bin-packing via ``binpacking`` library, same algo as ms-swift)
# ---------------------------------------------------------------------------

def _calculate_matched_group(sequences, packing_length, is_finished=True):
    if not sequences:
        return [], []
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret = sequences[:-1], sequences[-1]
    else:
        ret = []
    return sequences, ret


def compute_packing_groups(lengths, packing_length):
    """Greedy bin-packing over ``[(index, length), ...]``.

    Returns ``(packed_idx, packed_length)`` — both lists of lists/ints.
    """
    BATCH_SIZE = 1000
    packed_idx, packed_length = [], []
    remaining = []
    i = 0
    with tqdm(total=len(lengths), desc='Packing') as bar:
        while True:
            new = lengths[i:i + BATCH_SIZE]
            remaining += new
            if not remaining:
                break
            i += BATCH_SIZE
            done = i >= len(lengths)
            groups, remaining = _calculate_matched_group(
                remaining, packing_length, is_finished=done)
            bar.update(len(new))
            for g in groups:
                packed_idx.append([x[0] for x in g])
                packed_length.append(sum(x[1] for x in g))
    return packed_idx, packed_length


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Standalone Qwen3-VL data cache (ms-swift compatible output)')
    p.add_argument('--model', required=True,
                   help='HuggingFace model id or local path (for tokenizer/processor)')
    p.add_argument('--datasets', nargs='+', required=True,
                   help='Dataset specs, e.g. "name:subset#sample"')
    p.add_argument('--max_length', type=int, default=4096)
    p.add_argument('--packing', action='store_true',
                   help='Pre-compute packing groups for train split')
    p.add_argument('--packing_length', type=int, default=None,
                   help='Target packing length (default: max_length)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_proc', type=int, default=1,
                   help='Parallel workers for dataset.map()')
    p.add_argument('--split_dataset_ratio', type=float, default=0.01,
                   help='Fraction to split as validation (0 = no val split)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--system', type=str, default='You are a helpful assistant.',
                   help='System prompt')
    p.add_argument('--image_max_tokens', type=int, default=1024)
    p.add_argument('--video_max_tokens', type=int, default=128)
    p.add_argument('--fps_max_frames', type=int, default=16)
    args = p.parse_args()

    if args.packing_length is None:
        args.packing_length = args.max_length

    os.environ['MAX_PIXELS'] = str(args.image_max_tokens * 28 * 28)
    os.environ['VIDEO_MAX_PIXELS'] = str(args.video_max_tokens * 28 * 28)
    os.environ['FPS_MAX_FRAMES'] = str(args.fps_max_frames)

    from transformers import AutoProcessor
    print(f'[Cache] Loading processor from {args.model} ...')
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    encoder = Qwen3VLEncoder(processor, max_length=args.max_length, system=args.system)

    # ---- load raw datasets ------------------------------------------------
    print('[Cache] Loading raw datasets ...')
    all_train, all_val = [], []
    for ds_str in args.datasets:
        ds = load_single_dataset(ds_str, seed=args.seed)
        keep = [c for c in ds.column_names if c in ('messages', 'images', 'videos')]
        ds = ds.select_columns(keep)
        if args.split_dataset_ratio > 0 and len(ds) > 10:
            split = ds.train_test_split(test_size=args.split_dataset_ratio, seed=args.seed)
            all_train.append(split['train'])
            all_val.append(split['test'])
        else:
            all_train.append(ds)

    raw_train = concatenate_datasets(all_train)
    raw_val = concatenate_datasets(all_val) if all_val else None
    print(f'[Cache] Raw samples: train={len(raw_train)}, '
          f'val={len(raw_val) if raw_val else 0}')

    # ---- encode -----------------------------------------------------------
    _empty = {
        'input_ids': [], 'labels': [], 'lengths': [0], 'loss_scale': None,
        'pixel_values_bytes': None, 'pixel_values_shape': None,
        'image_grid_thw': None,
        'pixel_values_videos_bytes': None, 'pixel_values_videos_shape': None,
        'video_grid_thw': None,
    }
    _error_counter = [0]

    def _encode_map(row):
        try:
            encoded = encoder.encode(row)
            return serialize_encoded(encoded)
        except Exception:
            if _error_counter[0] < 10:
                traceback.print_exc()
            _error_counter[0] += 1
            return _empty

    os.makedirs(args.output_dir, exist_ok=True)
    num_proc = args.num_proc if args.num_proc > 1 else None

    for split_name, raw_ds in [('train', raw_train), ('val', raw_val)]:
        if raw_ds is None:
            continue
        _error_counter[0] = 0
        print(f'[Cache] Encoding {split_name} ({len(raw_ds)} samples) ...')
        encoded_ds = raw_ds.map(
            _encode_map, num_proc=num_proc,
            remove_columns=raw_ds.column_names, desc=f'Encode {split_name}')
        before = len(encoded_ds)
        encoded_ds = encoded_ds.filter(
            lambda x: len(x['input_ids']) > 0, num_proc=num_proc)
        n_skip = before - len(encoded_ds)

        pv = encoded_ds['pixel_values_bytes']
        pvv = encoded_ds['pixel_values_videos_bytes']
        n_image = sum(1 for x in pv if x is not None)
        n_video = sum(1 for x in pvv if x is not None)
        n_text = len(encoded_ds) - n_image - n_video
        print(f'[Cache] {split_name}: {len(encoded_ds)} samples '
              f'(text={n_text}, image={n_image}, video={n_video}, skipped={n_skip})')

        out_path = os.path.join(args.output_dir, split_name)
        encoded_ds.save_to_disk(out_path)
        print(f'[Cache] Saved {split_name} -> {out_path}')

        # ---- packing (train only) -----------------------------------------
        if args.packing and split_name == 'train':
            print(f'[Cache] Computing packing groups '
                  f'(packing_length={args.packing_length}) ...')
            lengths_col = encoded_ds['lengths']
            indexed_lengths = []
            for i, l in enumerate(lengths_col):
                eff = max(l) if isinstance(l, list) else l
                if eff > 0:
                    indexed_lengths.append((i, eff))

            packed_idx, packed_length = compute_packing_groups(
                indexed_lengths, args.packing_length)
            print(f'[Cache] {len(encoded_ds)} samples -> '
                  f'{len(packed_idx)} packed groups')
            packing_path = os.path.join(args.output_dir, 'train_packing')
            Dataset.from_dict({
                'packed_idx': packed_idx,
                'packed_length': packed_length,
            }).save_to_disk(packing_path)
            print(f'[Cache] Saved packing -> {packing_path}')

    # ---- metadata ---------------------------------------------------------
    meta = {
        'full_encode': True,
        'packing': args.packing,
        'max_length': args.max_length,
        'model': args.model,
    }
    meta_path = os.path.join(args.output_dir, 'cache_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- summary ----------------------------------------------------------
    print(f'\n[Cache] Done!  Output: {args.output_dir}/')
    print(f'  train/            -- encoded training samples (Arrow)')
    if raw_val:
        print(f'  val/              -- encoded validation samples (Arrow)')
    if args.packing:
        print(f'  train_packing/    -- packing group metadata (Arrow)')
    print(f'  cache_meta.json   -- metadata')
    print(f'\nTo train with ms-swift:')
    print(f'  megatron sft \\')
    print(f'    --cached_dataset {args.output_dir}/train \\')
    if raw_val:
        print(f'    --cached_val_dataset {args.output_dir}/val \\')
    if args.packing:
        print(f'    --packing true \\')
    print(f'    --max_length {args.max_length} ...')


if __name__ == '__main__':
    main()
