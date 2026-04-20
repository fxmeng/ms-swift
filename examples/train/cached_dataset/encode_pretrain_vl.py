# Copyright (c) ModelScope Contributors. All rights reserved.
"""Encode a pretraining VL dataset into ms-swift's full_encode cached format.

Source dataset must be an HF dataset saved via ``dataset.save_to_disk(...)``,
loaded here with ``datasets.load_from_disk``. Each row must contain:
    {
        "images":    List[str],    # paths relative to --image_root
        "input_ids": List[int],    # pre-tokenized sequence with image regions
    }

IMPORTANT: the order of entries in ``images`` must follow the left-to-right
order of <vision_start> ... <vision_end> regions inside ``input_ids``.
``images[0]`` → first region, ``images[1]`` → second region, etc.

Image region layout in the source `input_ids`:

    ... <vision_start=151652> <image_pad=151655> * k <discrete tokens ...>
        <vision_end=151653> ... text ...

This script:
  1. Parses every <vision_start> ... <vision_end> region.
  2. Loads each image and runs the Qwen3-VL image processor to get
     `pixel_values` + `image_grid_thw`.
  3. Replaces the (possibly-wrong) number of image_pad tokens inside each
     region with the correct count:
         n_pad = image_grid_thw[i].prod() // (merge_size ** 2)
  4. Keeps the original discrete image tokens exactly as they appeared.
  5. Builds labels where image regions (<vision_start>, pads, discretes,
     <vision_end>) are masked with -100 and all text tokens are learned
     (pretraining). No Qwen chat-template prefix is added.
  6. Writes an Arrow dataset whose schema matches ms-swift's
     `FullEncodePreprocessor`, so training can load it directly via
     `--cached_dataset`. A per-row `discrete_tokens` column is added for
     convenience (list-of-list for multi-image rows).

Output layout:
    {output_dir}/train/            Arrow dataset
    {output_dir}/val/              Arrow dataset (when --val_ratio > 0)
    {output_dir}/cache_meta.json

Example:
    python encode_pretrain_vl.py \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --source_dataset /path/to/saved_to_disk_dir \
        --image_root /fsx/youtu-vl/jiayikuang/data_02111332/vl_images/ \
        --output_dir ./qwen3_vl_pretrain_cached \
        --num_proc 8 \
        --val_ratio 0.01
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Features, Sequence, Value, load_from_disk
from PIL import Image

VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655

# Explicit schema keeps multi-shard / multi-proc map() from breaking when some
# shards contain only text-only rows (where image fields are all None/empty).
OUTPUT_FEATURES = Features({
    'input_ids': Sequence(Value('int32')),
    'labels': Sequence(Value('int32')),
    'lengths': Sequence(Value('int32')),
    'loss_scale': Sequence(Value('float32')),
    'discrete_tokens': Sequence(Sequence(Value('int32'))),
    'pixel_values_bytes': Value('binary'),
    'pixel_values_shape': Sequence(Value('int32')),
    'pixel_values_videos_bytes': Value('binary'),
    'pixel_values_videos_shape': Sequence(Value('int32')),
    'image_grid_thw': Sequence(Sequence(Value('int32'))),
    'video_grid_thw': Sequence(Sequence(Value('int32'))),
})

_PROCESSOR_CACHE: Dict[str, Any] = {}


def _get_processor(model_id: str):
    """Lazily load the image processor per worker process."""
    if model_id not in _PROCESSOR_CACHE:
        from transformers import AutoProcessor
        _PROCESSOR_CACHE[model_id] = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return _PROCESSOR_CACHE[model_id]


def _parse_image_regions(input_ids: List[int]) -> List[Dict[str, Any]]:
    """Return one dict per <vision_start> .. <vision_end> region.

    Each dict has:
        start:    index of VISION_START_ID
        end:      index of VISION_END_ID
        discrete: list of non-pad tokens that appeared between start and end
    """
    regions: List[Dict[str, Any]] = []
    i, n = 0, len(input_ids)
    while i < n:
        tok = input_ids[i]
        if tok == VISION_START_ID:
            start = i
            j = i + 1
            while j < n and input_ids[j] != VISION_END_ID:
                j += 1
            if j >= n:
                raise ValueError(
                    f'VISION_START_ID={VISION_START_ID} at pos {start} has no '
                    f'matching VISION_END_ID={VISION_END_ID}.')
            inner = input_ids[start + 1:j]
            discrete = [t for t in inner if t != IMAGE_PAD_ID]
            regions.append({'start': start, 'end': j, 'discrete': discrete})
            i = j + 1
        else:
            i += 1
    return regions


def _build_sample(row: Dict[str, Any], model_id: str, image_root: str) -> Optional[Dict[str, Any]]:
    processor = _get_processor(model_id)
    merge_size = int(getattr(processor.image_processor, 'merge_size', 2))
    merge_len = merge_size * merge_size

    orig_input_ids = list(row['input_ids'])
    images = row.get('images') or []
    if isinstance(images, str):
        images = [images]

    regions = _parse_image_regions(orig_input_ids)
    if len(regions) != len(images):
        raise ValueError(
            f'Number of image regions ({len(regions)}) does not match '
            f'number of image paths ({len(images)}).')

    if images:
        pil_images = [Image.open(os.path.join(image_root, p)).convert('RGB') for p in images]
        image_inputs = processor.image_processor(images=pil_images, return_tensors='pt')
        pixel_values = image_inputs['pixel_values']          # [sum_i t_i*h_i*w_i, patch_dim]
        image_grid_thw = image_inputs['image_grid_thw']      # Long [num_images, 3]
        # Guard against the processor silently dropping an image (e.g. decode
        # failure). Multi-image rows must preserve exact 1:1 correspondence
        # between `images[i]`, `regions[i]` and `image_grid_thw[i]`.
        if image_grid_thw.shape[0] != len(regions):
            raise ValueError(
                f'image_processor returned {image_grid_thw.shape[0]} grids but '
                f'input_ids has {len(regions)} image regions and row has '
                f'{len(images)} image paths.')
    else:
        pixel_values = None
        image_grid_thw = None

    new_input_ids: List[int] = []
    new_labels: List[int] = []
    discrete_tokens_list: List[List[int]] = []

    cursor = 0
    for i, region in enumerate(regions):
        if region['start'] > cursor:
            chunk = orig_input_ids[cursor:region['start']]
            new_input_ids.extend(chunk)
            new_labels.extend(chunk)

        grid = image_grid_thw[i].tolist()
        n_pad = int(grid[0] * grid[1] * grid[2]) // merge_len
        discrete = region['discrete']

        image_region = (
            [VISION_START_ID]
            + [IMAGE_PAD_ID] * n_pad
            + discrete
            + [VISION_END_ID]
        )
        new_input_ids.extend(image_region)
        new_labels.extend([-100] * len(image_region))
        discrete_tokens_list.append(discrete)

        cursor = region['end'] + 1

    if cursor < len(orig_input_ids):
        tail = orig_input_ids[cursor:]
        new_input_ids.extend(tail)
        new_labels.extend(tail)

    result: Dict[str, Any] = {
        'input_ids': new_input_ids,
        'labels': new_labels,
        'lengths': [len(new_input_ids)],
        'loss_scale': None,
        'discrete_tokens': discrete_tokens_list,
        'pixel_values_videos_bytes': None,
        'pixel_values_videos_shape': None,
        'video_grid_thw': None,
    }
    if pixel_values is not None:
        t = pixel_values.to(torch.float16).contiguous()
        result['pixel_values_bytes'] = t.numpy().tobytes()
        result['pixel_values_shape'] = list(t.shape)
        result['image_grid_thw'] = image_grid_thw.tolist()
    else:
        result['pixel_values_bytes'] = None
        result['pixel_values_shape'] = None
        result['image_grid_thw'] = None
    return result


_OUTPUT_COLUMNS = [
    'input_ids', 'labels', 'lengths', 'loss_scale', 'discrete_tokens',
    'pixel_values_bytes', 'pixel_values_shape',
    'pixel_values_videos_bytes', 'pixel_values_videos_shape',
    'image_grid_thw', 'video_grid_thw',
]


def _none_row() -> Dict[str, Any]:
    return {k: None for k in _OUTPUT_COLUMNS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Model id/path, used only to load the image processor.')
    parser.add_argument('--source_dataset', required=True,
                        help='Path to an HF dataset folder produced by dataset.save_to_disk(...).')
    parser.add_argument('--image_root', required=True,
                        help='Directory that all `images` entries are relative to.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_proc', type=int, default=8)
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='If > 0, randomly hold out this ratio into {output_dir}/val.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strict', action='store_true',
                        help='Raise on any per-row failure instead of skipping it.')
    args = parser.parse_args()

    _get_processor(args.model)

    dataset = load_from_disk(args.source_dataset)

    strict = args.strict
    model_id = args.model
    image_root = args.image_root

    def _map_fn(row):
        try:
            out = _build_sample(row, model_id, image_root)
            return out if out is not None else _none_row()
        except Exception:
            if strict:
                raise
            return _none_row()

    dataset = dataset.map(
        _map_fn,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        features=OUTPUT_FEATURES,
        desc='Encoding',
    )
    dataset = dataset.filter(lambda r: r['input_ids'] is not None, num_proc=args.num_proc)

    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    if args.val_ratio and args.val_ratio > 0:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        split['train'].save_to_disk(train_dir)
        split['test'].save_to_disk(val_dir)
        print(f'cached_dataset:     `{train_dir}` ({len(split["train"])} samples)')
        print(f'cached_val_dataset: `{val_dir}` ({len(split["test"])} samples)')
    else:
        dataset.save_to_disk(train_dir)
        print(f'cached_dataset: `{train_dir}` ({len(dataset)} samples)')

    meta = {
        'full_encode': True,
        'packing': False,
        'model': args.model,
        'source_dataset': args.source_dataset,
        'image_root': args.image_root,
    }
    meta_path = os.path.join(args.output_dir, 'cache_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'cache metadata: `{meta_path}`')


if __name__ == '__main__':
    main()
