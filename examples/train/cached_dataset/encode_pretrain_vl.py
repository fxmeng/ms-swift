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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Pin thread counts BEFORE importing torch / numpy / datasets so that child
# workers forked by datasets.map inherit a single-threaded BLAS/OMP config.
# Without this, each of the N map workers spawns its own BLAS + torch intraop
# pool (typically one thread per physical core), blowing load average to N*cores.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import torch  # noqa: E402
import warnings  # noqa: E402
from datasets import Features, Sequence, Value, load_from_disk  # noqa: E402
from PIL import Image  # noqa: E402

# Silence PIL noise about palette-mode PNGs with transparency. We handle those
# images correctly below via _load_rgb_image (RGBA-composite onto white).
# Without this, tqdm gets drowned out when num_proc is large.
warnings.filterwarnings(
    'ignore',
    message='Palette images with Transparency.*',
    category=UserWarning,
)

# torch has its own intra-op / inter-op thread pools independent of OMP.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads must be called before any parallel work; if
    # something else already fired it up we silently skip.
    pass

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
# Populated once in main() and inherited by forked workers. Avoids passing
# these through every map() call.
_PROCESSOR_KWARGS: Dict[str, Any] = {}


def _get_processor(model_id: str):
    """Lazily load the image processor per worker process."""
    if model_id not in _PROCESSOR_CACHE:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **_PROCESSOR_KWARGS)
        # Also set on image_processor instance so any __call__ path picks them up.
        for k, v in _PROCESSOR_KWARGS.items():
            if hasattr(processor.image_processor, k):
                setattr(processor.image_processor, k, v)
        _PROCESSOR_CACHE[model_id] = processor
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


def _assemble_row(
    orig_input_ids: List[int],
    regions: List[Dict[str, Any]],
    pixel_values: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.Tensor],
    merge_len: int,
) -> Dict[str, Any]:
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


def _load_rgb_image(path: str) -> Image.Image:
    """Open an image and return an RGB copy.

    Handles palette-mode PNGs with transparency (common for rendered diagrams
    such as molecule structures): converts via RGBA then composites onto a
    white background. A plain ``.convert('RGB')`` on such images fills the
    transparent region with an arbitrary palette color, which silently shifts
    the background distribution away from "white" and emits a PIL warning per
    image.
    """
    img = Image.open(path)
    mode = img.mode
    if mode == 'RGB':
        return img
    if mode == 'RGBA' or (mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')


def _load_pil(path: str, image_root: str) -> Optional[Image.Image]:
    try:
        return _load_rgb_image(os.path.join(image_root, path))
    except Exception:
        return None


def _build_batch(
    batch: Dict[str, List[Any]],
    model_id: str,
    image_root: str,
    strict: bool,
    io_threads: int,
) -> Dict[str, List[Any]]:
    """Process a batch of rows.

    Optimizations vs. per-row processing:
      * ``datasets.map(batched=True)`` amortizes map/pickle/arrow overhead
        across ``batch_size`` rows.
      * A ThreadPoolExecutor overlaps image disk I/O (I/O bound, releases the
        GIL during disk read) with image-processor CPU work. This is the main
        speedup on network-mounted image storage (e.g. /fsx).
    """
    processor = _get_processor(model_id)
    merge_size = int(getattr(processor.image_processor, 'merge_size', 2))
    merge_len = merge_size * merge_size

    n = len(batch['input_ids'])

    # Pass 1: parse regions per row.
    plans: List[Optional[Dict[str, Any]]] = []
    for i in range(n):
        try:
            orig_input_ids = list(batch['input_ids'][i])
            images = batch['images'][i] or []
            if isinstance(images, str):
                images = [images]
            regions = _parse_image_regions(orig_input_ids)
            if len(regions) != len(images):
                raise ValueError(
                    f'Number of image regions ({len(regions)}) does not match '
                    f'number of image paths ({len(images)}).')
            plans.append({'input_ids': orig_input_ids, 'regions': regions, 'images': images})
        except Exception:
            if strict:
                raise
            plans.append(None)

    # Pass 2: flat (row_idx, img_idx, path) list, then parallel PIL load.
    flat: List = []
    for row_idx, plan in enumerate(plans):
        if plan is None:
            continue
        for j, path in enumerate(plan['images']):
            flat.append((row_idx, j, path))

    pil_by_row: Dict[int, List[Image.Image]] = {}
    if flat:
        max_workers = max(1, min(io_threads, len(flat)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            loaded = list(ex.map(lambda item: (item[0], item[1], _load_pil(item[2], image_root)), flat))

        failed_rows = set()
        for row_idx, _j, img in loaded:
            if img is None:
                failed_rows.add(row_idx)
                continue
            pil_by_row.setdefault(row_idx, []).append(img)

        for row_idx in failed_rows:
            if strict:
                raise RuntimeError(f'Failed to load one or more images for row {row_idx}.')
            plans[row_idx] = None
        # Also drop rows where image count mismatched after load.
        for row_idx, plan in enumerate(plans):
            if plan is None:
                continue
            if len(pil_by_row.get(row_idx, [])) != len(plan['images']):
                if strict:
                    raise RuntimeError(f'PIL load count mismatch for row {row_idx}.')
                plans[row_idx] = None

    # Pass 3: per-row image-processor call + output assembly.
    # Keeping the processor call per-row (instead of stacking a mega-batch) is
    # intentional:
    #   * Qwen2VL image processor iterates images internally anyway.
    #   * Per-row calls keep exception handling simple and let us mark one
    #     failing row ``None`` without affecting the rest of the batch.
    out: Dict[str, List[Any]] = {k: [] for k in OUTPUT_FEATURES}

    def _append_none():
        for k in out:
            out[k].append(None)

    for row_idx in range(n):
        plan = plans[row_idx]
        if plan is None:
            _append_none()
            continue
        try:
            pil_images = pil_by_row.get(row_idx, [])
            if pil_images:
                image_inputs = processor.image_processor(images=pil_images, return_tensors='pt')
                pixel_values = image_inputs['pixel_values']
                image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw.shape[0] != len(plan['regions']):
                    raise ValueError(
                        f'image_processor returned {image_grid_thw.shape[0]} grids but '
                        f'row has {len(plan["regions"])} image regions.')
            else:
                pixel_values = None
                image_grid_thw = None

            row_out = _assemble_row(
                plan['input_ids'], plan['regions'], pixel_values, image_grid_thw, merge_len)
            for k in out:
                out[k].append(row_out[k])
        except Exception:
            if strict:
                raise
            _append_none()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Model id/path, used only to load the image processor.')
    parser.add_argument('--source_dataset', required=True,
                        help='Path to an HF dataset folder produced by dataset.save_to_disk(...).')
    parser.add_argument('--image_root', required=True,
                        help='Directory that all `images` entries are relative to.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_proc', type=int, default=32,
                        help='Number of worker processes for datasets.map. Usually physical '
                             'CPU cores, or slightly less. Going past that wastes on scheduling.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='datasets.map(batched=True) batch size. Smaller batches give more '
                             'even load-balancing across workers when some rows have many images.')
    parser.add_argument('--io_threads', type=int, default=4,
                        help='PIL image loading threads per worker process. Keep small (2-4) '
                             'when the job is CPU-bound (top shows 0%% iowait), larger (8-16) '
                             'only if images live on high-latency storage.')
    parser.add_argument('--writer_batch_size', type=int, default=256,
                        help='datasets.map writer_batch_size; larger reduces arrow write overhead.')
    parser.add_argument('--max_pixels', type=int, default=None,
                        help='Cap image_processor resolution (e.g. 1048576 = 1024*1024). '
                             'This is the single biggest CPU-time knob: it limits the number '
                             'of patches per image, which directly drives resize + patch-flatten '
                             'cost and output pixel_values size.')
    parser.add_argument('--min_pixels', type=int, default=None,
                        help='Floor image_processor resolution (e.g. 65536 = 256*256).')
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='If > 0, randomly hold out this ratio into {output_dir}/val.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strict', action='store_true',
                        help='Raise on any per-row failure instead of skipping it.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Debug knob: only process the first N source rows (after --offset). '
                             'Useful for smoke-testing pipeline + parameters on a small subset '
                             'before committing to a multi-hour full run.')
    parser.add_argument('--offset', type=int, default=0,
                        help='Debug knob: skip the first N source rows before applying --limit.')
    parser.add_argument('--shuffle_subset', action='store_true',
                        help='Debug knob: shuffle the subset selected by --offset/--limit, so '
                             'you see a random sample instead of only the first rows.')
    args = parser.parse_args()

    if args.max_pixels is not None:
        _PROCESSOR_KWARGS['max_pixels'] = args.max_pixels
    if args.min_pixels is not None:
        _PROCESSOR_KWARGS['min_pixels'] = args.min_pixels
    _get_processor(args.model)

    dataset = load_from_disk(args.source_dataset)
    total_rows = len(dataset)

    if args.offset or args.limit is not None:
        start = max(0, args.offset)
        end = total_rows if args.limit is None else min(total_rows, start + args.limit)
        if start >= end:
            raise ValueError(
                f'offset={args.offset} / limit={args.limit} selects an empty range '
                f'from a dataset of {total_rows} rows.')
        if args.shuffle_subset:
            # Shuffle first so --limit can give a representative random subset
            # (e.g. for estimating cache size or catching edge cases), then
            # take the slice.
            dataset = dataset.shuffle(seed=args.seed).select(range(start, end))
        else:
            dataset = dataset.select(range(start, end))
        print(f'[subset] using rows [{start}, {end}) of {total_rows} '
              f'({end - start} rows, shuffle={args.shuffle_subset})')

    strict = args.strict
    model_id = args.model
    image_root = args.image_root
    io_threads = args.io_threads

    def _map_fn(batch):
        return _build_batch(batch, model_id, image_root, strict, io_threads)

    dataset = dataset.map(
        _map_fn,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        features=OUTPUT_FEATURES,
        writer_batch_size=args.writer_batch_size,
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
