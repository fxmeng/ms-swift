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
     the authoritative ``image_grid_thw`` (and, in ``pixel_values`` mode,
     the decoded ``pixel_values`` tensor).
  3. Replaces the (possibly-wrong) number of image_pad tokens inside each
     region with the correct count:
         n_pad = image_grid_thw[i].prod() // (merge_size ** 2)
  4. Keeps the original discrete image tokens exactly as they appeared.
  5. Builds labels where image regions (<vision_start>, pads, discretes,
     <vision_end>) are masked with -100 and all text tokens are learned
     (pretraining). No Qwen chat-template prefix is added.
  6. Writes an Arrow dataset whose schema is a superset of ms-swift's
     ``FullEncodePreprocessor`` schema plus an ``image_bytes`` column, so
     training can load it directly via ``--cached_dataset``.

Storage modes (--store_mode):

  image_bytes   (default; recommended for pretraining-scale corpora)
      Stores the raw on-disk bytes of each image (JPEG/PNG bytes, typically
      50-150 KB/image) and the authoritative ``image_grid_thw``. At training
      time ``CachedEncodedDataset`` re-runs the image_processor with the
      exact same ``max_pixels`` / ``min_pixels`` recorded in ``cache_meta``,
      producing bit-identical ``pixel_values`` to the legacy mode. Cache
      size scales with the compressed source images (~50-150 KB/row) rather
      than the decoded fp16 tensor (~2-3 MB/row in the legacy mode).

  pixel_values  (legacy; single-pass encode, zero-CPU at training time)
      Decodes images once, runs image_processor, and serializes the fp16
      ``pixel_values`` tensor into the cache. Training reads them back with
      zero image work. Produces 20-50x larger caches than ``image_bytes``
      because JPEG-compressed bytes get expanded into dense float tensors.

Both modes produce identical ``input_ids`` / ``labels`` / ``image_grid_thw``
/ ``discrete_tokens`` and identical per-batch tensors at training time. The
only difference is WHEN the decode+process work happens (encode vs. train).

Output layout:
    {output_dir}/train/            Arrow dataset
    {output_dir}/val/              Arrow dataset (when --val_ratio > 0)
    {output_dir}/cache_meta.json   Root metadata for the whole cache
    {output_dir}/shards/shard-XXXXXX-of-YYYYYY/
        cache_meta.json            Per-shard metadata (identical fields);
                                   makes the shard dir self-describing so
                                   training can recover image_processor
                                   settings from any shard path alone.
        train/ val/

Example:
    python encode_pretrain_vl.py \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --source_dataset /path/to/saved_to_disk_dir \
        --image_root /fsx/youtu-vl/jiayikuang/data_02111332/vl_images/ \
        --output_dir ./qwen3_vl_pretrain_cached \
        --store_mode image_bytes \
        --num_proc 8 \
        --val_ratio 0.01
"""

import argparse
import json
import math
import os
import shutil
import time
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
from io import BytesIO  # noqa: E402
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
#
# `image_bytes` (new) stores raw on-disk JPEG/PNG bytes per image. It coexists
# with `pixel_values_bytes` (legacy, decoded fp16 tensor) so a cache built in
# either mode is loadable by the same training code, which picks the mode at
# runtime based on which column is populated (see swift/dataset/utils.py
# CachedEncodedDataset).
OUTPUT_FEATURES = Features({
    'input_ids': Sequence(Value('int32')),
    'labels': Sequence(Value('int32')),
    'lengths': Sequence(Value('int32')),
    'loss_scale': Sequence(Value('float32')),
    'discrete_tokens': Sequence(Sequence(Value('int32'))),
    # image_bytes mode:
    'image_bytes': Sequence(Value('binary')),
    # pixel_values mode (legacy):
    'pixel_values_bytes': Value('binary'),
    'pixel_values_shape': Sequence(Value('int32')),
    # videos (unused by this script, kept for schema compat):
    'pixel_values_videos_bytes': Value('binary'),
    'pixel_values_videos_shape': Sequence(Value('int32')),
    # grid_thw shared by both modes:
    'image_grid_thw': Sequence(Sequence(Value('int32'))),
    'video_grid_thw': Sequence(Sequence(Value('int32'))),
})

# Storage modes.
STORE_MODE_IMAGE_BYTES = 'image_bytes'
STORE_MODE_PIXEL_VALUES = 'pixel_values'
VALID_STORE_MODES = (STORE_MODE_IMAGE_BYTES, STORE_MODE_PIXEL_VALUES)

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
    raw_image_bytes: Optional[List[bytes]],
    merge_len: int,
    store_mode: str,
) -> Dict[str, Any]:
    """Build one output row.

    Args:
        orig_input_ids: source sequence (contains the original vision regions).
        regions: parsed <vision_start> / <vision_end> spans.
        pixel_values: image_processor output stacked across regions. Required
            for ``store_mode == 'pixel_values'``; may be ``None`` in
            ``image_bytes`` mode (unused there).
        image_grid_thw: per-image ``(T, H_patches, W_patches)`` tensor,
            authoritative for n_pad computation. Required whenever there are
            images.
        raw_image_bytes: original on-disk bytes per image. Required for
            ``image_bytes`` mode; unused for ``pixel_values`` mode.
        merge_len: ``merge_size ** 2``; used to compute n_pad from the grid.
        store_mode: one of ``VALID_STORE_MODES``.
    """
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

        # input_ids keeps only the text-side vision placeholders:
        #   <vision_start> <image_pad>*n_pad <vision_end>
        # The discrete image tokens are NOT embedded into input_ids here;
        # they live in the separate `discrete_tokens` field so the model
        # code can route them through its own embedding / aligner path.
        # labels are -100 for the entire vision region (image supervision
        # is not text supervision).
        image_region_in_ids = (
            [VISION_START_ID]
            + [IMAGE_PAD_ID] * n_pad
            + [VISION_END_ID]
        )
        new_input_ids.extend(image_region_in_ids)
        new_labels.extend([-100] * len(image_region_in_ids))
        discrete_tokens_list.append(discrete)

        cursor = region['end'] + 1

    if cursor < len(orig_input_ids):
        tail = orig_input_ids[cursor:]
        new_input_ids.extend(tail)
        new_labels.extend(tail)

    # Base row; fill image-side columns below depending on store_mode.
    result: Dict[str, Any] = {
        'input_ids': new_input_ids,
        'labels': new_labels,
        'lengths': [len(new_input_ids)],
        'loss_scale': None,
        'discrete_tokens': discrete_tokens_list,
        'image_bytes': None,
        'pixel_values_bytes': None,
        'pixel_values_shape': None,
        'pixel_values_videos_bytes': None,
        'pixel_values_videos_shape': None,
        'image_grid_thw': None,
        'video_grid_thw': None,
    }

    has_images = image_grid_thw is not None and len(image_grid_thw) > 0
    if has_images:
        result['image_grid_thw'] = image_grid_thw.tolist()
        if store_mode == STORE_MODE_PIXEL_VALUES:
            if pixel_values is None:
                raise ValueError('pixel_values is required in pixel_values store_mode')
            t = pixel_values.to(torch.float16).contiguous()
            result['pixel_values_bytes'] = t.numpy().tobytes()
            result['pixel_values_shape'] = list(t.shape)
        elif store_mode == STORE_MODE_IMAGE_BYTES:
            if raw_image_bytes is None or len(raw_image_bytes) != len(regions):
                raise ValueError(
                    'raw_image_bytes must be provided for every region in '
                    'image_bytes store_mode')
            result['image_bytes'] = list(raw_image_bytes)
        else:
            raise ValueError(f'unknown store_mode: {store_mode}')
    return result


def _decode_rgb_image(raw_bytes: bytes) -> Image.Image:
    """Decode raw image bytes into an RGB PIL image.

    Handles palette-mode PNGs with transparency (common for rendered diagrams
    such as molecule structures): converts via RGBA then composites onto a
    white background. A plain ``.convert('RGB')`` on such images fills the
    transparent region with an arbitrary palette color, which silently shifts
    the background distribution away from "white" and emits a PIL warning per
    image.

    This helper is the single source of truth for "bytes → RGB PIL" used both
    at encode time (to feed into the image_processor for grid_thw) and, more
    importantly, at training time inside ``CachedEncodedDataset`` which must
    reproduce the same RGB conversion to get bit-identical pixel_values.
    """
    img = Image.open(BytesIO(raw_bytes))
    mode = img.mode
    if mode == 'RGB':
        # Force decode now so downstream doesn't hit IO surprises.
        img.load()
        return img
    if mode == 'RGBA' or (mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')


def _load_image(path: str, image_root: str) -> Optional[Dict[str, Any]]:
    """Read raw bytes + produce an RGB PIL image.

    Returns a ``{'pil': Image.Image, 'raw': bytes}`` dict, or ``None`` on any
    failure (missing file, bad bytes, etc.). We always read the raw bytes
    (even in pixel_values mode) because it's the same cost as ``Image.open``
    with a file path and lets the caller freely choose what to serialize.
    """
    try:
        abs_path = os.path.join(image_root, path)
        with open(abs_path, 'rb') as f:
            raw = f.read()
        pil = _decode_rgb_image(raw)
        return {'pil': pil, 'raw': raw}
    except Exception:
        return None


def _build_batch(
    batch: Dict[str, List[Any]],
    model_id: str,
    image_root: str,
    strict: bool,
    io_threads: int,
    store_mode: str,
) -> Dict[str, List[Any]]:
    """Process a batch of rows.

    Optimizations vs. per-row processing:
      * ``datasets.map(batched=True)`` amortizes map/pickle/arrow overhead
        across ``batch_size`` rows.
      * A ThreadPoolExecutor overlaps image disk I/O (I/O bound, releases the
        GIL during disk read) with image-processor CPU work. This is the main
        speedup on network-mounted image storage (e.g. /fsx).

    Irrespective of ``store_mode`` we always call the image_processor to get
    the authoritative ``image_grid_thw``: in ``image_bytes`` mode we throw
    away the decoded pixel_values right after (saves ~50x storage) but the
    grid_thw is written into the cache so training produces the same n_pad.
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

    # Pass 2: flat (row_idx, img_idx, path) list, then parallel load.
    # `_load_image` returns {'pil': <Image>, 'raw': <bytes>} per image.
    flat: List = []
    for row_idx, plan in enumerate(plans):
        if plan is None:
            continue
        for j, path in enumerate(plan['images']):
            flat.append((row_idx, j, path))

    # Per row: keep PIL images (for image_processor) and raw bytes (for
    # image_bytes storage mode). Same list order in both, keyed on row index.
    loaded_by_row: Dict[int, List[Dict[str, Any]]] = {}
    if flat:
        max_workers = max(1, min(io_threads, len(flat)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            loaded = list(ex.map(
                lambda item: (item[0], item[1], _load_image(item[2], image_root)),
                flat))

        failed_rows = set()
        for row_idx, _j, payload in loaded:
            if payload is None:
                failed_rows.add(row_idx)
                continue
            loaded_by_row.setdefault(row_idx, []).append(payload)

        for row_idx in failed_rows:
            if strict:
                raise RuntimeError(f'Failed to load one or more images for row {row_idx}.')
            plans[row_idx] = None
        # Also drop rows where image count mismatched after load.
        for row_idx, plan in enumerate(plans):
            if plan is None:
                continue
            if len(loaded_by_row.get(row_idx, [])) != len(plan['images']):
                if strict:
                    raise RuntimeError(f'image load count mismatch for row {row_idx}.')
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
            loaded_items = loaded_by_row.get(row_idx, [])
            if loaded_items:
                pil_images = [item['pil'] for item in loaded_items]
                raw_bytes_list = [item['raw'] for item in loaded_items]
                image_inputs = processor.image_processor(
                    images=pil_images, return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw.shape[0] != len(plan['regions']):
                    raise ValueError(
                        f'image_processor returned {image_grid_thw.shape[0]} grids but '
                        f'row has {len(plan["regions"])} image regions.')
                # Only keep the heavy pixel_values tensor in pixel_values mode.
                pixel_values = (
                    image_inputs['pixel_values']
                    if store_mode == STORE_MODE_PIXEL_VALUES else None)
            else:
                pixel_values = None
                image_grid_thw = None
                raw_bytes_list = None

            row_out = _assemble_row(
                plan['input_ids'],
                plan['regions'],
                pixel_values,
                image_grid_thw,
                raw_bytes_list,
                merge_len,
                store_mode,
            )
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
                        help='If > 0, randomly hold out this ratio into {output_dir}/val (non-shard '
                             'mode) or into each shard\'s val/ subdir (shard mode).')
    parser.add_argument('--shard_rows', type=int, default=0,
                        help='Enable resumable shard mode: slice the source dataset into chunks of '
                             'this many rows and process / save each chunk independently. 0 disables '
                             'shard mode (legacy single-pass behavior). Typical value: 200000-500000.')
    parser.add_argument('--max_shard_size', type=str, default='2GB',
                        help='save_to_disk shard size (default datasets value is 500MB). Larger '
                             'shards mean fewer arrow files, which matters for multi-TB caches '
                             'where the default produces tens of thousands of small files.')
    parser.add_argument('--save_num_proc', type=int, default=16,
                        help='Parallelism for save_to_disk (default datasets value is 1). Using '
                             'multiple workers dramatically speeds up writing a large cache.')
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
    parser.add_argument('--store_mode', choices=VALID_STORE_MODES, default=STORE_MODE_IMAGE_BYTES,
                        help='How to serialize image data into the cache:\n'
                             '  image_bytes  (default) store raw JPEG/PNG bytes + image_grid_thw;\n'
                             '                training runs image_processor on demand.\n'
                             '                Storage scales with compressed JPEG size (~50-150KB/row).\n'
                             '  pixel_values  store fp16 pixel_values tensor + image_grid_thw;\n'
                             '                training reads them with zero image CPU work.\n'
                             '                Storage scales with decoded tensor size (~2-3MB/row),\n'
                             '                i.e. 20-50x bigger than image_bytes.\n'
                             'Both modes produce bit-identical per-batch tensors at training time.')
    args = parser.parse_args()

    if args.max_pixels is not None:
        _PROCESSOR_KWARGS['max_pixels'] = args.max_pixels
    if args.min_pixels is not None:
        _PROCESSOR_KWARGS['min_pixels'] = args.min_pixels
    _get_processor(args.model)

    # Base metadata describing this cache; written at cache root + per-shard.
    # `store_mode` + `model` + (max_pixels, min_pixels) are what the training
    # reader needs to reproduce the image_processor call that produced the
    # authoritative ``image_grid_thw`` recorded in the Arrow dataset.
    meta_base: Dict[str, Any] = {
        'full_encode': True,
        'packing': False,
        'store_mode': args.store_mode,
        'model': args.model,
        'source_dataset': args.source_dataset,
        'image_root': args.image_root,
        'max_pixels': args.max_pixels,
        'min_pixels': args.min_pixels,
    }

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
    store_mode = args.store_mode

    def _map_fn(batch):
        return _build_batch(batch, model_id, image_root, strict, io_threads, store_mode)

    def _encode_and_save(src, out_dir, desc):
        """Apply map+filter to `src` and atomically materialize train[/val] subdirs at `out_dir`.

        Atomicity: we write to ``out_dir + '.tmp'`` and ``os.rename`` on success.
        A crash therefore leaves at most a ``.tmp`` sibling, never a half-finished
        ``out_dir``. The caller uses ``os.path.exists(out_dir)`` as the "done"
        marker when resuming.

        We also drop a ``cache_meta.json`` inside ``tmp_dir`` before the rename,
        so the successfully-committed output is always self-describing: the
        training reader can recover image_processor settings from any cached
        dir (per-shard or per-output) without reading a remote/separate file.
        """
        tmp_dir = out_dir + '.tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        processed = src.map(
            _map_fn,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=src.column_names,
            features=OUTPUT_FEATURES,
            writer_batch_size=args.writer_batch_size,
            desc=desc,
        )
        processed = processed.filter(
            lambda r: r['input_ids'] is not None, num_proc=args.num_proc)

        save_kwargs = {
            'max_shard_size': args.max_shard_size,
            'num_proc': args.save_num_proc,
        }

        train_n = val_n = 0
        if args.val_ratio and args.val_ratio > 0 and len(processed) > 1:
            split = processed.train_test_split(
                test_size=args.val_ratio, seed=args.seed, shuffle=True)
            split['train'].save_to_disk(os.path.join(tmp_dir, 'train'), **save_kwargs)
            split['test'].save_to_disk(os.path.join(tmp_dir, 'val'),
                                       max_shard_size=args.max_shard_size)
            train_n, val_n = len(split['train']), len(split['test'])
        else:
            processed.save_to_disk(os.path.join(tmp_dir, 'train'), **save_kwargs)
            train_n = len(processed)

        # Drop per-shard cache_meta.json so the shard is self-describing.
        # (The same fields are also written once at the cache-root level in
        # main(); having both lets training code find meta regardless of
        # which path it was handed.)
        with open(os.path.join(tmp_dir, 'cache_meta.json'), 'w') as f:
            json.dump({**meta_base, 'train_samples': train_n, 'val_samples': val_n},
                      f, indent=2, ensure_ascii=False)

        os.rename(tmp_dir, out_dir)
        return train_n, val_n

    os.makedirs(args.output_dir, exist_ok=True)

    if args.shard_rows and args.shard_rows > 0:
        # ---- shard mode: resumable, incremental ----
        shard_rows = args.shard_rows
        num_shards = math.ceil(len(dataset) / shard_rows)
        shards_root = os.path.join(args.output_dir, 'shards')
        os.makedirs(shards_root, exist_ok=True)

        train_paths: List[str] = []
        val_paths: List[str] = []

        done, total_train, total_val = 0, 0, 0
        wall_t0 = time.time()
        for i in range(num_shards):
            shard_dir = os.path.join(shards_root, f'shard-{i:06d}-of-{num_shards:06d}')
            train_sub = os.path.join(shard_dir, 'train')
            val_sub = os.path.join(shard_dir, 'val')

            if os.path.exists(shard_dir):
                # Shard was completed in a previous run. Skip re-computation;
                # still record its paths so the summary / meta is correct.
                if os.path.exists(train_sub):
                    train_paths.append(train_sub)
                if os.path.exists(val_sub):
                    val_paths.append(val_sub)
                done += 1
                print(f'[shard {i+1}/{num_shards}] skip (already done): {shard_dir}')
                continue

            start = i * shard_rows
            end = min(start + shard_rows, len(dataset))
            shard_src = dataset.select(range(start, end))

            t0 = time.time()
            tn, vn = _encode_and_save(
                shard_src, shard_dir, desc=f'shard {i+1}/{num_shards}')
            dt = time.time() - t0
            total_train += tn
            total_val += vn
            done += 1
            if os.path.exists(train_sub):
                train_paths.append(train_sub)
            if os.path.exists(val_sub):
                val_paths.append(val_sub)

            # Simple ETA extrapolation across not-yet-done shards.
            avg = (time.time() - wall_t0) / max(1, done)
            remain = (num_shards - done) * avg
            print(
                f'[shard {i+1}/{num_shards}] done in {dt:.1f}s '
                f'(train={tn}, val={vn}, avg={avg:.1f}s/shard, '
                f'ETA={remain/3600:.1f}h) -> {shard_dir}')

        meta = {
            **meta_base,
            'shard_rows': shard_rows,
            'num_shards': num_shards,
            'train_shard_paths': train_paths,
            'val_shard_paths': val_paths,
            'total_train_samples': total_train,
            'total_val_samples': total_val,
        }
        meta_path = os.path.join(args.output_dir, 'cache_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'\ncache metadata: `{meta_path}`')
        print(f'cached_dataset:     {len(train_paths)} shard paths (pass all to --cached_dataset)')
        if val_paths:
            print(f'cached_val_dataset: {len(val_paths)} shard paths (pass all to --cached_val_dataset)')
        print('\nExample training command:')
        print(
            '  --cached_dataset \\\n    ' +
            ' \\\n    '.join(train_paths[:3]) +
            (' ...' if len(train_paths) > 3 else ''))
    else:
        # ---- legacy single-pass mode ----
        train_dir = os.path.join(args.output_dir, 'train')
        val_dir = os.path.join(args.output_dir, 'val')
        # Re-use _encode_and_save by pointing out_dir at a sibling that wraps
        # both train/ and val/. That keeps atomicity semantics the same.
        wrap_dir = os.path.join(args.output_dir, '_single')
        if os.path.exists(wrap_dir):
            shutil.rmtree(wrap_dir)
        tn, vn = _encode_and_save(dataset, wrap_dir, desc='Encoding')
        # Flatten into final train_dir / val_dir layout (compat with existing
        # downstream code that expects {output_dir}/train and /val).
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.rename(os.path.join(wrap_dir, 'train'), train_dir)
        if os.path.exists(os.path.join(wrap_dir, 'val')):
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
            os.rename(os.path.join(wrap_dir, 'val'), val_dir)
        shutil.rmtree(wrap_dir)

        print(f'cached_dataset: `{train_dir}` ({tn} samples)')
        if vn:
            print(f'cached_val_dataset: `{val_dir}` ({vn} samples)')

        meta = {
            **meta_base,
            'total_train_samples': tn,
            'total_val_samples': vn,
        }
        meta_path = os.path.join(args.output_dir, 'cache_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'cache metadata: `{meta_path}`')


if __name__ == '__main__':
    main()
