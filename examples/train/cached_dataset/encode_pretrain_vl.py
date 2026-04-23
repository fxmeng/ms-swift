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
  2. Loads each image and computes the authoritative ``image_grid_thw``
     (either analytically via ``_smart_resize`` in preresize mode, or by
     calling the Qwen3-VL image_processor once in non-preresize mode).
  3. Replaces the (possibly-wrong) number of image_pad tokens inside each
     region with the correct count:
         n_pad = image_grid_thw[i].prod() // (merge_size ** 2)
  4. Keeps the original discrete image tokens exactly as they appeared.
  5. Builds labels where image regions (<vision_start>, pads, discretes,
     <vision_end>) are masked with -100 and all text tokens are learned
     (pretraining). No Qwen chat-template prefix is added.
  6. Writes an Arrow dataset compatible with ms-swift's
     ``FullEncodePreprocessor`` schema plus an ``image_bytes`` column, so
     training can load it directly via ``--cached_dataset``.

Storage: images are stored as JPEG/PNG bytes in the ``image_bytes`` column
together with the authoritative ``image_grid_thw``. At training time,
``CachedEncodedDataset`` re-runs the image_processor with the exact same
``max_pixels`` / ``min_pixels`` recorded in ``cache_meta.json``, so the
per-batch tensors downstream code sees are reproducible across runs.
Cache size scales with compressed source images (~50-150 KB/row), ~20-50x
smaller than storing decoded fp16 ``pixel_values`` tensors would be.

See ``--preresize_jpeg_quality`` for an even more compact layout that
pre-resizes + re-encodes to target resolution at encode time; it trades
bit-exact pixel reproducibility for smaller cache and faster training.

Output layout:
    {output_dir}/cache_meta.json   Root metadata for the whole cache
    {output_dir}/shards/shard-XXXXXX-of-YYYYYY/
        cache_meta.json            Per-shard metadata (identical fields);
                                   makes the shard dir self-describing so
                                   training can recover image_processor
                                   settings from any shard path alone.
        train/ val/                Arrow datasets

Example:
    python encode_pretrain_vl.py \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --source_dataset /path/to/saved_to_disk_dir \
        --image_root /fsx/youtu-vl/jiayikuang/data_02111332/vl_images/ \
        --output_dir ./qwen3_vl_pretrain_cached \
        --shard_rows 500000 \
        --num_proc 8 \
        --val_ratio 0.01
"""

import argparse
import json
import math
import os
import shutil
import socket
import threading
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
from PIL import Image, ImageOps  # noqa: E402

# Silence PIL noise about palette-mode PNGs with transparency. We handle those
# images correctly below via _load_rgb_image (RGBA-composite onto white).
# Without this, tqdm gets drowned out when num_proc is large.
warnings.filterwarnings(
    'ignore',
    message='Palette images with Transparency.*',
    category=UserWarning,
)

# Decompression-bomb cap: any source image with more than this many pixels
# raises ``Image.DecompressionBombError`` on decode, which ``_load_image``
# catches and turns into a dropped row. PIL's default (~89M pixels) only emits
# a *warning* and still loads the image — that's what causes OOMs on 128-proc
# runs where a few rogue 20kx20k PNGs can collectively balloon to >100 GB of
# uint8 arrays. At our typical ``max_pixels <= 1.3M``, any source > 200M
# pixels is pathological and cheaper to drop than to downscale.
Image.MAX_IMAGE_PIXELS = 200_000_000

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
# `image_bytes` stores raw on-disk JPEG/PNG bytes per image. The legacy
# `pixel_values_bytes` / video columns are kept in the schema for
# compatibility with ms-swift's ``FullEncodePreprocessor`` (which *writes*
# them) and ``CachedEncodedDataset`` (which *reads* them): our encoder here
# always leaves them ``None``, but the columns must exist so a dataset built
# by us and a dataset built by ``swift export --full_encode`` share a single
# Arrow schema and can be concat / packed interchangeably.
OUTPUT_FEATURES = Features({
    'input_ids': Sequence(Value('int32')),
    'labels': Sequence(Value('int32')),
    'lengths': Sequence(Value('int32')),
    'loss_scale': Sequence(Value('float32')),
    'discrete_tokens': Sequence(Sequence(Value('int32'))),
    'image_bytes': Sequence(Value('binary')),
    # Legacy / ms-swift-native columns (always None in this script):
    'pixel_values_bytes': Value('binary'),
    'pixel_values_shape': Sequence(Value('int32')),
    'pixel_values_videos_bytes': Value('binary'),
    'pixel_values_videos_shape': Sequence(Value('int32')),
    # Grid_thw (authoritative, populated in both modes):
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


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """Reimplementation of Qwen2VL's ``smart_resize`` that matches the processor bit-for-bit.

    Kept here so we can compute the authoritative ``(h_bar, w_bar)`` (and
    therefore ``image_grid_thw``) without going through the full
    ``image_processor`` pipeline. That pipeline's hot path is rescale +
    normalize + patch-flatten, which is ~30-40ms / image on CPU. In preresize
    mode we don't need those outputs — just the target size — so this helper
    lets us skip the expensive part entirely.

    Invariant: for any (height, width) that Qwen's processor would accept,
    this function returns the same ``(h_bar, w_bar)`` as
    ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.
    """
    # Reject zero/negative dimensions up front. Without this the ``min_pixels``
    # branch divides by (height * width) and blows up with ZeroDivisionError
    # (or produces inf propagating into floor/ceil). ``_load_image`` + the
    # Pass 2.5 tiny-image filter normally catch these first, but this guard
    # keeps the helper safe to call in isolation (e.g. from tests).
    if height <= 0 or width <= 0:
        raise ValueError(f'_smart_resize got non-positive size: height={height}, width={width}')
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _preresize_and_encode_jpeg(pil_img: Image.Image, factor: int, min_pixels: int,
                               max_pixels: int, quality: int):
    """Resize ``pil_img`` to the smart_resize target, re-encode as JPEG, return (bytes, (h_bar, w_bar)).

    Uses BICUBIC to match the image_processor's default resample mode, so
    that at training time (when the processor's smart_resize becomes a no-op
    because the image is already at target size) the resulting pixel_values
    differ from the "store original bytes" path only by the JPEG re-encode
    quantization noise (~±1 LSB / channel at Q95).

    subsampling=0 (4:4:4 chroma) keeps full color fidelity — the default 4:2:0
    would halve chroma resolution, visibly degrading small-text OCR images.
    """
    w, h = pil_img.size
    h_bar, w_bar = _smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if (h, w) != (h_bar, w_bar):
        pil_img = pil_img.resize((w_bar, h_bar), Image.BICUBIC)
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality, subsampling=0, optimize=False)
    return buf.getvalue(), (h_bar, w_bar)


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
    image_grid_thw: Optional[torch.Tensor],
    raw_image_bytes: Optional[List[bytes]],
    merge_len: int,
) -> Dict[str, Any]:
    """Build one output row.

    Args:
        orig_input_ids: source sequence (contains the original vision regions).
        regions: parsed <vision_start> / <vision_end> spans.
        image_grid_thw: per-image ``(T, H_patches, W_patches)`` tensor,
            authoritative for n_pad computation. Required whenever there are
            images.
        raw_image_bytes: original / pre-resized on-disk bytes per image;
            required whenever ``regions`` is non-empty.
        merge_len: ``merge_size ** 2``; used to compute n_pad from the grid.
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

    # Base row; legacy columns are always None (kept for schema compat).
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

    if image_grid_thw is not None:
        result['image_grid_thw'] = image_grid_thw.tolist()
        if raw_image_bytes is None or len(raw_image_bytes) != len(regions):
            raise ValueError(
                'raw_image_bytes must be provided for every region '
                f'(regions={len(regions)}, got {None if raw_image_bytes is None else len(raw_image_bytes)})')
        result['image_bytes'] = list(raw_image_bytes)
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
    # Apply EXIF orientation. Phone cameras almost always write JPEGs with
    # Orientation=6/8 rather than pre-rotating pixels; PIL does *not* auto-
    # rotate on ``Image.open`` or ``.convert('RGB')``, so without this step
    # our cache silently stores sideways / upside-down images and training
    # sees them that way too. ``exif_transpose`` is a no-op when there's no
    # EXIF tag, so it's safe to run unconditionally.
    img = ImageOps.exif_transpose(img)
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
    failure (missing file, bad bytes, etc.). Reading raw bytes up-front is
    the same cost as ``Image.open`` from a path and lets us forward them to
    the ``image_bytes`` storage column when not preresizing.
    """
    try:
        abs_path = os.path.join(image_root, path)
        with open(abs_path, 'rb') as f:
            raw = f.read()
        pil = _decode_rgb_image(raw)
        w, h = pil.size
        if w <= 0 or h <= 0:
            # Treat zero/negative dimensions as a load failure so downstream
            # smart_resize / image_processor never sees them. Rare but real:
            # some SVG-like PNGs + libjpeg edge cases produce (0, *) sizes.
            return None
        return {'pil': pil, 'raw': raw}
    except Exception:
        return None


def _build_batch(
    batch: Dict[str, List[Any]],
    model_id: str,
    image_root: str,
    strict: bool,
    io_threads: int,
    max_length: Optional[int],
    preresize_jpeg_quality: Optional[int],
    drop_below_min_pixels: bool,
    max_aspect_ratio: Optional[float],
) -> Dict[str, List[Any]]:
    """Process a batch of rows into the cached output schema.

    Optimizations vs. per-row processing:
      * ``datasets.map(batched=True)`` amortizes map/pickle/arrow overhead
        across ``batch_size`` rows.
      * A ThreadPoolExecutor overlaps image disk I/O (I/O bound, releases the
        GIL during disk read) with image-processor CPU work. This is the main
        speedup on network-mounted image storage (e.g. /fsx).

    The authoritative ``image_grid_thw`` is computed either analytically via
    ``_smart_resize`` (when ``preresize_jpeg_quality`` is set) or by calling
    the image_processor once; either way its decoded ``pixel_values`` tensor
    is discarded and only the bytes + grid are written to disk.
    """
    processor = _get_processor(model_id)
    ip = processor.image_processor
    merge_size = int(getattr(ip, 'merge_size', 2))
    merge_len = merge_size * merge_size
    patch_size = int(getattr(ip, 'patch_size', 14))
    # `factor` is the smart_resize rounding unit: every resized side is a
    # multiple of it. Qwen2VL uses patch_size * merge_size = 28.
    factor = patch_size * merge_size
    # Pull effective min/max pixels from the processor (already populated from
    # --max_pixels/--min_pixels via _PROCESSOR_KWARGS). Fall back to Qwen2VL
    # defaults only if the processor somehow doesn't expose them.
    min_pixels_eff = int(getattr(ip, 'min_pixels', 56 * 56))
    max_pixels_eff = int(getattr(ip, 'max_pixels', 14 * 14 * 4 * 1280))
    # Full vocab size (base vocab + added/special tokens). Used by the OOV
    # check in Pass 3 to validate the FINAL (post-_assemble_row) input_ids
    # — i.e. tokens that will actually index the text embedding at forward
    # time. Discrete image tokens from the separate image codebook are
    # legitimately above ``vocab_size`` but are routed to ``discrete_tokens``
    # by ``_assemble_row`` and never reach the text embedding, so running
    # the check post-assembly is both simpler and semantically correct.
    vocab_size = len(processor.tokenizer)

    n = len(batch['input_ids'])

    # Pass 1: parse regions per row.
    # First-class, separately-counted drop reasons (see summary log at the
    # bottom of this function). Keeping these as explicit branches instead
    # of a generic try/except makes systemic data issues observable — e.g.
    # if 40% of your source rows have region/image mismatches you want to
    # know that on the first shard, not after a 10-hour encode run.
    plans: List[Optional[Dict[str, Any]]] = []
    dropped_parse = 0           # malformed <vision_start>/<vision_end> markers
    dropped_count_mismatch = 0  # #regions(input_ids) != #images(paths)
    dropped_stray_pad = 0       # <image_pad> outside any vision region
    # OOV drops happen in Pass 3 on the final (post-_assemble_row) input_ids.
    dropped_oov = 0
    for i in range(n):
        try:
            orig_input_ids = list(batch['input_ids'][i])
            images = batch['images'][i] or []
            if isinstance(images, str):
                images = [images]
            try:
                regions = _parse_image_regions(orig_input_ids)
            except ValueError:
                # Unmatched <vision_start>/<vision_end>: structurally invalid
                # input_ids. Drop separately so users can distinguish this
                # from the (much more common) region/image count mismatch.
                dropped_parse += 1
                if strict:
                    raise
                plans.append(None)
                continue
            if len(regions) != len(images):
                # The input_ids claim N image regions but the row provides M
                # image paths. Surviving this would either:
                #   - under-count (len(images) < len(regions)): we have no
                #     image to bind to one of the regions; image_processor
                #     call would silently shift all subsequent region->image
                #     mappings by one → every downstream image_pad refers
                #     to the wrong image.
                #   - over-count (len(images) > len(regions)): extra image
                #     paths are loaded, wasting I/O, and image_grid_thw ends
                #     up with more rows than there are regions, which we'd
                #     catch later — but only after the expensive load.
                # Filtering here is cheap (no image I/O yet) and is what the
                # user actually wants: "<image> token count != real image count".
                dropped_count_mismatch += 1
                if strict:
                    raise ValueError(
                        f'Number of image regions ({len(regions)}) does not match '
                        f'number of image paths ({len(images)}).')
                plans.append(None)
                continue
            # Stray-image_pad check. Every IMAGE_PAD_ID must live strictly
            # inside some (<vision_start>, <vision_end>) open interval. A pad
            # outside any region would survive _assemble_row verbatim (only
            # tokens *inside* regions are re-emitted via n_pad expansion), so
            # at forward time it would hit the text embedding table at
            # position IMAGE_PAD_ID and contribute pure noise to the loss on
            # a position that *looks* like it should be an image token. We
            # walk regions once with a pointer, O(n_tokens).
            stray = False
            if regions:
                r_idx = 0
                r_start = regions[0]['start']
                r_end = regions[0]['end']
                for idx, tok in enumerate(orig_input_ids):
                    if tok != IMAGE_PAD_ID:
                        continue
                    while idx > r_end and r_idx + 1 < len(regions):
                        r_idx += 1
                        r_start = regions[r_idx]['start']
                        r_end = regions[r_idx]['end']
                    if not (r_start < idx < r_end):
                        stray = True
                        break
            else:
                # No regions at all: any IMAGE_PAD_ID is stray by definition.
                if IMAGE_PAD_ID in orig_input_ids:
                    stray = True
            if stray:
                dropped_stray_pad += 1
                if strict:
                    raise ValueError(
                        'Found IMAGE_PAD_ID outside any <vision_start>/<vision_end> region.')
                plans.append(None)
                continue
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

    # Pass 2.5: degenerate-image filters.
    # Two independent filters, both checked on the raw PIL size (i.e. BEFORE
    # any resize), so pathological sources never reach the processor.
    #
    # (a) tiny-image filter (--drop_below_min_pixels): smart_resize would
    #     upscale a 1x1 PNG to 56x56 to satisfy min_pixels, producing a
    #     constant-color patch that wastes training compute and emits the
    #     "channel dimension is ambiguous" warning from transformers.
    #
    # (b) aspect-ratio filter (--max_aspect_ratio): matches transformers'
    #     qwen2_vl.smart_resize ``MAX_RATIO`` validation. Without this,
    #     a 1x20000 source would get smart_resize'd to (28, 19992) and
    #     written into the cache by preresize mode; at training time the
    #     processor would then re-validate and raise ValueError on that
    #     same image, aborting the training run. Filtering at encode time
    #     turns a fatal training-time crash into a silent encode-time drop.
    #     This also guards the non-preresize code path: even when the
    #     processor is called at encode time, it would raise on these rows
    #     and poison the batch; the filter here drops them cleanly first.
    # ``max_aspect_ratio`` is already normalized to ``None`` in main() when
    # disabled (<= 0), so a non-None value here always means "check enabled".
    if drop_below_min_pixels or max_aspect_ratio is not None:
        for row_idx, items in list(loaded_by_row.items()):
            if plans[row_idx] is None:
                continue
            drop = False
            for item in items:
                w, h = item['pil'].size
                if drop_below_min_pixels and w * h < min_pixels_eff:
                    drop = True
                    break
                if max_aspect_ratio is not None and min(w, h) > 0:
                    ratio = max(w, h) / min(w, h)
                    if ratio > max_aspect_ratio:
                        drop = True
                        break
            if drop:
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
                if preresize_jpeg_quality is not None:
                    # Preresize mode: we never call the image_processor here.
                    # Instead we do the (resize -> JPEG encode) step ourselves
                    # and compute image_grid_thw analytically from the target
                    # size. At training time the processor will see an image
                    # that's already at target size, so its internal
                    # smart_resize is a no-op and it just runs rescale +
                    # normalize + patch-flatten on our JPEG-decoded pixels.
                    #
                    # Storage win comes from two compounding effects:
                    #   1. we store at *target* resolution, not source (often
                    #      10x+ smaller when sources are 4K JPEGs capped to
                    #      ~1M pixels).
                    #   2. JPEG at Q95/4:4:4 is ~8-10x smaller than PNG and
                    #      ~3x smaller than WebP lossless.
                    # Speed win at training time: resize is skipped (that's
                    # the single most expensive step in the processor).
                    raw_bytes_list = []
                    grid_rows: List[List[int]] = []
                    for item in loaded_items:
                        jpeg_bytes, (h_bar, w_bar) = _preresize_and_encode_jpeg(
                            item['pil'], factor, min_pixels_eff, max_pixels_eff,
                            preresize_jpeg_quality)
                        raw_bytes_list.append(jpeg_bytes)
                        # grid_t=1 for single-frame images (temporal folding
                        # happens inside the patch-flatten, not in grid_thw).
                        grid_rows.append([1, h_bar // patch_size, w_bar // patch_size])
                    image_grid_thw = torch.tensor(grid_rows, dtype=torch.long)
                else:
                    # Non-preresize: call the image_processor once to obtain
                    # the authoritative image_grid_thw, then discard the
                    # heavy ``pixel_values`` tensor (we only store bytes).
                    pil_images = [item['pil'] for item in loaded_items]
                    raw_bytes_list = [item['raw'] for item in loaded_items]
                    image_inputs = processor.image_processor(
                        images=pil_images, return_tensors='pt')
                    image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw.shape[0] != len(plan['regions']):
                    raise ValueError(
                        f'got {image_grid_thw.shape[0]} grids but row has '
                        f'{len(plan["regions"])} image regions.')
            else:
                image_grid_thw = None
                raw_bytes_list = None

            row_out = _assemble_row(
                plan['input_ids'],
                plan['regions'],
                image_grid_thw,
                raw_bytes_list,
                merge_len,
            )
            # Vocab-range check on the FINAL input_ids. By this point
            # ``_assemble_row`` has already routed discrete image tokens out
            # of ``input_ids`` and into ``discrete_tokens``, so every id left
            # in ``row_out['input_ids']`` is one of:
            #   - a text token (must be in [0, vocab_size)), OR
            #   - VISION_START_ID / VISION_END_ID / IMAGE_PAD_ID (special tokens
            #     that live in the added-tokens range of the tokenizer and are
            #     therefore < vocab_size by construction).
            # Checking here — on the canonical output — avoids duplicating the
            # region-walk in Pass 1 and keeps OOV detection semantically
            # aligned with "what actually enters the model at forward time".
            final_ids = row_out['input_ids']
            if final_ids:
                mn = min(final_ids)
                mx = max(final_ids)
                if mn < 0 or mx >= vocab_size:
                    dropped_oov += 1
                    if strict:
                        raise ValueError(
                            f'post-assembly input_ids has out-of-vocab id: '
                            f'min={mn} max={mx} vocab_size={vocab_size}')
                    _append_none()
                    continue
            # Drop rows whose encoded length exceeds max_length. The training
            # loader (swift/pipelines/utils.py `_select_dataset`) applies the
            # exact same filter: max(lengths) <= max_length. Pre-filtering
            # here keeps the encoded shard size == what training sees, which
            # is what ``swift export --cached_dataset --packing`` assumes when
            # it writes `packed_idx`. Without this filter, even a handful of
            # oversize rows (8 / 491k in our corpus) causes ms-swift to mark
            # the dataset as "modified" at training time and silently skip
            # the precomputed packing cache. This is NOT treated as a strict
            # failure; it's an expected soft drop.
            if max_length is not None and len(final_ids) > max_length:
                _append_none()
                continue
            for k in out:
                out[k].append(row_out[k])
        except Exception:
            if strict:
                raise
            _append_none()

    if dropped_parse or dropped_count_mismatch or dropped_stray_pad or dropped_oov:
        # Worker-local one-liner; with num_proc=128 you get up to 128 of these
        # per batch interval, which is fine — they let you spot systemic data
        # issues early (e.g. a sub-dataset with a broken export). Aggregating
        # across workers is non-trivial in datasets.map, skipped by design.
        print(
            f'[encode] batch size={n}: dropped '
            f'{dropped_count_mismatch} (image count mismatch) + '
            f'{dropped_parse} (malformed vision markers) + '
            f'{dropped_stray_pad} (stray IMAGE_PAD_ID) + '
            f'{dropped_oov} (out-of-vocab input_ids)')

    return out


# ---------------------------------------------------------------------------
# Multi-pod / multi-host shard claiming.
#
# Goal: let N independent processes (on different hosts, possibly different
# pods / nodes) all encode into the same --output_dir without stepping on
# each other, with no external coordinator (no DB, no Redis, no ZK).
#
# Model: each shard has three possible states on the shared filesystem:
#   - committed:          shard-XXXXXX-of-YYYYYY/          (done, immutable)
#   - claimed-in-flight:  shard-XXXXXX-of-YYYYYY.claim/    (someone working)
#   - unclaimed:          (neither dir exists)             (free to grab)
#
# The claim directory is created with ``os.mkdir`` which is atomic on POSIX
# and NFSv3+: exactly one process out of N concurrent callers wins the race,
# the rest get FileExistsError. That single syscall is our mutual-exclusion
# primitive — no advisory locks (flock isn't reliable on NFS), no rename
# tricks, no external services.
#
# Inside the claim dir lives ``owner.json`` with {hostname, pid, started_ts,
# heartbeat_ts}. A daemon thread in the owning process re-writes heartbeat_ts
# every ``heartbeat_interval`` seconds. Other processes, on seeing a claim
# dir, read heartbeat_ts:
#   - fresh (heartbeat_ts within stale_threshold): skip this shard, try next
#   - stale (last heartbeat > stale_threshold ago): the owner is dead (pod
#     evicted, node crashed, OOM'd); rmtree the claim and attempt to mkdir
#     it ourselves. The rmtree-then-mkdir sequence is racy with other
#     stealers, but mkdir is the final arbiter: only one wins.
#
# Failure modes handled:
#   - Pod killed mid-shard:        heartbeat stops → after 10min another pod
#                                  steals the claim, re-processes the shard.
#   - NFS metadata cache lag:      heartbeat writes use rename-in-place which
#                                  forces NFS to invalidate and flush; worst-
#                                  case we see ~30s of staleness.
#   - Two pods stealing at once:   one mkdir wins, the other gets FileExists
#                                  and moves on (idempotent).
#   - Crash during commit:         the atomic rename shard_dir.tmp → shard_dir
#                                  is the commit point; either the shard is
#                                  fully there or not. Partial shards are
#                                  impossible.
# ---------------------------------------------------------------------------

def _claim_owner_info(claim_dir: str) -> Dict[str, Any]:
    return {
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'started_ts': time.time(),
        'heartbeat_ts': time.time(),
        'claim_dir': claim_dir,
    }


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON via ``tmp → rename`` so readers never see a partial file."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.rename(tmp, path)


def _start_claim_heartbeat(claim_dir: str, interval_s: float) -> threading.Event:
    """Spawn a daemon thread that updates owner.json every ``interval_s`` seconds.

    Returns a stop Event: the caller must ``.set()`` it before releasing the
    claim, otherwise a dying main thread would leave a zombie heartbeat
    writing into an already-deleted directory.
    """
    owner_path = os.path.join(claim_dir, 'owner.json')
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval_s):
            try:
                # Re-read previous owner to preserve started_ts across beats.
                if os.path.exists(owner_path):
                    with open(owner_path) as f:
                        owner = json.load(f)
                else:
                    owner = _claim_owner_info(claim_dir)
                owner['heartbeat_ts'] = time.time()
                _atomic_write_json(owner_path, owner)
            except Exception:
                # Claim dir may have been removed (we're shutting down) or
                # filesystem is temporarily down; don't crash the encoder
                # thread — the next beat will retry or the stop flag will fire.
                pass

    thread = threading.Thread(target=_beat, daemon=True, name='claim-heartbeat')
    thread.start()
    return stop


def _read_claim_age(claim_dir: str) -> Optional[float]:
    """Return seconds since last heartbeat, or ``None`` if unreadable.

    Unreadable is treated by callers as "infinitely old" so the claim gets
    stolen; this matches the intent (if we can't parse owner.json we're not
    going to wait for it to self-heal).
    """
    owner_path = os.path.join(claim_dir, 'owner.json')
    try:
        with open(owner_path) as f:
            owner = json.load(f)
        return time.time() - float(owner.get('heartbeat_ts', 0))
    except Exception:
        return None


def _try_claim_shard(shard_dir: str, claim_dir: str, stale_threshold_s: float,
                     ) -> bool:
    """Attempt to acquire an exclusive claim on ``shard_dir``.

    Returns True iff we now own ``claim_dir`` (i.e. ``os.mkdir`` succeeded for
    *us*, not some prior attempt). Callers MUST start a heartbeat on the
    returned claim and MUST ``shutil.rmtree`` the claim after commit.
    """
    if os.path.exists(shard_dir):
        return False  # already committed by someone; nothing to do.

    # Fast path: no existing claim, we try to create one.
    try:
        os.mkdir(claim_dir)
        _atomic_write_json(os.path.join(claim_dir, 'owner.json'),
                           _claim_owner_info(claim_dir))
        return True
    except FileExistsError:
        pass

    # There's already a claim. Decide whether to steal.
    age = _read_claim_age(claim_dir)
    if age is not None and age < stale_threshold_s:
        return False  # owner still alive, leave them alone.

    # Stale or unreadable → attempt steal. rmtree + mkdir is racy across
    # stealers, but mkdir is the final arbiter: only one wins.
    try:
        shutil.rmtree(claim_dir)
    except FileNotFoundError:
        pass  # someone else cleaned up before us, fine.
    except Exception as e:
        # FS permission / transient NFS error — give up on this shard for
        # this pass; the next outer loop iteration or a subsequent run
        # will try again.
        print(f'[claim] failed to rmtree stale {claim_dir}: {e!r}')
        return False

    try:
        os.mkdir(claim_dir)
        _atomic_write_json(os.path.join(claim_dir, 'owner.json'),
                           _claim_owner_info(claim_dir))
        print(f'[claim] stole stale claim on {claim_dir} (age={age:.0f}s)')
        return True
    except FileExistsError:
        # Raced with another stealer — they won. Move on.
        return False


def _release_claim(claim_dir: str, stop_heartbeat: Optional[threading.Event]) -> None:
    """Stop heartbeat + remove claim dir. Idempotent; safe on double-call."""
    if stop_heartbeat is not None:
        stop_heartbeat.set()
    try:
        shutil.rmtree(claim_dir)
    except FileNotFoundError:
        pass
    except Exception as e:
        # Log but don't raise: the shard itself is already committed
        # (release is called *after* the rename). A lingering claim dir
        # would self-expire via the stale-threshold path on the next run.
        print(f'[claim] warning: failed to rmtree claim {claim_dir}: {e!r}')


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
    parser.add_argument('--shard_rows', type=int, required=True,
                        help='Slice the source dataset into chunks of this many rows and process / '
                             'save each chunk independently. Required: shard mode is the only '
                             'supported mode (enables crash-resume + multi-pod cooperation). '
                             'Typical value: 200000-500000.')
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
    parser.add_argument('--shuffle_source', action='store_true',
                        help='Globally shuffle the source dataset (deterministic, --seed) before '
                             'slicing into shards. Recommended for multi-source pretraining corpora '
                             'that are only "soft-shuffled" (sub-datasets concatenated then mildly '
                             'permuted): produces uniform per-shard sizes and within-shard domain '
                             'diversity. Uses HF datasets index-only shuffling (near-free at 120M+ '
                             'rows; no arrow rewrite). MUST stay consistent across resumable runs: '
                             'toggling this mid-run makes already-committed shards point at a '
                             'different slice of the source data than subsequent shards.')
    parser.add_argument('--max_length', type=int, default=None,
                        help='If set, drop any sample whose encoded input_ids length exceeds '
                             'this value. Must match the --max_length you will use at training '
                             'time: ms-swift applies the same filter at load time, and a mismatch '
                             'between the cached shard and the training-filtered shard invalidates '
                             'any precomputed packing cache (forcing on-the-fly packing every run). '
                             'Typical value: 4096. Leave unset to keep the legacy behavior of '
                             'writing all rows and deferring length filtering to training.')
    parser.add_argument('--preresize_jpeg_quality', type=int, default=None,
                        help='If set (e.g. 95), resize every image to its smart_resize target '
                             '(derived from --max_pixels / --min_pixels) and re-encode as JPEG '
                             'at this quality before storing, instead of keeping the original '
                             'bytes.\n'
                             'Trade-offs vs. storing original bytes:\n'
                             '  + 5-20x smaller cache (stores at target resolution, not source).\n'
                             '  + Faster training: processor skips resize (biggest CPU cost).\n'
                             '  + Faster encode: one JPEG decode per image vs. decode + resize +\n'
                             '    rescale + normalize + patch-flatten in the full processor call.\n'
                             '  - NOT bit-identical: JPEG re-encode adds ~±1 LSB/channel noise at\n'
                             '    Q95 (negligible for pretraining; do not use <Q90).\n'
                             '  - Locks the cache to the (max_pixels, min_pixels) used at encode '
                             '    time: changing them at training time has no effect because the '
                             '    stored images are already at target size.\n'
                             'Recommended: 95 for pretraining at scale, leave unset when you need '
                             'exact pixel reproducibility (e.g. eval metric comparisons).')
    parser.add_argument('--drop_below_min_pixels', action='store_true',
                        help='Drop any sample that has at least one image smaller than the '
                             'image_processor\'s min_pixels (i.e. H*W < min_pixels). These are '
                             'typically degenerate sources (1x1 placeholders, corrupt thumbnails); '
                             'without this flag smart_resize upscales them to satisfy min_pixels, '
                             'producing a constant-color patch that wastes training compute and '
                             'emits the "channel dimension is ambiguous" warning from transformers. '
                             'Applies regardless of --preresize_jpeg_quality.')
    parser.add_argument('--claim_stale_seconds', type=float, default=600.0,
                        help='Multi-pod claim heartbeat timeout (seconds). When another pod\'s '
                             'claim on a shard has not been refreshed for this long, we consider '
                             'the owner dead and steal the claim. Must be safely larger than '
                             'your longest expected shard duration: if a slow shard takes 20min '
                             'to process and --claim_stale_seconds=600 (10min), another pod will '
                             'steal mid-encode and you\'ll do double work. Default 600 targets '
                             '~5min shards with 2x safety margin. Heartbeat interval is this / 10.')
    parser.add_argument('--claim_pass_wait_seconds', type=float, default=0,
                        help='After the first pass over all shards, if some are still claimed by '
                             'other pods, wait this many seconds for them to commit or go stale '
                             'and do another pass. 0 (default) means exit immediately — the user '
                             'is expected to monitor the commit count and re-launch if needed. '
                             'Set to e.g. 3600 for a fully-automatic "keep retrying for an hour" '
                             'behavior.')
    parser.add_argument('--max_aspect_ratio', type=float, default=200.0,
                        help='Drop any sample that has at least one image with max(H,W)/min(H,W) '
                             'strictly greater than this value (on the raw PIL size, before any '
                             'resize). Default 200.0 matches transformers qwen2_vl.smart_resize '
                             'MAX_RATIO, so surviving rows are guaranteed to be accepted by the '
                             'processor at training time. Necessary because our preresize path '
                             'bypasses the processor at encode time and would otherwise write '
                             'shapes like (28, 19992) into the cache — legal numerically, but '
                             'rejected by the processor during dataloader ingestion. Set to 0 '
                             '(or negative) to disable the check entirely.')
    args = parser.parse_args()

    if args.preresize_jpeg_quality is not None:
        if not (1 <= args.preresize_jpeg_quality <= 100):
            raise ValueError(
                f'--preresize_jpeg_quality must be in [1, 100], got {args.preresize_jpeg_quality}.')

    if args.max_pixels is not None:
        _PROCESSOR_KWARGS['max_pixels'] = args.max_pixels
    if args.min_pixels is not None:
        _PROCESSOR_KWARGS['min_pixels'] = args.min_pixels
    main_processor = _get_processor(args.model)
    main_ip = main_processor.image_processor
    # Snapshot of the *effective* processor knobs at encode time. These are
    # what _build_batch actually used; capturing them in cache_meta.json makes
    # the cache self-describing so (a) training-time readers can reproduce
    # identical n_pad / pixel_values, and (b) we can detect
    # transformers-version drift (e.g. a default constant changed upstream)
    # with a simple equality check at load time.
    effective_processor = {
        'merge_size': int(getattr(main_ip, 'merge_size', 2)),
        'patch_size': int(getattr(main_ip, 'patch_size', 14)),
        'min_pixels_eff': int(getattr(main_ip, 'min_pixels', 56 * 56)),
        'max_pixels_eff': int(getattr(main_ip, 'max_pixels', 14 * 14 * 4 * 1280)),
        'vocab_size': len(main_processor.tokenizer),
    }
    try:
        import transformers  # noqa: E402
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    try:
        import PIL  # noqa: E402
        pil_version = PIL.__version__
    except Exception:
        pil_version = None

    # Base metadata describing this cache; written at cache root + per-shard.
    # `model` + (max_pixels, min_pixels) are what the training reader needs
    # to reproduce the image_processor call that produced the authoritative
    # ``image_grid_thw`` recorded in the Arrow dataset. `store_mode` is
    # pinned to ``'image_bytes'``: kept in the schema so downstream code
    # branching on it still works, but this script never writes the legacy
    # pixel_values mode (removed in a cleanup pass).
    meta_base: Dict[str, Any] = {
        'full_encode': True,
        'packing': False,
        'store_mode': 'image_bytes',
        'model': args.model,
        'source_dataset': args.source_dataset,
        'image_root': args.image_root,
        'max_pixels': args.max_pixels,
        'min_pixels': args.min_pixels,
        'max_length': args.max_length,
        'shuffle_source': args.shuffle_source,
        'seed': args.seed,
        # New fields: let the training reader know whether stored image bytes
        # are at source resolution (bit-exact) or at target resolution (JPEG
        # re-encoded). Useful for debugging if pixel_values drift from an
        # earlier run.
        'preresize_jpeg_quality': args.preresize_jpeg_quality,
        'drop_below_min_pixels': args.drop_below_min_pixels,
        'max_aspect_ratio': args.max_aspect_ratio,
        # Version + effective-processor snapshot. Training-time readers
        # should assert the runtime `transformers.__version__` is compatible
        # with this value; a mismatch is the most common cause of
        # "silent n_pad drift" bugs where smart_resize's defaults change
        # between versions.
        'transformers_version': transformers_version,
        'pil_version': pil_version,
        'effective_processor': effective_processor,
    }

    dataset = load_from_disk(args.source_dataset)
    total_rows = len(dataset)

    if args.shuffle_source:
        # Source data is typically a concat of multiple sub-datasets, so
        # contiguous row ranges tend to be homogeneous (e.g. all molecules,
        # all photos). Without shuffling here, Step 1's shard-XXXXXX output
        # sizes vary 2-5x because shard-level image resolution / count is
        # correlated with source region. A global shuffle flattens that,
        # producing uniform shard sizes and better packing load balance.
        #
        # `.shuffle()` only rewrites the HF index map, not the underlying
        # Arrow files, so it's O(n) memory + near-free runtime even at 120M+
        # rows. Downstream random reads are fine on SSD / fsx.
        #
        # IMPORTANT: this must be deterministic (fixed seed) so that resuming
        # a crashed run hits the exact same shard → row assignment. `seed` is
        # set via --seed; don't change it between runs.
        dataset = dataset.shuffle(seed=args.seed)
        print(f'[shuffle_source] applied global shuffle (seed={args.seed}) '
              f'to {total_rows} rows')

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
    max_length = args.max_length
    preresize_jpeg_quality = args.preresize_jpeg_quality
    drop_below_min_pixels = args.drop_below_min_pixels
    # Normalize "disabled" to None so _build_batch only checks `is not None`.
    max_aspect_ratio = args.max_aspect_ratio if args.max_aspect_ratio > 0 else None

    def _map_fn(batch):
        return _build_batch(
            batch, model_id, image_root, strict, io_threads, max_length,
            preresize_jpeg_quality, drop_below_min_pixels, max_aspect_ratio)

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

        # Sanity check: the final ``lengths`` column must respect --max_length.
        # ``_build_batch`` already drops oversize rows, so this is a belt-and-
        # suspenders guard that catches silent regressions in the drop path
        # (e.g. schema changes, off-by-one bugs). Reading only the ``lengths``
        # column is cheap because Arrow is columnar: no pixel_values / image_
        # bytes data is touched, so this is sub-second even on 500k-row shards.
        if max_length is not None and len(processed) > 0:
            lengths_col = processed['lengths']
            actual_max = max(
                (max(L) if isinstance(L, list) else L)
                for L in lengths_col)
            if actual_max > max_length:
                raise RuntimeError(
                    f'encode sanity check failed for `{out_dir}`: '
                    f'max(lengths) = {actual_max} > --max_length = {max_length}. '
                    f'This should never happen; inspect _build_batch drop path.')

        save_kwargs = {
            'max_shard_size': args.max_shard_size,
            'num_proc': args.save_num_proc,
        }

        train_n = val_n = 0
        if args.val_ratio > 0 and len(processed) > 1:
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

    # Shard mode is the only supported mode: it's resumable, incremental and
    # multi-pod-safe. The loop is structured as "scan all shards, claim +
    # process whatever is free". Multiple pods pointed at the same
    # output_dir cooperate via the filesystem-only protocol documented
    # above the ``_try_claim_shard`` helper — no external coordinator.
    if args.shard_rows <= 0:
        raise ValueError(f'--shard_rows must be > 0, got {args.shard_rows}')
    shard_rows = args.shard_rows
    num_shards = math.ceil(len(dataset) / shard_rows)
    shards_root = os.path.join(args.output_dir, 'shards')
    os.makedirs(shards_root, exist_ok=True)

    # Heartbeat cadence: 1/10th of the stale threshold. At the default
    # --claim_stale_seconds=600 this is 60s, which is well within NFS
    # metadata cache TTL (typically 30-60s) so other pods see fresh
    # heartbeats. Don't lower much below ~10s: each heartbeat is a
    # rename-in-place which, on large NFS mounts, costs ~50-200ms.
    heartbeat_interval = max(5.0, args.claim_stale_seconds / 10.0)
    host_tag = f'{socket.gethostname()}/{os.getpid()}'
    print(f'[claim] this pod = {host_tag}, '
          f'stale_threshold={args.claim_stale_seconds:.0f}s, '
          f'heartbeat_interval={heartbeat_interval:.0f}s')

    train_paths: List[str] = []
    val_paths: List[str] = []

    total_train, total_val = 0, 0
    committed_by_us = 0        # shards we personally encoded this run
    committed_elsewhere = 0    # shards already done before we got here
                               # (or committed by other pods mid-loop)
    wall_t0 = time.time()

    def _collect_shard_paths(shard_dir: str) -> None:
        """Append this shard's train/val subdirs to the global lists if they exist."""
        t_sub = os.path.join(shard_dir, 'train')
        v_sub = os.path.join(shard_dir, 'val')
        if os.path.exists(t_sub):
            train_paths.append(t_sub)
        if os.path.exists(v_sub):
            val_paths.append(v_sub)

    def _scan_pass(pass_label: str) -> int:
        """Attempt to claim + process every shard once. Returns the number
        of shards still in flight (claimed by someone else, not yet
        committed) at the end of the pass."""
        nonlocal total_train, total_val, committed_by_us, committed_elsewhere
        still_in_flight = 0
        for i in range(num_shards):
            shard_dir = os.path.join(
                shards_root, f'shard-{i:06d}-of-{num_shards:06d}')
            claim_dir = shard_dir + '.claim'

            # Already committed (by us earlier, by another pod, or by a
            # prior run). Record paths and move on.
            if os.path.exists(shard_dir):
                _collect_shard_paths(shard_dir)
                committed_elsewhere += 1
                continue

            # Try to grab the shard. Failure here means either:
            #   (a) another pod is actively working (fresh heartbeat), OR
            #   (b) we raced a stealer and lost.
            # Both are OK; we'll retry in a later pass if configured.
            claimed = _try_claim_shard(
                shard_dir, claim_dir, args.claim_stale_seconds)
            if not claimed:
                still_in_flight += 1
                continue

            # We own claim_dir. Start heartbeat BEFORE the expensive
            # encode call so other pods see us alive the whole time.
            stop_hb = _start_claim_heartbeat(claim_dir, heartbeat_interval)

            try:
                start = i * shard_rows
                end = min(start + shard_rows, len(dataset))
                shard_src = dataset.select(range(start, end))

                t0 = time.time()
                tn, vn = _encode_and_save(
                    shard_src, shard_dir,
                    desc=f'[{pass_label}] shard {i+1}/{num_shards} ({host_tag})')
                dt = time.time() - t0
                total_train += tn
                total_val += vn
                committed_by_us += 1
                _collect_shard_paths(shard_dir)

                done_total = committed_by_us + committed_elsewhere
                # ETA uses only shards *we* did, since committed_elsewhere
                # could have been completed at any wall-clock rate.
                avg = ((time.time() - wall_t0) / committed_by_us
                       if committed_by_us else 0.0)
                remaining_estimate = (num_shards - done_total) * avg
                print(
                    f'[{pass_label}] shard {i+1}/{num_shards} done in {dt:.1f}s '
                    f'(train={tn}, val={vn}, our_avg={avg:.1f}s/shard, '
                    f'done_global={done_total}/{num_shards}, '
                    f'ETA_if_alone={remaining_estimate/3600:.1f}h) -> {shard_dir}')
            finally:
                # Always release: even on exception, we must stop the
                # heartbeat so other pods can eventually take over.
                _release_claim(claim_dir, stop_hb)
        return still_in_flight

    still_in_flight = _scan_pass(pass_label='pass-1')

    # Optional additional passes: wait for in-flight shards owned by other
    # pods to either commit (skipped next pass) or go stale (stolen).
    if still_in_flight > 0 and args.claim_pass_wait_seconds > 0:
        wait_deadline = time.time() + args.claim_pass_wait_seconds
        pass_idx = 2
        print(
            f'[claim] pass-1 complete: we committed {committed_by_us}, '
            f'others had {committed_elsewhere}, {still_in_flight} still in '
            f'flight. Waiting up to {args.claim_pass_wait_seconds:.0f}s for '
            f'them to complete.')
        while time.time() < wait_deadline and still_in_flight > 0:
            # Sleep one heartbeat interval so we don't hammer the FS.
            sleep_for = min(heartbeat_interval, wait_deadline - time.time())
            if sleep_for > 0:
                time.sleep(sleep_for)
            still_in_flight = _scan_pass(pass_label=f'pass-{pass_idx}')
            pass_idx += 1

    if still_in_flight > 0:
        print(
            f'[claim] warning: {still_in_flight} shards still claimed by '
            f'other pods when we finished. They may still be working — '
            f're-run this script when they finish to collect their output '
            f'into the summary. Your own committed shards are safe on disk.')

    # Summary line covering the whole multi-pod picture:
    #   committed_by_us   — we personally encoded these this run.
    #   committed_elsewhere — shards already done when we saw them
    #                         (prior runs or other pods).
    #   still_in_flight   — shards not committed at script exit.
    print(
        f'\n[claim summary] committed_by_us={committed_by_us}, '
        f'committed_elsewhere={committed_elsewhere}, '
        f'still_in_flight={still_in_flight} (of {num_shards} total)')
    print(
        f'               our wall_clock = {(time.time() - wall_t0)/60:.1f} min, '
        f'our throughput = {total_train} train + {total_val} val samples written.')

    # cache_meta at the output_dir root. In a multi-pod scenario this is
    # written by every pod on every run, last-writer-wins. That's OK:
    # all pods compute identical meta_base (same args), and the shard
    # path lists reflect what each pod can see on the shared FS at its
    # exit time — which is only fully accurate for the *last* pod to
    # finish. If you care about an authoritative final summary, re-run
    # the script once after all pods have exited: every shard_dir will
    # exist and the "scan" pass completes in seconds, producing a
    # complete cache_meta.json.
    meta = {
        **meta_base,
        'shard_rows': shard_rows,
        'num_shards': num_shards,
        'train_shard_paths': train_paths,
        'val_shard_paths': val_paths,
        'total_train_samples': total_train,  # only samples WE wrote
        'total_val_samples': total_val,      # only samples WE wrote
        'committed_by_this_pod': committed_by_us,
        'committed_by_others_at_exit': committed_elsewhere,
        'still_in_flight_at_exit': still_in_flight,
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


if __name__ == '__main__':
    main()
