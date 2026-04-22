# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import json
import numpy as np
import os
import tempfile
import torch
from datasets import Dataset as HfDataset
from io import BytesIO
from modelscope.hub.utils.utils import get_cache_dir
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Union

from swift.template import MaxLengthError, Template
from swift.utils import get_logger
from .preprocessor import RowPreprocessor

logger = get_logger()


def _decode_rgb_image_from_bytes(raw_bytes: bytes) -> Image.Image:
    """Decode raw image bytes into an RGB PIL image.

    MUST stay bit-equivalent to ``examples/train/cached_dataset/encode_pretrain_vl.py``'s
    ``_decode_rgb_image``: both sides of the cache pipeline use this to convert
    raw bytes into the RGB PIL image that the image_processor consumes. The
    palette+transparency branch composites onto a white background instead of
    PIL's default (transparent → arbitrary palette color), which is the exact
    conversion done at encode time.
    """
    img = Image.open(BytesIO(raw_bytes))
    mode = img.mode
    if mode == 'RGB':
        img.load()
        return img
    if mode == 'RGBA' or (mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')


def _find_cache_meta(cache_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Locate cache_meta.json for a given Arrow-dataset path.

    Looks in the dataset dir itself first, then walks up two parents. This
    matches how encode_pretrain_vl.py drops meta files:

        {output_dir}/cache_meta.json                     (cache root)
        {output_dir}/shards/shard-.../cache_meta.json    (per-shard)
        {output_dir}/shards/shard-.../{train,val}/       (actual Arrow dirs)
    """
    if not cache_path:
        return None
    candidates: List[str] = []
    p = os.path.abspath(cache_path.rstrip('/'))
    candidates.append(os.path.join(p, 'cache_meta.json'))
    parent = os.path.dirname(p)
    if parent:
        candidates.append(os.path.join(parent, 'cache_meta.json'))
    gparent = os.path.dirname(parent) if parent else ''
    if gparent:
        candidates.append(os.path.join(gparent, 'cache_meta.json'))
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f'failed to read {candidate}: {e}')
                return None
    return None


def sample_dataset(
        dataset: HfDataset,
        dataset_sample: Optional[int],
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
        shuffle_all: bool = False,  # For compatibility, this defaults to False.
) -> HfDataset:
    """Sample dataset by a dataset_sample number
    Args:
        dataset: The dataset instance, iterable dataset is not supported
        dataset_sample: The sample number
        shuffle: Whether to perform random sampling on non-streaming datasets
        random_state: The random state
    Returns:
        The sampled dataset
    """
    if dataset_sample is None:
        return dataset

    n_repeat_sample = dataset_sample // len(dataset)
    n_remain_sample = dataset_sample % len(dataset)
    if n_repeat_sample >= 1 and n_remain_sample >= 1:
        logger.warning(f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
                       'repeated sampling will be performed.')
    idx = np.tile(range(len(dataset)), n_repeat_sample)
    if random_state is None:
        random_state = np.random.RandomState()
    if n_remain_sample >= 1:
        if shuffle:
            idx_remain = random_state.permutation(len(dataset))[:n_remain_sample]
        else:
            idx_remain = np.arange(n_remain_sample)
        idx = np.concatenate([idx, idx_remain])
    if n_repeat_sample >= 1 and shuffle and shuffle_all:
        random_state.shuffle(idx)
    dataset = dataset.select(idx)
    return dataset


class LazyLLMDataset(Dataset):
    """This class if used to lazy tokenize the dataset, and skips bad ones when training"""

    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 *,
                 n_try_fetch: int = 10,
                 strict: bool = False,
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 traceback_limit: int = 10) -> None:
        self.dataset = dataset
        self.encode_func = encode_func

        n_try_fetch = 1 if strict else min(n_try_fetch, len(self.dataset))
        assert n_try_fetch >= 1
        self.strict = strict
        self.n_try_fetch = n_try_fetch

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

        self.traceback_limit = traceback_limit
        self._traceback_counter = 0
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, str):
            return self.dataset[idx]
        for i in range(self.n_try_fetch):
            if i > 0:
                idx = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            data = self.dataset[idx]
            try:
                return self.encode_func(data, return_length=True)
            except Exception as e:
                if self.strict:
                    logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                if isinstance(e, MaxLengthError):
                    continue
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆👆👆There are errors in the template.encode, '
                                   'and another piece of data will be randomly selected.')
                    self._traceback_counter += 1

        raise ValueError('Failed to retrieve the dataset. You can avoid this issue by increasing `max_length` or '
                         'modifying the `truncation_strategy`.')

    def __len__(self) -> int:
        return len(self.dataset)


class EncodePreprocessor(RowPreprocessor):

    def __init__(self, template: 'Template'):
        super().__init__()
        self.template = template

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.template.encode(row, return_length=True)


class AddLengthPreprocessor(EncodePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = super().preprocess(row)
        row['lengths'] = encoded['lengths']
        return row


class FullEncodePreprocessor(EncodePreprocessor):
    """Fully encode and serialize multimodal tensors for Arrow storage.

    Produces a flat dict with tokenized text fields and binary-serialized
    tensor fields that can be stored directly in Arrow format. Training can
    then reconstruct tensors from bytes with zero image/video processing.
    """

    _TENSOR_FIELDS = [
        ('pixel_values', 'pixel_values_bytes', 'pixel_values_shape'),
        ('pixel_values_videos', 'pixel_values_videos_bytes', 'pixel_values_videos_shape'),
    ]
    _THW_FIELDS = ['image_grid_thw', 'video_grid_thw']

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        encoded = super().preprocess(row)
        if encoded is None:
            return None
        result: Dict[str, Any] = {
            'input_ids': encoded['input_ids'],
            'labels': encoded['labels'],
            'lengths': encoded['lengths'],
            'loss_scale': encoded.get('loss_scale'),
        }

        for tensor_key, bytes_key, shape_key in self._TENSOR_FIELDS:
            val = encoded.get(tensor_key)
            if val is not None and isinstance(val, torch.Tensor):
                t = val.to(torch.float16)
                result[bytes_key] = t.numpy().tobytes()
                result[shape_key] = list(t.shape)
            else:
                result[bytes_key] = None
                result[shape_key] = None

        for thw_key in self._THW_FIELDS:
            val = encoded.get(thw_key)
            if val is not None:
                result[thw_key] = val.tolist() if isinstance(val, torch.Tensor) else val
            else:
                result[thw_key] = None

        return result


class CachedEncodedDataset(Dataset):
    """Wraps a fully-encoded Arrow dataset, reconstructing tensors on access.

    Two storage modes are supported transparently, picked per-row at read
    time based on which column is populated:

      pixel_values mode (legacy)
        Row has ``pixel_values_bytes`` + ``pixel_values_shape`` → rebuild
        the fp16 tensor via ``np.frombuffer``. Zero image-processor work at
        training time.

      image_bytes mode (cache_meta.store_mode == 'image_bytes')
        Row has ``image_bytes`` (list of raw JPEG/PNG bytes). Training runs
        the image_processor on demand to reproduce ``pixel_values`` with
        the same ``max_pixels`` / ``min_pixels`` used at encode time (read
        from the sibling ``cache_meta.json``). The ``image_grid_thw`` is
        still read verbatim from the cache (authoritative; computed at
        encode time) so ``n_pad`` in ``input_ids`` always matches the
        tensor shape.

    Training-side per-batch outputs are bit-identical between the two modes
    given the same source data and the same ``max_pixels`` / ``min_pixels``.
    The trade-off is storage (image_bytes ~20-50x smaller) vs. per-batch
    CPU cost (pixel_values has none; image_bytes runs image_processor).
    """

    def __init__(self, arrow_dataset: HfDataset, cache_path: Optional[str] = None):
        self.dataset = arrow_dataset
        self.cache_path = cache_path
        self._meta = _find_cache_meta(cache_path) or {}
        self._store_mode = self._meta.get('store_mode') or 'pixel_values'
        self._has_image_bytes_col = 'image_bytes' in arrow_dataset.column_names
        if self._has_image_bytes_col and not self._meta:
            # A cache built in image_bytes mode without a discoverable
            # cache_meta.json is unusable: we won't know which model's
            # image_processor to load, nor max/min_pixels. Fail early.
            logger.warning(
                f"cached_dataset at `{cache_path}` has an `image_bytes` column but no "
                f"cache_meta.json was found; image processing will be attempted with default "
                f"image_processor settings and may produce DIFFERENT pixel_values from what "
                f"the encode step used. Rebuild the cache or drop a cache_meta.json next to it.")
        # image_processor is loaded lazily, per-worker, on first image_bytes
        # row. This matters because dataloader workers are forked and we
        # don't want to pay the load cost in the parent.
        self._image_processor = None

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_image_processor(self):
        if self._image_processor is not None:
            return self._image_processor
        from transformers import AutoProcessor
        model_path = self._meta.get('model')
        if not model_path:
            raise RuntimeError(
                f"cached_dataset at `{self.cache_path}` is in image_bytes mode but "
                f"cache_meta.json doesn't record the model path. Cannot load image_processor.")
        processor_kwargs = {}
        for k in ('max_pixels', 'min_pixels'):
            v = self._meta.get(k)
            if v is not None:
                processor_kwargs[k] = v
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, **processor_kwargs)
        # Also set on the image_processor instance directly so any internal
        # call paths that read these attributes pick them up (mirrors what
        # encode_pretrain_vl.py does in _get_processor).
        for k, v in processor_kwargs.items():
            if hasattr(processor.image_processor, k):
                setattr(processor.image_processor, k, v)
        self._image_processor = processor.image_processor
        return self._image_processor

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.dataset[idx]
        row = self.dataset[idx]
        result: Dict[str, Any] = {
            'input_ids': row['input_ids'],
            'labels': row['labels'],
            'length': max(row['lengths']) if isinstance(row['lengths'], list) else row['lengths'],
        }
        if 'loss_scale' in row and row['loss_scale'] is not None:
            result['loss_scale'] = row['loss_scale']

        # ---- images ----
        # Prefer new image_bytes path; fall back to legacy pixel_values_bytes.
        img_bytes_list = row.get('image_bytes') if self._has_image_bytes_col else None
        if img_bytes_list:
            pil_images = [_decode_rgb_image_from_bytes(b) for b in img_bytes_list if b is not None]
            if pil_images:
                processor = self._get_image_processor()
                image_inputs = processor(images=pil_images, return_tensors='pt')
                # fp16 cast here matches what pixel_values mode stored on disk,
                # so downstream code (collator + model forward) sees the same
                # dtype regardless of storage mode.
                result['pixel_values'] = image_inputs['pixel_values'].to(torch.float16)
                # Trust encode-time image_grid_thw over what we just recomputed:
                # it's the authoritative value that n_pad in input_ids was
                # derived from. In the normal case they match; if the user
                # changed max/min_pixels between encode and train, taking the
                # stored value keeps n_pad / pixel_values consistent.
                if row.get('image_grid_thw') is not None:
                    result['image_grid_thw'] = torch.tensor(row['image_grid_thw'])
                else:
                    result['image_grid_thw'] = image_inputs['image_grid_thw']
        elif row.get('pixel_values_bytes') is not None:
            shape = row['pixel_values_shape']
            arr = np.frombuffer(row['pixel_values_bytes'], dtype=np.float16).reshape(shape)
            result['pixel_values'] = torch.from_numpy(arr.copy())
            if row.get('image_grid_thw') is not None:
                result['image_grid_thw'] = torch.tensor(row['image_grid_thw'])
        elif row.get('image_grid_thw') is not None:
            # Grid present but no image data: unusual, but preserve it so
            # the model doesn't crash on missing pixel_values for images.
            result['image_grid_thw'] = torch.tensor(row['image_grid_thw'])

        # ---- videos (unchanged) ----
        if row.get('pixel_values_videos_bytes') is not None:
            shape = row['pixel_values_videos_shape']
            arr = np.frombuffer(row['pixel_values_videos_bytes'], dtype=np.float16).reshape(shape)
            result['pixel_values_videos'] = torch.from_numpy(arr.copy())
        if row.get('video_grid_thw') is not None:
            result['video_grid_thw'] = torch.tensor(row['video_grid_thw'])

        return result


class CachedPackingDataset(Dataset):
    """Packing dataset that uses pre-computed packing groups from Arrow."""

    def __init__(self, cached_dataset: 'CachedEncodedDataset', packing_arrow: HfDataset):
        self.dataset = cached_dataset
        self.packed_idx: List[List[int]] = packing_arrow['packed_idx']
        self.packed_length: List[int] = packing_arrow['packed_length']
        logger.info(f'CachedPackingDataset: {len(cached_dataset)} samples -> {len(self.packed_idx)} packed groups')

    def __len__(self) -> int:
        return len(self.packed_idx)

    def __getitem__(self, index) -> List[Dict[str, Any]]:
        indices = self.packed_idx[index]
        return [self.dataset[i] for i in indices]


TEMP_DIR_POOL = {}


def get_temporary_cache_files_directory(prefix=None):
    if prefix is None:
        import datasets.config
        prefix = datasets.config.TEMP_CACHE_DIR_PREFIX
    if prefix in TEMP_DIR_POOL:
        TEMP_DIR = TEMP_DIR_POOL[prefix]
    else:
        tmp_dir = os.path.join(get_cache_dir(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        kwargs = {}
        parameters = inspect.signature(tempfile.TemporaryDirectory.__init__).parameters
        if 'ignore_cleanup_errors' in parameters:
            kwargs['ignore_cleanup_errors'] = True
        TEMP_DIR = tempfile.TemporaryDirectory(prefix=prefix, dir=tmp_dir, **kwargs)
        logger.info(f'create tmp_dir: {TEMP_DIR.name}')
        TEMP_DIR_POOL[prefix] = TEMP_DIR

    return TEMP_DIR.name
