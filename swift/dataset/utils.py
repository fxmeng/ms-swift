# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import numpy as np
import os
import tempfile
import torch
from datasets import Dataset as HfDataset
from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Union

from swift.template import MaxLengthError, Template
from swift.utils import get_logger
from .preprocessor import RowPreprocessor

logger = get_logger()


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
    """Wraps a fully-encoded Arrow dataset, reconstructing tensors from bytes on access."""

    def __init__(self, arrow_dataset: HfDataset):
        self.dataset = arrow_dataset

    def __len__(self) -> int:
        return len(self.dataset)

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

        if row.get('pixel_values_bytes') is not None:
            shape = row['pixel_values_shape']
            arr = np.frombuffer(row['pixel_values_bytes'], dtype=np.float16).reshape(shape)
            result['pixel_values'] = torch.from_numpy(arr.copy())
        if row.get('image_grid_thw') is not None:
            result['image_grid_thw'] = torch.tensor(row['image_grid_thw'])

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
