# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
from datasets import load_from_disk

from swift.dataset import CachedEncodedDataset, CachedPackingDataset, DatasetSyntax, sample_dataset
from swift.template import update_generation_config_eos_token
from swift.tuner_plugin import tuners_map
from swift.tuners import Swift
from swift.utils import get_logger

logger = get_logger()


def prepare_adapter(args, model, adapters=None):
    if args.tuner_backend == 'unsloth':
        if args.model_meta.is_multimodal:
            from unsloth import FastVisionModel as UnslothModel
        else:
            from unsloth import FastLanguageModel as UnslothModel
        UnslothModel.for_inference(model)
        return model
    if args.tuner_type in tuners_map:
        tuner = tuners_map[args.tuner_type]
    else:
        tuner = Swift
    # compat deploy
    adapters = adapters if adapters is not None else args.adapters
    for adapter in adapters:
        model = tuner.from_pretrained(model, adapter)
    if args.tuner_type == 'bone':
        # Bone has a problem of float32 matmul with bloat16 in `peft==0.14.0`
        model.to(model.dtype)
    return model


def prepare_model_template(args, **kwargs):
    adapters = kwargs.get('adapters')
    model, processor = args.get_model_processor(**kwargs)
    template = args.get_template(processor)
    if model is not None:
        if template.use_model:
            template.model = model
        model = prepare_adapter(args, model, adapters=adapters)
        if args.task_type == 'causal_lm':
            update_generation_config_eos_token(model.generation_config, template)
    return model, template


def _select_dataset(dataset, max_length):
    if 'length' in dataset.column_names and 'lengths' not in dataset.column_names:
        # Compatible with ms-swift 3.x cache_dataset
        dataset = dataset.rename_column('length', 'lengths')
    idxs = [
        i for i, length in enumerate(dataset['lengths'])
        if (max(length) if isinstance(length, list) else length) <= max_length
    ]
    new_dataset = dataset.select(idxs)
    if len(new_dataset) < len(dataset):
        logger.info(f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(new_dataset)}')
    return new_dataset


def _is_fully_encoded(dataset):
    """Check if a cached dataset was created with --full_encode (has pre-tokenized input_ids)."""
    return 'input_ids' in dataset.column_names


def _resolve_cache_path(raw_path):
    """Resolve a cached_dataset path, handling #sample syntax.

    Returns (path, sample_count_or_None).
    """
    if os.path.exists(raw_path):
        return raw_path, None
    path, dataset_sample = DatasetSyntax._safe_split(raw_path, '#', True, 'right')
    return path, dataset_sample


def _try_load_packing(train_path, args):
    """Load pre-computed packing metadata if it exists alongside a train cache."""
    packing_path = train_path.rstrip('/') + '_packing'
    if not os.path.exists(packing_path):
        parent = os.path.dirname(train_path.rstrip('/'))
        packing_path = os.path.join(parent, 'train_packing')
    if os.path.exists(packing_path):
        return load_from_disk(packing_path)
    return None


def get_cached_dataset(args):
    train_datasets, val_datasets = [], []
    random_state = np.random.RandomState(args.data_seed)
    for cached_dataset, datasets in zip([args.cached_dataset, args.cached_val_dataset], [train_datasets, val_datasets]):
        for raw_path in cached_dataset:
            path, dataset_sample = _resolve_cache_path(raw_path)
            arrow_dataset = load_from_disk(path)

            if _is_fully_encoded(arrow_dataset):
                original_len = len(arrow_dataset)
                arrow_dataset = _select_dataset(arrow_dataset, args.max_length)
                if dataset_sample is not None:
                    arrow_dataset = sample_dataset(
                        arrow_dataset, int(dataset_sample), args.dataset_shuffle,
                        random_state=random_state, shuffle_all=True)
                dataset = CachedEncodedDataset(arrow_dataset)
                dataset_modified = len(arrow_dataset) != original_len
                if getattr(args, 'packing', False):
                    if dataset_modified:
                        logger.info(
                            f'Dataset `{path}` was filtered/sampled ({original_len} -> {len(arrow_dataset)}). '
                            'Pre-computed packing metadata will not be used; packing will be computed at training time.')
                    else:
                        packing_arrow = _try_load_packing(path, args)
                        if packing_arrow is not None:
                            dataset = CachedPackingDataset(dataset, packing_arrow)
                        else:
                            logger.warning(
                                f'Packing is enabled but no pre-computed packing metadata found for `{path}`. '
                                'Packing will be computed at training time.')
                datasets.append(dataset)
            else:
                dataset = _select_dataset(arrow_dataset, args.max_length)
                if dataset_sample is not None:
                    dataset = sample_dataset(
                        dataset, int(dataset_sample), args.dataset_shuffle,
                        random_state=random_state, shuffle_all=True)
                datasets.append(dataset)
    return train_datasets, val_datasets
