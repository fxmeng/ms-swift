# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import torch
from datasets import Dataset as HfDataset, load_from_disk
from tqdm import tqdm
from typing import List, Optional, Union

from swift.arguments import ExportArguments
from swift.dataset import FullEncodePreprocessor
from swift.dataset.packing import calculate_matched_group
from swift.utils import get_logger
from ..train import SwiftSft

logger = get_logger()

PACKING_BATCH_SIZE = 1000


class ExportCachedDataset(SwiftSft):
    args_class = ExportArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], ExportArguments]] = None) -> None:
        super(SwiftSft, self).__init__(args)
        args = self.args
        self.train_msg = {}  # dummy
        if args.cached_dataset and args.packing:
            return
        template_cls = args.template_meta.template_cls
        if template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self._prepare_model_tokenizer(**kwargs)
        self._prepare_template()
        self.template.set_mode(args.template_mode)

    def _post_process_datasets(self, datasets: List) -> List:
        return datasets

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        if self.args.full_encode:
            return self._full_encode_dataset(train_dataset, val_dataset)
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _full_encode_dataset(self, train_dataset, val_dataset):
        """Fully encode all samples: tokenize text and serialize multimodal tensors."""
        template = self.template
        args = self.args

        origin_template_model = template.model
        template.model = None
        preprocessor = FullEncodePreprocessor(template=template)
        batch_size = 100 if args.model_meta.is_multimodal else 1000

        datasets = [train_dataset, val_dataset]
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            dataset = preprocessor(
                dataset,
                num_proc=args.dataset_num_proc,
                load_from_cache_file=args.load_from_cache_file,
                strict=args.strict,
                batch_size=batch_size)
            if len(dataset) == 0:
                dataset = None
            datasets[i] = dataset
        template.model = origin_template_model
        return datasets

    def _compute_packing_from_cache(self):
        """Compute packing groups from an already-cached dataset (packing-only mode)."""
        args = self.args
        for i, raw_path in enumerate(args.cached_dataset):
            path = raw_path.rstrip('/')
            arrow_dataset = load_from_disk(path)
            packing_length = getattr(args, 'packing_length', None) or args.max_length
            packed_idx, packed_length = _compute_packing_groups(arrow_dataset, packing_length)
            logger.info(f'{len(arrow_dataset)} samples -> {len(packed_idx)} packed groups '
                        f'(packing_length={packing_length})')
            packing_ds = HfDataset.from_dict({
                'packed_idx': packed_idx,
                'packed_length': packed_length,
            })
            if args.output_dir is not None:
                if len(args.cached_dataset) == 1:
                    packing_path = args.output_dir
                else:
                    packing_path = os.path.join(args.output_dir, f'packing_{i}')
                os.makedirs(os.path.dirname(packing_path) or '.', exist_ok=True)
            else:
                packing_path = path + '_packing'
            packing_ds.save_to_disk(packing_path)
            logger.info(f'packing metadata: `{packing_path}`')

    def main(self):
        args = self.args

        if args.cached_dataset and args.packing:
            self._compute_packing_from_cache()
            return

        train_dataset, val_dataset = self._prepare_dataset()
        os.makedirs(args.output_dir, exist_ok=True)

        train_data_dir = os.path.join(args.output_dir, 'train')
        val_data_dir = os.path.join(args.output_dir, 'val')
        train_dataset.save_to_disk(train_data_dir)
        logger.info(f'cached_dataset: `{train_data_dir}` ({len(train_dataset)} samples)')
        if val_dataset is not None:
            val_dataset.save_to_disk(val_data_dir)
            logger.info(f'cached_val_dataset: `{val_data_dir}` ({len(val_dataset)} samples)')

        if args.full_encode and args.packing and train_dataset is not None:
            packing_length = getattr(args, 'packing_length', None) or args.max_length
            packed_idx, packed_length = _compute_packing_groups(train_dataset, packing_length)
            logger.info(f'{len(train_dataset)} samples -> {len(packed_idx)} packed groups '
                        f'(packing_length={packing_length})')
            packing_ds = HfDataset.from_dict({
                'packed_idx': packed_idx,
                'packed_length': packed_length,
            })
            packing_path = os.path.join(args.output_dir, 'train_packing')
            packing_ds.save_to_disk(packing_path)
            logger.info(f'packing metadata: `{packing_path}`')

        meta = {
            'full_encode': args.full_encode,
            'packing': args.packing if args.full_encode else False,
            'max_length': args.max_length,
            'model': args.model,
        }
        meta_path = os.path.join(args.output_dir, 'cache_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f'cache metadata: `{meta_path}`')


def _compute_packing_groups(dataset, packing_length: int):
    """Compute bin-packing groups from the dataset's lengths column."""
    lengths_col = dataset['lengths']
    data = []
    for i, length in enumerate(lengths_col):
        effective_length = max(length) if isinstance(length, list) else length
        if effective_length > 0:
            data.append((i, effective_length))

    packed_idx, packed_length = [], []
    remaining = []
    i = 0
    with tqdm(total=len(data), desc='Computing packing groups') as bar:
        while True:
            new = data[i:i + PACKING_BATCH_SIZE]
            remaining += new
            if not remaining:
                break
            i += PACKING_BATCH_SIZE
            done = i >= len(data)
            groups, remaining = calculate_matched_group(remaining, packing_length, is_finished=done)
            bar.update(len(new))
            for g in groups:
                packed_idx.append([x[0] for x in g])
                packed_length.append(sum(x[1] for x in g))
    return packed_idx, packed_length


def export_cached_dataset(args: Optional[Union[List[str], ExportArguments]] = None):
    return ExportCachedDataset(args).main()
