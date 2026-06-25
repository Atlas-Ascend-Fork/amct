
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
from typing import Any, Callable, Iterable, List, Optional, Type, Union

import bmtrain as bmt
from modelbest_sdk.dataset.batch_packer.batch_packer_factory import MEGATRON_BATCH_PACKER, CPM_FLASH_ATTN_BATCH_PACKER
from modelbest_sdk.dataset.modelbest_dataloader import ModelbestDataloader
from modelbest_sdk.dataset.sampler.sampler_factory import WEIGHTED_MEGATRON_SAMPLER
from modelbest_sdk.dataset.segment.segment_factory import FIXED_LENGTH_SEGMENT, CONDITIONAL_FIXED_LENGTH_SEGMENT
from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext
from modelbest_sdk.dataset.thrift_wrapper.dataset_checkpoint import DatasetInfo, DatasetInfoList


class ModelBestSDKDatasetBuilder(object):
    """Builder class for the ModelBestSDKDatasetBuilder

    Args:
        config (DatasetContext): The config object which informs dataset creation
    """

    def __init__(
        self,
        config: DatasetContext,
    ):
        self.config = config

    def build(
            self, data_path, micro_batch_size, 
            max_seq_length, use_flash_cuda_packer=False, 
            tokenizer_path="", orig_tokenizer_path="") -> List[Optional[ModelbestDataloader]]:
        if data_path is not None:
            dataloader = self._build(data_path, micro_batch_size, max_seq_length, use_flash_cuda_packer, tokenizer_path, orig_tokenizer_path)
            return dataloader, None, None
        else:
            raise ValueError("No data path provided")
        

    def _build(self, data_path, micro_batch_size, max_seq_length, use_flash_cuda_packer=False, tokenizer_path="", orig_tokenizer_path="") -> ModelbestDataloader:
        dataset_info_list = self.parse_data_path(data_path)
        batcher_type = CPM_FLASH_ATTN_BATCH_PACKER if use_flash_cuda_packer else MEGATRON_BATCH_PACKER
        dataloader = ModelbestDataloader(
            self.config, 
            dataset_info_list, 
            batch_size=micro_batch_size, 
            max_len=max_seq_length, 
            chunk_size=64, 
            cuda_prefetch=False, 
            segment_type=CONDITIONAL_FIXED_LENGTH_SEGMENT, 
            sampler_type=WEIGHTED_MEGATRON_SAMPLER, 
            batch_packer_type=batcher_type, 
            dp_group=None, 
            clear_sampler_state=False,
            tokenizer_path=tokenizer_path,
            orig_tokenizer_path=orig_tokenizer_path
        )
        return dataloader

    def normalize(self, weights):
        total = sum(weights)
        return [w / total for w in weights]

    def parse_data_path(self, data_path):
        weights, paths = zip(*[(float(data_path[i]), data_path[i + 1].strip()) for i in range(0, len(data_path), 2)])
        weights = self.normalize(weights)
        dataset_info_list = [DatasetInfo(path=path, weight=weight) for path, weight in zip(paths, weights)]
        return DatasetInfoList(dataset_info_list)

