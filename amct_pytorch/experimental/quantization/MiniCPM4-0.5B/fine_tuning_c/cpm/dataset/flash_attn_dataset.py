import torch
from . import ModelBestSDKDatasetBuilder
from ..dragonfly.training_tasks.pretrain_indexed import CudaPrefetcher

class AttentionMaskTensor(torch.Tensor):
    def flatten(self):
        # print("attention mask flatten", self.shape, self[:10, :10])
        flattened_tensor = torch.ones(self.sum(), dtype=self.dtype, device=self.device)
        return torch.Tensor(flattened_tensor)
    

class HgFaceFADataLoader:
    
    def __init__(
            self,
            context, 
            data_path, 
            micro_batch_size, 
            max_seq_length, 
            use_flash_cuda_packer=False, 
            tp_rank=1,
            padding_side='left', 
            ignore_idx=-100,
            tokenizer_path="",
            orig_tokenizer_path=""
        ):
        self.dataloader, _, _ = ModelBestSDKDatasetBuilder(context).build(
            data_path=data_path, 
            micro_batch_size=micro_batch_size,
            max_seq_length=max_seq_length,
            use_flash_cuda_packer=use_flash_cuda_packer,
            tokenizer_path=tokenizer_path,
            orig_tokenizer_path=orig_tokenizer_path)
        self.data_iterator = CudaPrefetcher(self.dataloader, tp_size=1, tp_rank=tp_rank)
        assert padding_side in ['left', 'right'], padding_side
        self.padding_side = padding_side
        self.ignore_idx = int(ignore_idx)
    
    def __iter__(self):
        for data in self.data_iterator.__iter__():
            # print(data.keys())
            self.dataloader.update(data["indexes"])
            batch_size = len(data['cu_seqlens'])-1 if data['cu_seqlens'][-2] != data['cu_seqlens'][-1] else len(data['cu_seqlens'])-2
            max_length = data['max_seqlen']
            attention_mask = data['input_ids'].new_zeros(batch_size, max_length, dtype=torch.bool)
            for i in range(batch_size):
                start_src = data['cu_seqlens'][i].item()
                end_src = data['cu_seqlens'][i+1].item()
                if self.padding_side == 'right':
                    start_tgt = 0
                    end_tgt = end_src - start_src
                elif self.padding_side == 'left':
                    start_tgt = max_length - (end_src-start_src)
                    end_tgt = max_length
                else:
                    raise NotImplementedError(self.padding_side)
                attention_mask[i][start_tgt:end_tgt] = True

            labels = data["target_ids"].new_zeros(data["target_ids"].shape)
            labels[0, 0] = data["input_ids"][0, 0]
            labels[0, 1:] = data["target_ids"][0, :-1]
            fa_attention_mask = AttentionMaskTensor(attention_mask)

            yield {
                "input_ids": data['input_ids'],
                "attention_mask": fa_attention_mask,
                "labels": labels.to(dtype=torch.long),
            }


class HgFaceDataLoader:
    
    def __init__(
            self,
            context, 
            data_path, 
            micro_batch_size, 
            max_seq_length, 
            use_flash_cuda_packer=False, 
            tp_rank=1,
            padding_side='left', 
            ignore_idx=-100,
            tokenizer_path="",
            orig_tokenizer_path=""
        ):
        self.dataloader, _, _ = ModelBestSDKDatasetBuilder(context).build(
            data_path=data_path, 
            micro_batch_size=micro_batch_size,
            max_seq_length=max_seq_length,
            use_flash_cuda_packer=use_flash_cuda_packer,
            tokenizer_path=tokenizer_path,
            orig_tokenizer_path=orig_tokenizer_path)
        self.data_iterator = CudaPrefetcher(self.dataloader, tp_size=1, tp_rank=tp_rank)
        assert padding_side in ['left', 'right'], padding_side
        self.padding_side = padding_side
        self.ignore_idx = int(ignore_idx)
    
    def __iter__(self):
        for data in self.data_iterator.__iter__():
            self.dataloader.update(data["indexes"])
            batch_size = len(data['cu_seqlens'])-1 if data['cu_seqlens'][-2] != data['cu_seqlens'][-1] else len(data['cu_seqlens'])-2
            max_length = data['max_seqlen']
            input_ids = data['input_ids'].new_zeros(batch_size, max_length)
            attention_mask = data['input_ids'].new_zeros(batch_size, max_length, dtype=torch.bool)
            labels = data['target_ids'].new_ones(batch_size, max_length) * self.ignore_idx
            for i in range(batch_size):
                start_src = data['cu_seqlens'][i].item()
                end_src = data['cu_seqlens'][i+1].item()
                if self.padding_side == 'right':
                    start_tgt = 0
                    end_tgt = end_src - start_src
                elif self.padding_side == 'left':
                    start_tgt = max_length - (end_src-start_src)
                    end_tgt = max_length
                else:
                    raise NotImplementedError(self.padding_side)

                input_ids[i][start_tgt:end_tgt] = data['input_ids'][0, start_src:end_src]
                attention_mask[i][start_tgt:end_tgt] = True
                labels[i][start_tgt+1:end_tgt] = data['target_ids'][0, start_src:end_src-1]
                labels[i][start_tgt] = self.ignore_idx
            
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels.to(dtype=torch.long)
            }