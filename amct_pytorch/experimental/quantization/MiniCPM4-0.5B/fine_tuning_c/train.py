# -*- coding: utf-8 -*-
import os
import math
import json
import random
import logging
from typing import Optional, Dict
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]-[%(asctime)s] (%(pathname)s): %(message)s')
logger = logging.getLogger(__name__)

import torch
import bmtrain

from transformers.optimization import get_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, HfArgumentParser, TrainerCallback

from model import *

from cpm.dataset.flash_attn_dataset import HgFaceFADataLoader
from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/")
    group_size: Optional[int] = 128
    vec_size: Optional[int] = 4
    vec_bit: Optional[int] = 2
    proj_type: Optional[str] = "bypass"
    use_hadamard: Optional[bool] = True
    orig_tokenizer_path: Optional[str] = field(default="/data/groups/QY_LLM_Core/HQData250626/modelbestdata/new_73k.model")


@dataclass
class DataArguments:
    train_data_config_path: str = field(
        default=None,
        metadata={"help": "Path to the training config file. Will overwrite 'train_data_path'"},
    )
   

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    output_dir: str = field(default="outputs")
    seed: int = field(default=3014)
    auto_resume: bool = field(default=False)
    init_from_pretrained: bool = field(default=False)
    stable_ratio: float = field(default=0.4)


def init_bmt():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE","1"))
    pipe_size = -1
    bmtrain.global_var.config["pipe_size"] = pipe_size if pipe_size > 0 else 1
    bmtrain.global_var.config["pipe_enabled"] = pipe_size > 0
    bmtrain.global_var.config["local_rank"] = local_rank
    bmtrain.global_var.config["local_size"] = local_size
    bmtrain.global_var.config["rank"] = rank
    bmtrain.global_var.config["world_size"] = world_size
    bmtrain.global_var.config["tp_size"] = 1
    bmtrain.global_var.config["topology"] = bmtrain.init.topology(bmtrain.global_var.config)
    context = DatasetContext(
        rank=rank,
        world_size=world_size,
        dataset_checkpoint_path='/home/wanghaoyu2/extremeLowBit/modelbestdata',
    )
    return context


class WSDTrainer(Trainer):

    def __init__(self, *args, fa_train_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.fa_train_loader = fa_train_loader

    def get_train_dataloader(self):
        return self.fa_train_loader
    
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        super().log(logs=logs, start_time=start_time)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            lr_scheduler_kwargs = self.args.lr_scheduler_kwargs if self.args.lr_scheduler_kwargs is not None else {}
            if self.args.lr_scheduler_type == "warmup_stable_decay":
                num_stable_steps = math.ceil(num_training_steps * self.args.stable_ratio)
                num_decay_steps = math.ceil(num_training_steps * (1-self.args.warmup_ratio-self.args.stable_ratio))
                lr_scheduler_kwargs["num_stable_steps"] = num_stable_steps
                lr_scheduler_kwargs["num_decay_steps"] = num_decay_steps
            # print("lr", lr_scheduler_kwargs)
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler


if __name__ == "__main__":

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    context = init_bmt()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)

    model.to(dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    for n, p in model.named_parameters():
        if "lm_head" in n or "embed_tokens" in n:
            p.requires_grad = False
        print(n, p.requires_grad)
    
    total_param = 0
    if bmtrain.global_var.config["local_rank"] == 0: 
        for n, p in model.named_parameters():
            if p.requires_grad:
                total_param += p.numel()
        print(f"trainable param: {total_param/1e9:.4f}B")

    tasks_list = []
    parse_path = []
    with open(data_args.train_data_config_path, "r") as f:
        data_config = json.load(f)
        for c in data_config:
            parse_path.append(c["ratio"])
            parse_path.append(c["path"])
            tasks_list.append(c["name"])
    assert len(parse_path) >= 2, "Invalid data config for modelbest_sdk"
    
    train_dataloader = HgFaceFADataLoader(
        context=context, 
        data_path=parse_path,
        micro_batch_size=training_args.per_device_train_batch_size, 
        max_seq_length=training_args.model_max_length, 
        use_flash_cuda_packer=True,
        padding_side=tokenizer.padding_side,
        tp_rank=context.tp_rank,
        tokenizer_path=model_args.model_name_or_path,
        orig_tokenizer_path=model_args.orig_tokenizer_path
    )

    trainer = WSDTrainer(
        model=model,
        args=training_args,
        fa_train_loader=train_dataloader,
        tokenizer=tokenizer
    )

    if training_args.auto_resume:
        last_ckpt_step = 0
        for ckpt_dir in os.listdir(training_args.output_dir):
            if "checkpoint-" in ckpt_dir and os.path.isdir(os.path.join(training_args.output_dir, ckpt_dir)):
                ckpt_step = int(ckpt_dir.split("-")[1])
                if ckpt_step > last_ckpt_step:
                    last_ckpt_step = ckpt_step
        resume_ckpt = os.path.join(training_args.output_dir, f"checkpoint-{last_ckpt_step}")
    else:
        resume_ckpt = None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    tokenizer.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
