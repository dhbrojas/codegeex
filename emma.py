"""
The "emma" configuration uses the exact vocabulary from TokenGeeX.
"""

import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from streaming.base import LocalDataset
from tokengeex import Tokenizer as TokenGeeXTokenizer  # type: ignore
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from codegeex.config import Config
from codegeex.datasets import SampleProcessor
from codegeex.lr import warmup_cosine_decay
from codegeex.models.nano import CodeGeeXNanoConfig, CodeGeeXNanoForCausalLM
from codegeex.tokenizers import WrappedTokenGeeXTokenizer


class Emma800M(Config):
    def __init__(self):
        super().__init__(
            "emma-800m",
            steps=10000,
            sequence_length=2048,
            micro_batch_size=10,
            gradient_accumulation_steps=192,
        )
        tokenizer = TokenGeeXTokenizer.from_file(
            "resources/tokengeex/exact-32k-merged.json"
        )
        tokenizer.add_special_tokens(
            [
                "<|lang|>",
                "<|prefix|>",
                "<|suffix|>",
                "<|middle|>",
                "<|eos|>",
            ]
        )
        self.tokenizer = WrappedTokenGeeXTokenizer(
            tokenizer=tokenizer,
        )
        self.padded_vocab_size = 32768
        assert self.padded_vocab_size >= self.tokenizer.vocab_size()

    def model(self) -> torch.nn.Module:
        model = CodeGeeXNanoForCausalLM(
            CodeGeeXNanoConfig(
                hidden_size=1536,
                intermediate_size=4096,
                max_position_embeddings=self.sequence_length,
                num_attention_heads=16,
                norm_eps=1e-6,
                num_layers=24,
                vocab_size=self.padded_vocab_size,
                rope_percentage=0.25,
                rope_scaling_ratio=1,
                rope_theta=100000,
            ),
        )
        model.init_weights()
        return model

    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return AdamW(
            model.parameters(),
            lr=5e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )

    def lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return LambdaLR(
            optimizer,
            lr_lambda=warmup_cosine_decay(
                W=self.steps * 0.1,
                D=self.steps,
                min_lr_scale_factor=0.1,
            ),
        )

    def random_input(
        self, device: torch.device
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (
            (
                torch.randint(
                    0,
                    self.tokenizer.vocab_size(),
                    (self.micro_batch_size, self.sequence_length),
                    device=device,
                ),
                torch.randint(
                    0,
                    self.tokenizer.vocab_size(),
                    (self.micro_batch_size, self.sequence_length),
                    device=device,
                ),
            ),
            {},
        )

    def dataloader(
        self, device: torch.device
    ) -> Iterable[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        if "CODEGEEX_DEBUG" in os.environ:
            if dist.get_rank() == 0:
                print("Using DEBUG mode, fake data")

            def infinite_random_samples():
                while True:
                    yield self.random_input(device)

            return infinite_random_samples()

        dataset = LocalDataset(local="/workspace/datasets/mds/codegeex")
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
            rank=dist.get_rank() if dist.is_initialized() else 0,
            shuffle=True,
            seed=999,
            drop_last=True,
        )

        def collate_fn(batch: List[Tuple[Any, ...]]):
            assert len(batch) == 1
            return batch[0]

        cpu = device.type == "cpu"

        return DataLoader(
            SampleProcessor(
                dataset,
                sampler=sampler,
                tokenizer=self.tokenizer,
                micro_batch_size=self.micro_batch_size,
                sequence_length=self.sequence_length,
            ),
            num_workers=2 if not cpu else 0,
            prefetch_factor=2 if not cpu else None,
            pin_memory=not cpu,
            pin_memory_device=str(device),
            collate_fn=collate_fn,
        )
