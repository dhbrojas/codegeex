import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from streaming.base import LocalDataset
from tokengeex import Tokenizer as TokenGeeXTokenizer  # type: ignore
from tokenizers import Tokenizer as HFTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from codegeex.config import Config
from codegeex.datasets import (
    FIMNextTokenPredictionProcessor,
)
from codegeex.lr import warmup_cosine_decay, warmup_stable_decay
from codegeex.models.nano import CodeGeeXNanoConfig, CodeGeeXNanoForCausalLM
from codegeex.tokenizers import (
    Tokenizer,
    WrappedHFTokenizer,
    WrappedTokenGeeXTokenizer,
    load_custom_tiktoken_tokenizer,
)


class BaseConfig(Config):
    def __init__(
        self,
        name: str,
        steps: int,
        sequence_length: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        padded_vocab_size: int,
        tokenizer: Tokenizer,
        mpt: bool = False,
    ):
        tokenizer.add_special_tokens(
            [
                "<|lang|>",
                "<|prefix|>",
                "<|suffix|>",
                "<|middle|>",
                "<|eos|>",
            ]
        )
        super().__init__(
            name,
            steps=steps,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            padded_vocab_size=padded_vocab_size,
            tokenizer=tokenizer,
        )
        self.mpt = mpt

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
        if self.mpt:
            raise NotImplementedError()
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
            FIMNextTokenPredictionProcessor(
                dataset,
                sampler=sampler,
                tokenizer=self.tokenizer,
                micro_batch_size=self.micro_batch_size,
                sequence_length=self.sequence_length,
            ),
            num_workers=2 if not cpu else 0,
            prefetch_factor=2 if not cpu else None,
            pin_memory=not (cpu or self.mpt),
            pin_memory_device=str(device),
            collate_fn=collate_fn,
        )


class Nano800M(BaseConfig):
    def __init__(
        self,
        postfix: str,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        padded_vocab_size: int,
        tokenizer: Tokenizer,
        mpt: bool = False,
    ):
        super().__init__(
            name=f"nano-800m-{postfix}",
            steps=10000,
            sequence_length=2048,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            padded_vocab_size=padded_vocab_size,
            tokenizer=tokenizer,
        )

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


class Nano800M32K(Nano800M):
    def __init__(
        self,
        postfix: str,
        tokenizer: Tokenizer,
        mpt: bool = False,
    ):
        super().__init__(
            postfix=f"32k-{postfix}",
            gradient_accumulation_steps=192,
            micro_batch_size=10,
            padded_vocab_size=2**15,
            tokenizer=tokenizer,
            mpt=mpt,
        )


class Nano800M32KTokenGeeXExact(Nano800M32K):
    def __init__(self):
        tokenizer = TokenGeeXTokenizer.from_file(
            "resources/tokengeex/exact-32k-merged.json"
        )
        tokenizer = WrappedTokenGeeXTokenizer(
            tokenizer=tokenizer,
        )
        super().__init__("tokengeex-exact", tokenizer)


class Nano800M32KTokenGeeXExactWSD(Nano800M32K):
    def __init__(self):
        tokenizer = TokenGeeXTokenizer.from_file(
            "resources/tokengeex/exact-32k-merged.json"
        )
        tokenizer = WrappedTokenGeeXTokenizer(
            tokenizer=tokenizer,
        )
        super().__init__("tokengeex-exact-wsd", tokenizer)

    def lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        return LambdaLR(
            optimizer,
            warmup_stable_decay(
                W=self.steps * 0.1,
                S=self.steps * 0.9,
                D=self.steps,
                min_lr_scale_factor=0.1,
            ),
        )


class Nano800M32KTokenGeeXGeneral(Nano800M32K):
    def __init__(self):
        tokenizer = TokenGeeXTokenizer.from_file(
            "resources/tokengeex/general-32k-merged.json"
        )
        tokenizer = WrappedTokenGeeXTokenizer(
            tokenizer=tokenizer,
        )
        super().__init__("tokengeex-general", tokenizer)


class Nano800M32KTikTokenGPT4(Nano800M32K):
    def __init__(self):
        tokenizer = load_custom_tiktoken_tokenizer(
            "resources/tiktoken/gpt4-32k.tiktoken",
            "gpt-4",
            [],
        )
        super().__init__("tiktoken-gpt4", tokenizer)


class Nano800M32KHFDeepSeekCoder(Nano800M32K):
    def __init__(self):
        tokenizer = HFTokenizer.from_file("resources/hf/deepseek-coder-32k.json")
        tokenizer = WrappedHFTokenizer(
            tokenizer=tokenizer,
        )
        super().__init__("hf-deepseek", tokenizer)
