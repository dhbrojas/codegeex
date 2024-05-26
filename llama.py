import os
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.llama import LlamaConfig, llama_config_to_gpt2_config
from streaming.base import LocalDataset
from tokengeex import Tokenizer as TokenGeeXTokenizer  # type: ignore
from tokenizers import Tokenizer as HFTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from codegeex.config import Config
from codegeex.datasets import (
    FIMNextTokenPredictionProcessor,
)
from codegeex.lr import warmup_stable_decay
from codegeex.tokenizers import (
    Tokenizer,
    WrappedHFTokenizer,
    WrappedTokenGeeXTokenizer,
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
            lr_lambda=warmup_stable_decay(
                W=self.steps * 0.1,
                S=self.steps * 0.9,
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


class GPTForCausalLM(GPTLMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        logits = super().forward(inputs).logits
        batch_size, sequence_length, vocab_size = logits.size()

        targets = targets.view(-1)
        logits = logits.view(batch_size * sequence_length, vocab_size)

        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

        return loss


class Llama470M(BaseConfig):
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
            name=f"llama-470m-{postfix}",
            steps=15000,
            sequence_length=2048,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            padded_vocab_size=padded_vocab_size,
            tokenizer=tokenizer,
        )
        self.mpt = mpt

    def model(self) -> torch.nn.Module:
        model = GPTForCausalLM(
            llama_config_to_gpt2_config(
                LlamaConfig(
                    vocab_size=self.padded_vocab_size,
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    num_key_value_heads=None,
                    hidden_act="silu",
                    max_position_embeddings=self.sequence_length,
                    initializer_range=0.02,
                    rms_norm_eps=0.000001,
                    use_cache=False,
                    pad_token_id=None,
                    bos_token_id=self.tokenizer.special_token_to_id("<|eos|>"),  # type: ignore
                    eos_token_id=self.tokenizer.special_token_to_id("<|eos|>"),  # type: ignore
                    pretraining_tp=1,
                    tie_word_embeddings=False,
                    rope_theta=10000,
                    rope_scaling=None,
                    attention_bias=False,
                    attention_dropout=0,
                    mlp_bias=False,
                )
            )
        )
        return model


class Llama470M32K(Llama470M):
    def __init__(
        self,
        postfix: str,
        tokenizer: Tokenizer,
        mpt: bool = False,
    ):
        super().__init__(
            postfix=f"32k-{postfix}",
            gradient_accumulation_steps=80,
            micro_batch_size=8,
            padded_vocab_size=2**15,
            tokenizer=tokenizer,
            mpt=mpt,
        )


class Llama470M32KTokenGeeXExact(Llama470M32K):
    def __init__(self):
        tokenizer = TokenGeeXTokenizer.from_file(
            "resources/tokengeex/exact-32k-merged.json"
        )
        tokenizer = WrappedTokenGeeXTokenizer(
            tokenizer=tokenizer,
        )
        super().__init__("tokengeex-exact", tokenizer)


class Llama470M32KHFDeepSeekCoder(Llama470M32K):
    def __init__(self):
        tokenizer = HFTokenizer.from_file("resources/hf/deepseek-coder-32k.json")
        tokenizer = WrappedHFTokenizer(
            tokenizer=tokenizer,
        )
        super().__init__("hf-deepseek", tokenizer)
