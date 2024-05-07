from typing import Any, Dict, Tuple, Union

import torch
from tokengeex import Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from codegeex.config import Config
from codegeex.lr import wsd_learning_rate_scheduler
from codegeex.models.nano import CodeGeeXNanoConfig, CodeGeeXNanoForCausalLM
from codegeex.tokenizers import WrappedTokenGeeXTokenizer


class Emma130MConfig(Config):
    def __init__(self):
        self.steps = 10000
        self.sequence_length = 2048
        self.micro_batch_size = 32
        self.gradient_accumulation_steps = 32
        self.tokenizer = WrappedTokenGeeXTokenizer(
            tokenizer=Tokenizer.from_file("resources/tokengeex/exact-32k-merged.json"),
        )
        self.padded_vocab_size = 32768
        assert self.padded_vocab_size >= self.tokenizer.vocab_size()

    def model(self) -> torch.nn.Module:
        model = CodeGeeXNanoForCausalLM(
            CodeGeeXNanoConfig(
                hidden_size=1024,
                intermediate_size=4096,
                max_position_embeddings=self.sequence_length,
                num_attention_heads=16,
                norm_eps=1e-6,
                num_layers=16,
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
            lr_lambda=wsd_learning_rate_scheduler(
                W=self.steps * 0.1,
                S=self.steps * 0.9,
                D=self.steps,
                min_lr_scale_factor=0.1,
            ),
        )

    def random_input(
        self, device: torch.device
    ) -> Union[Tuple[Any, ...], Dict[str, Any]]:
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
