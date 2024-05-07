from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class Config(ABC):
    def __init__(
        self,
        steps: int,
        sequence_length: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
    ):
        self.steps = steps
        self.sequence_length = sequence_length
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

    @abstractmethod
    def model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        pass

    @abstractmethod
    def random_input(
        self, device: torch.device
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        pass

    @property
    def tokens_per_batch(self) -> int:
        return (
            self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.sequence_length
        )
