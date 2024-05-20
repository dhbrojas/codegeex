import os
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from safetensors.torch import save_model
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.flop_counter import FlopCounterMode

from codegeex.config import Config


def print_model_information(model, num_layers, hidden_size, sequence_length):
    num_parameters = calculate_num_parameters(model)
    print(f"Number of parameters: {num_parameters/1_000_000}M")
    estimated_flops = estimate_training_flops(
        model,
        num_layers=num_layers,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
    )
    print(f"Estimated training FLOPS: {estimated_flops / 10**12:.2f} TFLOPS")
    print(f"Memory: {round(torch.cuda.memory_allocated()/(1024**3))}GiB")


def total_tokens_processed(step: int, config: Config) -> int:
    return (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.sequence_length
        * step
    )


def step_tokens_per_second(step_duration: float, config: Config) -> float:
    return config.tokens_per_batch / step_duration


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def compile_model(
    config: Config,
    model: torch.nn.Module,
    device: torch.device,
    backend: str = "inductor",
) -> torch.nn.Module:
    model = torch.compile(model, backend=backend)  # type: ignore
    print_rank_0("Compiling model...")
    start = time.perf_counter()
    (args, kwargs) = config.random_input(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(*args, **kwargs)
    loss.backward()
    dist.barrier()
    print_rank_0(f"Compiled model in {time.perf_counter() - start:.2f}s")
    return model


def calculate_num_parameters(
    model: torch.nn.Module, requires_grad: bool = False
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if (p.requires_grad if requires_grad else True)
    )


def flops_per_param(
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    num_params: int,
) -> int:
    flops_per_token = 2 * num_params
    flops_per_seq = flops_per_token * sequence_length
    attn_flops_per_seq = num_layers * 2 * 2 * (hidden_size * (sequence_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_training_flops(
    module: torch.nn.Module,
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int = 1,
):
    """
    Estimated FLOPs for MFU. Uses all parameters which leads to over-estimation
    because not all model parameters actually contribute to this FLOP
    computation. For this reason, the result will be higher by a fixed
    percentage (~10%) compared to the measured FLOPs.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    num_trainable_params = calculate_num_parameters(module, requires_grad=True)
    num_frozen_params = calculate_num_parameters(module, requires_grad=False)

    # Each parameter is used in the forward and backward pass and during the
    # gradient computation.
    train_flops = flops_per_param(
        num_layers, hidden_size, sequence_length, num_trainable_params
    ) + (2 + (1 / gradient_accumulation_steps))

    # Frozen parameters are only used in the forward pass and during the
    # gradient computation.
    frozen_flops = (
        flops_per_param(num_layers, hidden_size, sequence_length, num_frozen_params) * 2
    )

    return train_flops + frozen_flops


def measure_flops(model: torch.nn.Module, *args):
    flop_counter = FlopCounterMode(model, display=False)
    ctx = nullcontext() if model.training else torch.no_grad()
    with ctx, flop_counter:
        y = model(*args)
        if model.training:
            y.sum().backward()
    return flop_counter.get_total_flops()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[LRScheduler],
    step: Optional[int],
    checkpoint_dir: str,
    overwrite: bool = False,
):
    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_checkpoint_path = (
            f"{checkpoint_dir}/model.safetensors"
            if step is None
            else f"{checkpoint_dir}/model-{step}.safetensors"
        )

        if not overwrite and os.path.exists(model_checkpoint_path):
            raise FileExistsError(
                f"Checkpoint already exists at {model_checkpoint_path}. Delete it or set `overwrite=True`."
            )

        save_model(
            model,
            model_checkpoint_path,
            {"step": f"{step}"} if step is not None else {},
        )

        print(f"[CHECKPOINT] kind=model saved_to={model_checkpoint_path}")

        if optimizer is not None and scheduler is not None and step is not None:
            state_checkpoint_path = f"{checkpoint_dir}/state-{step}.pt"

            if not overwrite and os.path.exists(state_checkpoint_path):
                raise FileExistsError(
                    f"Checkpoint already exists at {state_checkpoint_path}. Delete it or set `overwrite=True`."
                )

            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                state_checkpoint_path,
            )
            print(f"[CHECKPOINT] kind=state saved_to={state_checkpoint_path}")


class StepTracker:
    def __init__(self, gradient_accumulation_steps: int, world_size: int):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.local_gradient_accumulation_steps = (
            gradient_accumulation_steps // world_size
        )
        self.world_size = world_size
        self.start = time.perf_counter()
        self.index = 0
        self.accumulated_steps = 0
        self.duration = 0

    @staticmethod
    def from_config(config: Config) -> "StepTracker":
        return StepTracker(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            world_size=dist.get_world_size(),
        )

    def tick(self) -> bool:
        self.accumulated_steps += 1
        if self.accumulated_steps >= self.local_gradient_accumulation_steps:
            self.duration = time.perf_counter() - self.start
            self.start = time.perf_counter()
            self.accumulated_steps = 0
            self.index += 1
            return True
        return False
