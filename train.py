import argparse
import importlib
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from codegeex.config import Config
from codegeex.datasets import BinaryFileDataset
from codegeex.parameters import calculate_num_parameters


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[LRScheduler],
    step: Optional[int],
    name: str,
    checkpoint_dir: str,
):
    if dist.get_rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_checkpoint_path = (
            f"{checkpoint_dir}/{name}.safetensors"
            if step is None
            else f"{checkpoint_dir}/{name}-{step}.safetensors"
        )
        save_model(
            model,
            model_checkpoint_path,
            {"step": f"{step}"} if step is not None else {},
        )
        print(f"[CHECKPOINT] kind=model saved_to={model_checkpoint_path}")

        if optimizer is not None and scheduler is not None and step is not None:
            state_checkpoint_path = f"{checkpoint_dir}/{name}-{step}-state.pt"
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                state_checkpoint_path,
            )
            print(f"[CHECKPOINT] kind=state saved_to={state_checkpoint_path}")


def compile_model(
    config: Config, model: torch.nn.Module, device: int
) -> torch.nn.Module:
    model = torch.compile(model, backend="inductor")  # type: ignore
    print_rank_0("Compiling model...")
    start = time.perf_counter()
    (args, kwargs) = config.random_input(torch.device(f"cuda:{device}"))
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(*args, **kwargs)
    loss.backward()
    dist.barrier()
    print_rank_0(f"Compiled model in {time.perf_counter() - start:.2f}s")
    return model


def total_tokens_processed(step: int, config: Config) -> int:
    return (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.sequence_length
        * step
    )


def step_tokens_per_second(step_duration: float, config: Config) -> float:
    return config.tokens_per_batch / step_duration


def run(compile: bool, checkpoint_dir: str, checkpoint_interval: int, configpath: str):
    dist.init_process_group("nccl")
    torch.set_float32_matmul_precision("high")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = (rank + 2) % torch.cuda.device_count()

    modulename, classname = configpath.split(".")

    module = importlib.import_module(modulename)
    config: Config = getattr(module, classname)()

    model = config.model().cuda(device).train()
    optimizer = config.optimizer(model)
    scheduler = config.lr_scheduler(optimizer)

    assert (
        config.gradient_accumulation_steps % world_size == 0
    ), f"{config.gradient_accumulation_steps} must be divisible by {world_size}"

    print_rank_0(f"Running on {dist.get_world_size()} GPUs")
    print_rank_0(
        f"Training for {config.steps} steps, with {config.tokens_per_batch:,} tokens per batch"
    )
    print_rank_0(f"Model parameters: {calculate_num_parameters(model):,}")

    save_checkpoint(model, None, None, 0, classname, checkpoint_dir)

    # Fake forward pass to compile the model
    if compile:
        model = compile_model(config, model, device)
    else:
        print_rank_0("Running in eager mode, pass --compile to use torch.compile()")

    print_rank_0("Initializing DDP")
    model = DDP(model, device_ids=[device])
    device = model.device

    step = 0
    step_start = time.perf_counter()
    epoch = 0
    local_grad_acc = config.gradient_accumulation_steps // dist.get_world_size()
    accumulated_loss = 0

    print_rank_0("Starting training")
    while step < config.steps:
        dataset = BinaryFileDataset(
            "/workspace/datasets/tokenized/tokengeex/exact-32k-merged/train.bin",
            config.sequence_length,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=rank
        )

        for i, (inputs, targets) in enumerate(
            DataLoader(
                dataset,
                batch_size=config.micro_batch_size,
                num_workers=1,
                prefetch_factor=2,
                pin_memory=False,
                sampler=sampler,
                drop_last=True,
            )
        ):
            batch_size, sequence_length = inputs.size()
            assert batch_size == config.micro_batch_size
            assert sequence_length == config.sequence_length
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(inputs, targets)

            # Backward
            loss.backward()
            accumulated_loss += loss.item()

            if (i + 1) % local_grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # TODO: Refactor into reduce for metric
                gathered_accumulated_loss = torch.tensor(accumulated_loss).to(device)
                dist.all_reduce(gathered_accumulated_loss, op=dist.ReduceOp.SUM)
                mean_step_loss = (
                    gathered_accumulated_loss.item()
                    / dist.get_world_size()
                    / local_grad_acc
                )
                accumulated_loss = 0
                step_duration = time.perf_counter() - step_start
                step_start = time.perf_counter()
                step += 1

                print_rank_0(
                    f"[STEP] epoch={epoch} step={step} loss={mean_step_loss:.2f} lr={scheduler.get_last_lr()[0]:.5f} duration={step_duration:.2f}s tokens={total_tokens_processed(step, config):,} tokens_per_sec={step_tokens_per_second(step_duration, config):,.0f}"
                )

                if step % checkpoint_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, step, classname, checkpoint_dir
                    )

        epoch += 1

    save_checkpoint(model, None, None, step, classname, checkpoint_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--config", type=str, default="emma")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    args = parser.parse_args()
    run(args.compile, args.checkpoint_dir, args.checkpoint_interval, args.config)
