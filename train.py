import argparse
import importlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.modeling_outputs import CausalLMOutputWithPast

from codegeex.config import Config
from codegeex.metrics import DistMetric, MetricsReporter
from codegeex.utils import (
    StepTracker,
    calculate_num_parameters,
    compile_model,
    print_rank_0,
    save_checkpoint,
    step_tokens_per_second,
    total_tokens_processed,
)


def run(
    compile: bool,
    checkpoint_dir: str,
    checkpoint_interval: int,
    config: Config,
    device: torch.device,
):
    if (
        dist.get_rank() == 0
        and os.path.exists(checkpoint_dir)
        and len(os.listdir(checkpoint_dir)) > 0
    ):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} is not empty.")

    metrics = MetricsReporter(checkpoint_dir)
    model = config.model().cuda(device).train()  # type: ignore
    optimizer = config.optimizer(model)
    scheduler = config.lr_scheduler(optimizer)

    assert (
        config.gradient_accumulation_steps % dist.get_world_size() == 0
    ), f"{config.gradient_accumulation_steps} must be divisible by {dist.get_world_size()}"

    print_rank_0(f"Running on {dist.get_world_size()} GPUs")
    print_rank_0(
        f"Training for {config.steps} steps, with {config.tokens_per_batch:,} tokens per batch"
    )
    print_rank_0(f"Model parameters: {calculate_num_parameters(model):,}")

    # Fake forward pass to compile the model
    if compile:
        model = compile_model(config, model, device)
    else:
        print_rank_0("Running in eager mode, pass --compile to use torch.compile()")

    save_checkpoint(model, None, None, 0, checkpoint_dir, overwrite=False)

    print_rank_0("Initializing DDP")
    model = DDP(model, device_ids=[device.index])

    epoch = 0
    step = StepTracker.from_config(config)
    losses = DistMetric(device)

    print_rank_0("Starting training")

    while step.index < config.steps:
        # Epoch
        dataloader = config.dataloader(device)
        for i, (args, kwargs) in enumerate(dataloader):
            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(*args, **kwargs)
                if isinstance(loss, CausalLMOutputWithPast):
                    loss = loss.loss

            # Backward
            loss.backward()
            losses.add(loss.item())

            # Step
            if step.tick():
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                batch_losses = losses.all()
                step_loss = sum(batch_losses) / len(batch_losses)
                lr = scheduler.get_last_lr()[0]

                print_rank_0(
                    f"[STEP] epoch={epoch} step={step.index} loss={step_loss:.2f} lr={lr:.5f} duration={step.duration:.2f}s tokens={total_tokens_processed(step.index, config):,} tokens_per_sec={step_tokens_per_second(step.duration, config):,.0f}"
                )

                for i, batch_loss in enumerate(batch_losses):
                    metrics.record(
                        "batch/loss",
                        batch_loss,
                        step.index * step.local_gradient_accumulation_steps + i,
                    )
                metrics.record("step/loss", step_loss, step.index)
                metrics.record("step/lr", lr, step.index)
                metrics.record("step/duration", step.duration, step.index)
                metrics.record(
                    "step/tokens_per_sec",
                    step_tokens_per_second(step.duration, config),
                    step.index,
                )

                if step.index % checkpoint_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, step.index, checkpoint_dir
                    )

            if step.index >= config.steps:
                break

        epoch += 1

    save_checkpoint(model, None, None, None, checkpoint_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--config", type=str, default="emma")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--skip-n-devices", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.set_float32_matmul_precision("high")

    compile = args.compile
    cfgmodule, cfgname = args.config.rsplit(".", 1)
    config: Config = getattr(importlib.import_module(cfgmodule), cfgname)()
    checkpoint_dir = f"{args.checkpoint_dir}/{config.name}"
    checkpoint_interval = args.checkpoint_interval
    device = torch.device("cuda", dist.get_rank() + args.skip_n_devices)

    run(
        compile,
        checkpoint_dir,
        checkpoint_interval,
        config,
        device,
    )
