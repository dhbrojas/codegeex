import time

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.codegeexnano import CodeGeeXNanoConfig, CodeGeeXNanoForCausalLM
from utils.datasets import BinaryFileDataset
from utils.info import print_model_information
from utils.lr import wsd_learning_rate_scheduler

STEPS = 5000
GRADIENT_ACCUMULATION_STEPS = 64
MICRO_BATCH_SIZE = 10
MAX_LR = 1e-4
MIN_LR = 1e-6


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def run():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    config = CodeGeeXNanoConfig()
    model = CodeGeeXNanoForCausalLM(config).cuda(device)
    model = torch.compile(model, backend="inductor")

    if rank == 0:
        wandb.init(
            mode="disabled",
            project="codegeexnano-ddp-benchmark",
            config={
                "STEPS": STEPS,
                "GRADIENT_ACCUMULATION_STEPS": GRADIENT_ACCUMULATION_STEPS,
                "MICRO_BATCH_SIZE": MICRO_BATCH_SIZE,
                "MAX_LR": MAX_LR,
                "MIN_LR": MIN_LR,
            },
        )

    # Fake forward pass to compile the model
    print_rank_0("Compiling model...")
    start = time.perf_counter()
    x = torch.randint(
        0, config.vocab_size, (MICRO_BATCH_SIZE, config.max_position_embeddings)
    ).cuda(device)
    y = torch.randint(
        0, config.vocab_size, (MICRO_BATCH_SIZE, config.max_position_embeddings)
    ).cuda(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(x, y)
        loss.backward()
    dist.barrier()
    print_rank_0(f"Compiled model in {time.perf_counter() - start:.2f}s")

    model = DDP(model, device_ids=[device])

    if rank == 0:
        print(f"Running on {dist.get_world_size()} GPUs")
        print(
            f"Training for {STEPS} steps, with {MICRO_BATCH_SIZE*GRADIENT_ACCUMULATION_STEPS*config.max_position_embeddings:,} tokens per step"
        )
        print_model_information(model, config)

    optimizer = AdamW(model.parameters(), lr=MAX_LR)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=wsd_learning_rate_scheduler(
            1000, 4000, MAX_LR, decay_rate=0.95, decay_step=1
        ),
    )

    bench_start = time.perf_counter()
    step_start = time.perf_counter()
    forward_times = np.zeros(STEPS * GRADIENT_ACCUMULATION_STEPS)
    backward_times = np.zeros(STEPS * GRADIENT_ACCUMULATION_STEPS)
    step_times = np.zeros(STEPS)

    tokens_processed = 0
    current_step = 0
    local_grad_acc = GRADIENT_ACCUMULATION_STEPS // dist.get_world_size()
    accumulated_loss = 0

    while current_step < STEPS:
        dataset = BinaryFileDataset(
            "/workspace/data/train.bin", config.max_position_embeddings
        )
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=rank
        )

        for i, (input, target) in enumerate(
            DataLoader(
                dataset,
                batch_size=MICRO_BATCH_SIZE,
                num_workers=0,
                pin_memory=False,
                sampler=sampler,
            )
        ):
            batch_size, sequence_length = input.size()
            assert (
                (batch_size, sequence_length)
                == (
                    MICRO_BATCH_SIZE,
                    config.max_position_embeddings,
                )
            ), f"Expected {(MICRO_BATCH_SIZE, config.max_position_embeddings)}, got {(batch_size, sequence_length)}"
            input, target = input.cuda(device), target.cuda(device)
            forward_start = time.perf_counter()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(input, target)
            forward_times[i] = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            loss.backward()
            backward_times[i] = time.perf_counter() - backward_start
            tokens_processed += MICRO_BATCH_SIZE * config.max_position_embeddings
            total_tokens_processed = tokens_processed * dist.get_world_size()
            accumulated_loss += loss.item()

            if rank == 0:
                wandb.log({"batch/loss": loss.item()})

            if (i + 1) % local_grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                gathered_accumulated_loss = torch.tensor(accumulated_loss).to(device)
                dist.all_reduce(gathered_accumulated_loss, op=dist.ReduceOp.SUM)
                mean_step_loss = (
                    gathered_accumulated_loss.item()
                    / dist.get_world_size()
                    / local_grad_acc
                )
                accumulated_loss = 0

                step_times[current_step] = time.perf_counter() - step_start
                step_start = time.perf_counter()
                tokens_per_second = (
                    MICRO_BATCH_SIZE
                    * GRADIENT_ACCUMULATION_STEPS
                    * config.max_position_embeddings
                ) / (step_times[current_step])
                print_rank_0(
                    f"[STEP] i={current_step} loss={mean_step_loss:.2f} duration={step_times[current_step]:.2f}s tokens={total_tokens_processed:,} throughput={tokens_per_second:,.0f}-tokens/s"
                )
                current_step += 1

                if rank == 0:
                    wandb.log(
                        {
                            "step": current_step,
                            "step/loss": mean_step_loss,
                            "throughput/tokens_per_second": tokens_per_second,
                        }
                    )

    if rank == 0:
        print(f"Average forward pass time: {forward_times.mean() * 1000:.2f}ms")
        print(f"Average backward pass time: {backward_times.mean() * 1000:.2f}ms")
        print(f"Average step time: {step_times.mean():.2f}s")

        total_time_seconds = time.perf_counter() - bench_start

        print(f"Average step time (s): {(total_time_seconds) / STEPS:.2f}")
        print(f"Total time: {total_time_seconds:.2f}s")
        print(
            f"Peak memory usage: {round(torch.cuda.max_memory_allocated()/(1024**3))}/{round(torch.cuda.get_device_properties(0).total_memory/(1024**3))}GiB"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    run()
