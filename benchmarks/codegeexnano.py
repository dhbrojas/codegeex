import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, "../models")

from codegeex.models.nano import CodeGeeXNanoConfig, CodeGeeXNanoForCausalLM
from codegeex.utils import print_model_information

STEPS = 2
GRADIENT_ACCUMULATION_STEPS = 64
MICRO_BATCH_SIZE = 8
MAX_LR = 1e-4
MIN_LR = 1e-6


config = CodeGeeXNanoConfig()
model = CodeGeeXNanoForCausalLM(config).cuda()

print_model_information(
    model, config.num_layers, config.hidden_size, config.max_position_embeddings
)

optimizer = AdamW(model.parameters(), lr=MAX_LR)
scheduler = CosineAnnealingLR(optimizer, STEPS, eta_min=MIN_LR)


def random_input_generator(steps, batch_size):
    for _ in range(steps):
        input = torch.randint(
            0, config.vocab_size, (batch_size, config.max_position_embeddings)
        ).cuda()
        target = torch.randint(
            0, config.vocab_size, (batch_size, config.max_position_embeddings)
        ).cuda()
        yield input, target


total_tokens_processed = 0
progress = tqdm(
    random_input_generator(STEPS * GRADIENT_ACCUMULATION_STEPS, MICRO_BATCH_SIZE),
    desc="Training",
    total=STEPS * GRADIENT_ACCUMULATION_STEPS,
)

bench_start = time.perf_counter()

forward_times = np.zeros(STEPS * GRADIENT_ACCUMULATION_STEPS)
backward_times = np.zeros(STEPS * GRADIENT_ACCUMULATION_STEPS)
optimizer_step_times = np.zeros(STEPS)


for i, (input, target) in enumerate(progress):
    start = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        loss = model(input, target)
    end = time.perf_counter()
    forward_times[i] = end - start

    start = time.perf_counter()
    loss.backward()
    end = time.perf_counter()
    backward_times[i] = end - start

    total_tokens_processed += MICRO_BATCH_SIZE * config.max_position_embeddings
    tokens_per_second = total_tokens_processed / (time.perf_counter() - bench_start)
    progress.set_postfix(
        tokens=f"{total_tokens_processed:,}",
        tokens_per_second=f"{tokens_per_second:<8,.0f}",
    )

    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        end = time.perf_counter()
        optimizer_step_times[i // GRADIENT_ACCUMULATION_STEPS] = end - start

bench_end = time.perf_counter()

print(f"Average forward pass time (ms): {forward_times.mean() * 1000:.2f}")
print(f"Average backward pass time (ms): {backward_times.mean() * 1000:.2f}")
print(f"Average optimizer step time (ms): {optimizer_step_times.mean() * 1000:.2f}")

assert total_tokens_processed == (
    MICRO_BATCH_SIZE
    * config.max_position_embeddings
    * STEPS
    * GRADIENT_ACCUMULATION_STEPS
)

total_time_seconds = bench_end - bench_start

print(
    f"Training throughput (tokens/s): {round(total_tokens_processed / total_time_seconds):,}"
)

print(f"Average step time (s): {(total_time_seconds) / STEPS:.2f}")
print(f"Total time: {total_time_seconds:.2f}s")
print(f"Total tokens: {total_tokens_processed:,}")
print(
    f"Peak memory usage: {round(torch.cuda.max_memory_allocated()/(1024**3))}/{round(torch.cuda.get_device_properties(0).total_memory/(1024**3))}GiB"
)
