import time
from functools import partial

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from models.codegeexnano import CodeGeeXNanoConfig, CodeGeeXNanoModel
from utils.flops import estimate_training_flops, measure_flops
from utils.parameters import calculate_num_parameters

config = CodeGeeXNanoConfig(
    hidden_size=1024,
    num_layers=20,
    intermediate_size=4096,
    max_position_embeddings=2048,
    norm_eps=1e-5,
    num_attention_heads=16,
    vocab_size=32768,
)

model = CodeGeeXNanoModel(config)

num_parameters = calculate_num_parameters(model)
print(f"Number of parameters: {num_parameters/1_000_000}M")
estimated_flops = estimate_training_flops(
    model,
    num_layers=config.num_layers,
    hidden_size=config.hidden_size,
    sequence_length=config.max_position_embeddings,
)
print(f"Estimated training FLOPS: {estimated_flops / 10**12:.2f} TFLOPS")

model.apply(partial(model.init_weights, depth=config.num_layers))

model = model.to("cuda").bfloat16()

print(f"Memory: {round(torch.cuda.memory_allocated()/(1024**3))}GiB")


def random_input(batch_size, sequence_length):
    return torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), device="cuda"
    )


model.eval()
measured_inference_flops = measure_flops(
    model, random_input(4, config.max_position_embeddings)
)
print(f"Measured inference FLOPS: {measured_inference_flops / 10**12:.2f} TFLOPS")

model.train()
measured_training_flops = measure_flops(
    model, random_input(4, config.max_position_embeddings)
)
print(f"Measured training FLOPS: {measured_training_flops / 10**12:.2f} TFLOPS")

output = model(random_input(4, config.max_position_embeddings))

print(f"Output shape: {output.shape}")

# Benchmark the model
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 256
STEPS = 2

model = model.train()
model = model.to("cuda")
model = model.bfloat16()

optimizer = AdamW(model.parameters(), lr=1e-4)


def random_input_generator(n, batch_size):
    for _ in range(n):
        yield (
            random_input(batch_size, config.max_position_embeddings),
            random_input(batch_size, config.max_position_embeddings),
        )


forward_times = np.zeros(STEPS * GRADIENT_ACCUMULATION_STEPS)
backward_times = np.zeros(STEPS * GRADIENT_ACCUMULATION_STEPS)
optimizer_step_times = np.zeros(STEPS)


total_tokens_processed = 0
progress = tqdm(
    random_input_generator(STEPS * GRADIENT_ACCUMULATION_STEPS, MICRO_BATCH_SIZE),
    desc="Training",
    total=STEPS * GRADIENT_ACCUMULATION_STEPS,
)

bench_start = time.perf_counter()

for i, (input, target) in enumerate(progress):
    start = time.perf_counter()
    output = model(input)
    end = time.perf_counter()
    forward_times[i] = end - start

    start = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, config.vocab_size), target.view(-1)
    )
    loss.backward()
    end = time.perf_counter()
    backward_times[i] = end - start

    total_tokens_processed += MICRO_BATCH_SIZE * config.max_position_embeddings
    progress.set_postfix(
        tokens=f"{total_tokens_processed:,}",
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
