"""
Evaluate a model on HumanEvalX.
"""

import argparse
import glob
import importlib
import json
import os
import re
import time

import requests
import torch
from rich.console import Console
from rich.syntax import Syntax
from safetensors.torch import load_file

from codegeex.config import Config
from codegeex.metrics import MetricsReporter


def load_humanevalx():
    data = {}
    for file in glob.glob("resources/humanevalx/*.jsonl"):
        basename = os.path.basename(file).replace(".jsonl", "")
        data[basename] = []
        for line in open(file):
            data[basename].append(json.loads(line))
    return data


def run(
    checkpoint_dir: str,
    pattern: str,
    batch_size: int,
    config: Config,
):
    device = torch.device("cpu")
    model = config.model().to(device).eval()  # type: ignore
    metrics = MetricsReporter(checkpoint_dir)
    console = Console()

    humanevalx = load_humanevalx()
    # Stop when a dedent to the top-level is matched.
    stop_regex = re.compile(r"(?m)^(?!\s).+$")

    checkpoints = glob.glob(f"{checkpoint_dir}/{pattern}")
    checkpoints.sort()

    console.print(f"--- Found {len(checkpoints)} checkpoints")
    for filepath in checkpoints:
        # Fetch step from the filepath (model-{step}.safetensors)
        regex = re.compile(r"model-(\d+)\.safetensors")
        search = regex.search(filepath)
        if search is None:
            console.print(f"Could not parse step from '{filepath}', skipping")
            continue
        step = int(search.group(1))

        console.print(f"--- Loading checkpoint '{filepath}' step={step}")

        state = load_file(filepath)
        # Rename the keys of "state" if they start with "module._orig_mod".
        # Happens when using torch.compile()
        for key in list(state.keys()):
            state[key.replace("module._orig_mod.", "")] = state.pop(key)
        missing, expected = model.load_state_dict(state)
        assert (
            len(missing) + len(expected) == 0
        ), f"{len(missing)} missing, {len(expected)} unexpected"

        tokenizer = config.tokenizer
        eos_token = int(tokenizer.special_token_to_id("<|eos|>"))  # type: ignore
        lang_token = tokenizer.special_token_to_id("<|lang|>")
        prefix_token = tokenizer.special_token_to_id("<|prefix|>")
        suffix_token = tokenizer.special_token_to_id("<|suffix|>")
        middle_token = tokenizer.special_token_to_id("<|middle|>")

        print("----" * 8)

        with torch.no_grad():
            assert hasattr(model, "generate")

            for lang, problems in humanevalx.items():
                solutions = [
                    {
                        "stop_reason": "length",
                        "completion": "",
                        "decode_time": 0,
                    }
                    for _ in range(len(problems))
                ]
                num_problems = len(problems)
                num_problems_solved = 0

                for i in range(0, num_problems, batch_size):
                    # The last batch might be smaller than `batch_size`
                    batch_end = min(num_problems, i + batch_size)
                    batch_problems = problems[i:batch_end]

                    batch = torch.full(
                        (len(batch_problems), config.sequence_length),
                        eos_token,
                        dtype=torch.long,
                    )

                    for i, sample in enumerate(batch_problems):
                        taskid, prompt, test = (
                            sample["task_id"],
                            sample["prompt"],
                            sample["test"],
                        )
                        # Remove trailing '\n'
                        prompt = prompt.rstrip("\n")

                        ids = (
                            [lang_token]
                            + tokenizer.encode("Python")
                            + [suffix_token]
                            + tokenizer.encode("")
                            + [prefix_token]
                            + tokenizer.encode(prompt)
                            + [middle_token]
                        )

                        batch[i, : len(ids)] = torch.tensor(ids)

                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            for i in range(128):
                                output = model.generate(
                                    batch.to(device),
                                )
                                id = output[0].item()

                                if id >= tokenizer.vocab_size() or id == eos:
                                    stop_reason = "eos"
                                    break

                                ids.append(id)
                                value = tokenizer.decode(
                                    [id], decode_special_tokens=True
                                )  # type: ignore

                                if stop_regex.search(completion + value):
                                    stop_reason = f"regex({repr(value)})"
                                    break

                                completion += value

                        decode_time = time.perf_counter() - decode_start

                    # --------

                    res = requests.post(
                        "https://humanevalx.rojasdiego.com/v1/execute",
                        json={
                            "programs": [
                                {
                                    "runtime": "python3",
                                    "code": prompt + completion + test,
                                    "timeoutSecs": 15,
                                }
                            ]
                        },
                    )

                    success = None
                    if res.status_code != 200:
                        console.print(f"Error: HTTP {res.status_code}, body={res.text}")
                    else:
                        try:
                            body = res.json()
                            success = body["results"][0]["success"]

                            if success:
                                num_problems_solved += 1
                        except Exception:
                            pass

                    code: str = prompt + completion
                    syntax = Syntax(
                        code.strip(),
                        "python",
                        line_numbers=True,
                    )

                    console.print(syntax)
                    console.print(
                        f"--- [PROBLEM {taskid} stop={stop_reason} pass={success} decode_time={decode_time:.2}s total={num_problems_solved}/{num_problems} ({num_problems_solved / num_problems:.2%})] ------"
                    )

        # Evaluate
        metrics.record("step/humanevalx/python", 0.0, step)

    metrics.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="emma")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--pattern", type=str, default="*.safetensors")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfgmodule, cfgname = args.config.rsplit(".", 1)
    config: Config = getattr(importlib.import_module(cfgmodule), cfgname)()
    checkpoint_dir = f"{args.checkpoint_dir}/{config.name}"
    pattern = args.pattern
    batch_size = args.batch_size

    run(
        checkpoint_dir,
        pattern,
        batch_size,
        config,
    )
