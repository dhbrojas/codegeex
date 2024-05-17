"""
Convert a JSONL dataset to a streaming dataset.
"""

import glob
import json
import logging
import os
import random
import sys
from argparse import ArgumentParser

from streaming import MDSWriter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Glob pattern to the input JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    args = parser.parse_args()

    with MDSWriter(
        out=args.output,
        columns={"prefix": "str", "suffix": "str", "middle": "str", "lang": "str"},
        hashes=["sha1"],
        compression="zstd:7",
        size_limit="256MB",
    ) as writer:
        files = glob.glob(args.input)
        random.shuffle(files)
        progress = tqdm(files)
        num_faulty_samples = 0
        num_samples = 0

        for file in progress:
            progress.set_postfix(
                file=os.path.basename(file),
                faulty=f"{num_faulty_samples:,}",
                faulty_rate=f"{(num_faulty_samples / num_samples) if num_samples > 0 else 0:.2f}",
                total=f"{num_samples:,}",
            )

            with open(file, "r") as f:
                for i, line in enumerate(f):
                    if line is None or line == "":
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logging.warning(
                            f"Failed to parse JSONL line in '{file}'. Skipping."
                        )
                        continue

                    if data is None:
                        continue

                    num_samples += 1

                    required_keys = [
                        "code",
                        "prefix",
                        "suffix",
                        "middle",
                        "lang",
                    ]

                    keys = list(data.keys())

                    if not all(key in keys for key in required_keys):
                        logging.warning(
                            f"Missing keys. Expected {required_keys}. Got {keys}. Skipping."
                        )
                        num_faulty_samples += 1
                        sys.exit(1)

                    prefix, suffix, middle, lang, code = (
                        data.get("prefix", None),
                        data.get("suffix", None),
                        data.get("middle", None),
                        data.get("lang", None),
                        data.get("code", None),
                    )
                    prefix = prefix if prefix else code
                    suffix = suffix if suffix else ""
                    middle = middle if middle else ""
                    lang = lang if lang else ""

                    if not all(
                        isinstance(x, str) for x in [prefix, suffix, middle, lang]
                    ):
                        logging.warning(
                            f"Invalid type. Expected 'str' for prefix, suffix, middle, lang. Got {type(prefix)}, {type(suffix)}, {type(middle)}, {type(lang)}. Skipping."
                        )
                        num_faulty_samples += 1
                        sys.exit(1)

                    writer.write(
                        sample={
                            "prefix": prefix,
                            "middle": middle,
                            "suffix": suffix,
                            "lang": lang,
                        }
                    )
