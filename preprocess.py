"""
Utility script for tokenizing and preprocessing text data for LLM pre-training.
Expects an array of binary files which contain 0x00 separated UTF-8 encoded
samples.
"""

import argparse
import glob
import logging
import os
import random

import numpy as np
from tokengeex import Tokenizer as TokenGeeXTokenizer

from utils.tokenizers import WrappedTokenGeeXTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and preprocess text data")
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Glob pattern for input text files (e.g. data/*.txt)",
    )
    parser.add_argument(
        "-o", type=str, required=True, help="Output file for tokenized data"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        choices=["tiktoken", "tokengeex"],
        help="Tokenizer to use for tokenization",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocabulary file for the tokenizer",
    )
    args = parser.parse_args()

    if args.tokenizer == "tiktoken":
        raise NotImplementedError("TiktokenTokenizer not implemented yet")
    elif args.tokenizer == "tokengeex":
        tokenizer = WrappedTokenGeeXTokenizer(TokenGeeXTokenizer.from_file(args.vocab))
    else:
        raise ValueError(f"Unknown tokenizer: {args.tokenizer}")

    os.makedirs(os.path.dirname(args.o), exist_ok=True)

    eos = tokenizer.special_token_to_id("<|eos|>")
    assert eos is not None

    data = []

    for filepath in glob.glob(args.i):
        logging.info(f"Processing '{filepath}'")

        with open(filepath, "rb") as f:
            samples = [sample.decode("utf-8") for sample in f.read().split(b"\x00")]

            # Batch samples for more efficient tokenization
            batch_size = 4096
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                allids = tokenizer.encode_batch(batch, include_special_tokens=False)

                for ids in allids:
                    ids = [eos] + ids
                    data.append(np.array(ids, dtype=np.uint32))

    random.shuffle(data)

    data = np.concatenate(data)
    data.tofile(args.o)
