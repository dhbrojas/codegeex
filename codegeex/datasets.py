from typing import Any, Dict, Generator, Tuple

import torch
from torch.utils.data import Dataset, DistributedSampler, IterableDataset

from codegeex.tokenizers import Tokenizer


class SampleProcessor(IterableDataset):
    """
    Takes a StreamingDataset over sample of the form:

    >>> {'prefix': str, 'suffix': str, 'middle': str, 'lang': str}

    and transform these samples into batched tensors of shape (micro_batch_size,
    sequence_length) using the tokenizer.
    """

    def __init__(
        self,
        dataset: Dataset,
        sampler: DistributedSampler,
        tokenizer: Tokenizer,
        micro_batch_size: int,
        sequence_length: int,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.iter = iter(self.sampler)
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.lang = tokenizer.special_token_to_id("<|lang|>")
        self.prefix = tokenizer.special_token_to_id("<|prefix|>")
        self.suffix = tokenizer.special_token_to_id("<|suffix|>")
        self.middle = tokenizer.special_token_to_id("<|middle|>")
        self.eos = tokenizer.special_token_to_id("<|eos|>")
        self.buffer = []

        assert all(
            [
                self.lang is not None,
                self.prefix is not None,
                self.suffix is not None,
                self.middle is not None,
                self.eos is not None,
            ]
        )

    def fill_buffer(self) -> bool:
        try:
            sample_idx = next(self.iter)
        except StopIteration:
            return False

        sample = self.dataset[sample_idx]

        prefix, middle, suffix, lang = (
            sample["prefix"],
            sample["middle"],
            sample["suffix"],
            sample["lang"],
        )

        # FIM (SPM)
        encoded_prefix = self.tokenizer.encode(prefix)
        encoded_middle = self.tokenizer.encode(middle)
        encoded_suffix = self.tokenizer.encode(suffix)
        encoded_lang = self.tokenizer.encode(lang)

        self.buffer.extend(
            [self.lang]
            + encoded_lang
            + [self.suffix]
            + encoded_suffix
            + [self.prefix]
            + encoded_prefix
            + [self.middle]
            + encoded_middle
            + [self.eos]
        )

        return True

    def __iter__(self) -> Generator[Tuple[Tuple[Any, ...], Dict[str, Any]], None, None]:
        while True:
            while len(self.buffer) < self.micro_batch_size * self.sequence_length + 1:
                if not self.fill_buffer():
                    return

            inputs = torch.tensor(
                self.buffer[: self.micro_batch_size * self.sequence_length]
            )
            targets = torch.tensor(
                self.buffer[1 : self.micro_batch_size * self.sequence_length + 1]
            )

            inputs = inputs.reshape(
                self.micro_batch_size, self.sequence_length
            ).contiguous()
            targets = targets.reshape(
                self.micro_batch_size, self.sequence_length
            ).contiguous()

            assert inputs.shape == (self.micro_batch_size, self.sequence_length)
            assert targets.shape == (self.micro_batch_size, self.sequence_length)

            self.buffer = self.buffer[self.micro_batch_size * self.sequence_length :]

            yield (inputs, targets), {}
