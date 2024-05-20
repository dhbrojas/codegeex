from typing import Any, Callable, Dict, Generator, List, Tuple

import torch
from tokengeex import Tokenizer as TokenGeeXTokenizer
from torch.utils.data import Dataset, DistributedSampler, IterableDataset

from codegeex.tokenizers import Tokenizer


class FIMNextTokenPredictionProcessor(IterableDataset):
    """
    Takes a Dataset over sample of the form:

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


class FIMManyTokenPredictionProcessor(IterableDataset):
    """
    Takes a Dataset over sample of the form:

    >>> {'prefix': str, 'suffix': str, 'middle': str, 'lang': str}

    and transform these samples into batched tensors of shape (micro_batch_size,
    sequence_length) using the tokenizer. Instead of corresponding to the the
    token to predict, the `targets` tensor is the distribution expected
    distribution using the many-token prediction objective.

    Each possible token is weighed by `f`.
    """

    def __init__(
        self,
        dataset: Dataset,
        sampler: DistributedSampler,
        tokenizer: TokenGeeXTokenizer,
        micro_batch_size: int,
        sequence_length: int,
        padded_vocab_size: int,
        f: Callable[[List[bytes]], List[float]],
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.iter = iter(self.sampler)
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.padded_vocab_size = padded_vocab_size
        self.f = f
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
        encoded_prefix = self.tokenizer.encode_ordinary(prefix)
        encoded_middle = self.tokenizer.encode_ordinary(middle)
        encoded_suffix = self.tokenizer.encode_ordinary(suffix)
        encoded_lang = self.tokenizer.encode_ordinary(lang)

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
            next = torch.tensor(
                self.buffer[1 : self.micro_batch_size * self.sequence_length + 1]
            )

            inputs = inputs.reshape(
                self.micro_batch_size, self.sequence_length
            ).contiguous()
            next = next.reshape(
                self.micro_batch_size, self.sequence_length
            ).contiguous()

            # Here, we diverge from next-token prediction where a single token
            # is considered valid. Instead, we consider that all tokens which
            # are a valid prefixes of the next token are valid.
            #
            # For example, given the following prefix and suffix:
            # > ['let samples = '] ['HashSet<String>']
            # We assume that all the following tokens are valid:
            # > H
            # > Hash
            # > HashSet
            # > HashSet<String>
            # Provided that they exist in the vocabulary.
            #
            # Hence, `targets` is modeled as (batch_size, sequence_length,
            # vocab_size) and has dtype=float32.
            #
            # Each possible token is scored between 0 and 1 according to `f`,
            # usually based on the length of the token.

            targets = torch.zeros(
                self.micro_batch_size,
                self.sequence_length,
                self.padded_vocab_size,
                dtype=torch.float32,
            )

            for i in range(self.micro_batch_size):
                for j in range(self.sequence_length):
                    id = next[i, j].long().item()
                    assert isinstance(id, int)

                    # There's a bug in tokengeex where id_to_token(vocab_size)
                    # crashes instead of returning None.
                    if self.tokenizer.id_to_special_token(id) is not None:
                        targets[i, j, id] = 1.0
                        continue

                    token = self.tokenizer.id_to_token(id)

                    if token is None:
                        # Special tokens have a single valid continuation.
                        targets[i, j, id] = 1.0
                        continue

                    (value, score) = token

                    try:
                        value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        # This can happen if the token is a single byte or a
                        # sequence of bytes which don't end on a character
                        # boundary. We just ignore this case.
                        targets[i, j, id] = 1.0
                        continue

                    possible_tokens_ids: List[int] = list(
                        self.tokenizer.common_prefix_search(value)
                    )
                    possible_token_values = [
                        self.tokenizer.id_to_token(v)[0]  # type: ignore
                        for v in possible_tokens_ids
                    ]
                    scores = self.f(possible_token_values)

                    assert len(possible_tokens_ids) == len(scores)

                    for k, score in zip(possible_tokens_ids, scores):
                        targets[i, j, k] = score

            assert inputs.shape == (self.micro_batch_size, self.sequence_length)

            self.buffer = self.buffer[self.micro_batch_size * self.sequence_length :]

            yield (inputs, targets), {}
