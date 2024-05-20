"""
A collection of classes for tokenization and tokenizers.
"""

import base64
from abc import ABC, abstractmethod
from typing import List, Union

from tiktoken import Encoding as TiktokenTokenizer
from tiktoken import encoding_for_model as tiktoken_encoding_for_model
from tokengeex import Tokenizer as TokenGeeXTokenizer
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str, encode_special_tokens: bool = False) -> List[int]:
        pass

    @abstractmethod
    def encode_batch(
        self, texts: list, encode_special_tokens: bool = False
    ) -> List[List[int]]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int], decode_special_tokens: bool = False) -> str:
        pass

    @abstractmethod
    def decode_batch(
        self, tokens: List[List[int]], decode_special_tokens: bool = False
    ) -> List[str]:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def special_tokens(self) -> List[str]:
        pass

    @abstractmethod
    def add_special_tokens(self, tokens: List[str]):
        pass

    @abstractmethod
    def token_to_id(self, token: bytes) -> int | None:
        pass

    @abstractmethod
    def id_to_token(self, id: int) -> bytes | None:
        pass

    @abstractmethod
    def special_token_to_id(self, token: str) -> int | None:
        pass

    @abstractmethod
    def id_to_special_token(self, id: int) -> str | None:
        pass

    @abstractmethod
    def unwrap(self) -> Union[HFTokenizer, TokenGeeXTokenizer, TiktokenTokenizer]:
        pass


class WrappedTokenGeeXTokenizer(Tokenizer):
    def __init__(self, tokenizer: TokenGeeXTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, encode_special_tokens: bool = False) -> List[int]:
        if encode_special_tokens:
            return self.tokenizer.encode(text)
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(
        self, texts: list, encode_special_tokens: bool = False
    ) -> List[List[int]]:
        if encode_special_tokens:
            return self.tokenizer.encode_batch(texts)
        return self.tokenizer.encode_ordinary_batch(texts)

    def decode(self, tokens: List[int], decode_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(tokens, decode_special_tokens)

    def decode_batch(
        self, tokens: List[List[int]], decode_special_tokens: bool = False
    ) -> List[str]:
        return self.tokenizer.decode_batch(tokens, decode_special_tokens)

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    def special_tokens(self) -> List[str]:
        return self.tokenizer.special_tokens()

    def add_special_tokens(self, tokens: List[str]):
        self.tokenizer.add_special_tokens(tokens)

    def token_to_id(self, token: bytes) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> bytes | None:
        token = self.tokenizer.id_to_token(id)
        if token is not None:
            (value, score) = token
            return value
        return None

    def special_token_to_id(self, token: str) -> int | None:
        return self.tokenizer.special_token_to_id(token)

    def id_to_special_token(self, id: int) -> str | None:
        return self.tokenizer.id_to_special_token(id)

    def unwrap(self) -> TokenGeeXTokenizer:
        return self.tokenizer


class WrappedTiktokenTokenizer(Tokenizer):
    def __init__(self, tokenizer: TiktokenTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, encode_special_tokens: bool = False) -> List[int]:
        if encode_special_tokens:
            return self.tokenizer.encode(text, allowed_special="all")
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(
        self, texts: list, encode_special_tokens: bool = False
    ) -> List[List[int]]:
        if encode_special_tokens:
            return self.tokenizer.encode_batch(texts)
        return self.tokenizer.encode_ordinary_batch(texts)

    def decode(self, tokens: List[int], decode_special_tokens: bool = False) -> str:
        if not decode_special_tokens:
            raise NotImplementedError(
                "Tiktoken does not support decoding without special tokens."
            )
        return self.tokenizer.decode(tokens)

    def decode_batch(
        self, tokens: List[List[int]], decode_special_tokens: bool = False
    ) -> List[str]:
        if decode_special_tokens:
            raise NotImplementedError(
                "Tiktoken does not support decoding with special tokens."
            )
        return self.tokenizer.decode_batch(tokens)

    def vocab_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    def special_tokens(self) -> List[str]:
        return list(self.tokenizer.special_tokens_set)

    def add_special_tokens(self, tokens: List[str]):
        if all([token in self.tokenizer.special_tokens_set for token in tokens]):
            return
        self.tokenizer = TiktokenTokenizer(
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            pat_str=self.tokenizer._pat_str,
            name=self.tokenizer.name,
            special_tokens={
                **self.tokenizer._special_tokens,
                **{
                    token: i + self.tokenizer.max_token_value + 1
                    for i, token in enumerate(tokens)
                },
            },
        )

    def token_to_id(self, token: bytes) -> int | None:
        return self.tokenizer.encode_single_token(token)

    def id_to_token(self, id: int) -> bytes | None:
        return self.tokenizer.decode_single_token_bytes(id)

    def special_token_to_id(self, token: str) -> int | None:
        return self.tokenizer.encode_single_token(token)

    def id_to_special_token(self, id: int) -> str | None:
        return self.tokenizer.decode_single_token_bytes(id).decode("utf-8")

    def unwrap(self) -> TiktokenTokenizer:
        return self.tokenizer


class WrappedHFTokenizer(Tokenizer):
    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, encode_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=encode_special_tokens).ids

    def encode_batch(
        self, texts: List, encode_special_tokens: bool = False
    ) -> List[List[int]]:
        raise NotImplementedError(
            "Haven't yet implemented this method for HFTokenizer."
        )

    def decode(self, tokens: List[int], decode_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(
            tokens, skip_special_tokens=not decode_special_tokens
        )

    def decode_batch(
        self, tokens: List[List[int]], decode_special_tokens: bool = False
    ):
        raise NotImplementedError(
            "Haven't yet implemented this method for HFTokenizer."
        )

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def special_tokens(self) -> List[str]:
        raise NotImplementedError(
            "Hugging Face tokenizers do not support special tokens."
        )

    def add_special_tokens(self, tokens: List[str]):
        self.tokenizer.add_special_tokens(tokens)

    def token_to_id(self, token: bytes) -> int | None:
        return self.tokenizer.token_to_id(token.decode("utf-8"))

    def id_to_token(self, id: int) -> bytes | None:
        return self.tokenizer.id_to_token(id).encode("utf-8")

    def special_token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_special_token(self, id: int) -> str | None:
        return self.tokenizer.id_to_token(id)

    def unwrap(self) -> HFTokenizer:
        return self.tokenizer


def load_custom_tiktoken_tokenizer(file: str, model: str, special_tokens: List[str]):
    with open(file, "rb") as f:
        contents = f.read()
        mergeable_ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

        source = tiktoken_encoding_for_model(model)

        return WrappedTiktokenTokenizer(
            TiktokenTokenizer(
                name="custom",
                mergeable_ranks=mergeable_ranks,
                pat_str=source._pat_str,
                explicit_n_vocab=None,
                special_tokens={
                    v: len(mergeable_ranks) + i for i, v in enumerate(special_tokens)
                },
            )
        )
