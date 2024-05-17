import random
import time
import unittest

import torch
from tokengeex import Tokenizer as TokenGeeXTokenizer
from tqdm import tqdm

from nano import Nano800M32KTokenGeeXGeneralMTP


class TestNano32K800M(unittest.TestCase):
    def test_dataloader(self):
        nano = Nano800M32KTokenGeeXGeneralMTP()
        lang = nano.tokenizer.special_token_to_id("<|lang|>")

        for i, (args, kwargs) in tqdm(enumerate(nano.dataloader(torch.device("cpu")))):
            x, y = args

            assert x.shape == (
                nano.micro_batch_size,
                nano.sequence_length,
            ), f"{x.shape} != {(nano.micro_batch_size, nano.sequence_length)}"

            if not nano.mpt:
                assert y.shape == (
                    nano.micro_batch_size,
                    nano.sequence_length,
                ), f"{y.shape} != {(nano.micro_batch_size, nano.sequence_length)}"
            else:
                assert (
                    y.shape
                    == (
                        nano.micro_batch_size,
                        nano.sequence_length,
                        nano.padded_vocab_size,
                    )
                ), f"{y.shape} != {(nano.micro_batch_size, nano.sequence_length, nano.padded_vocab_size)}"
                assert y.dtype == torch.float32

            if not nano.mpt:
                langs = []
                for j in range(x.shape[0]):
                    lang_indices = torch.where(x[j] == lang)[0]
                    if len(lang_indices) == 0:
                        continue
                    lang_indices = lang_indices.tolist()
                    for lang_index in lang_indices:
                        if lang_index + 1 >= x.shape[1]:
                            continue
                        next_token = x[j, lang_index + 1].item()
                        langs.append(nano.tokenizer.id_to_token(next_token))

                langs = [v.decode("utf-8") for v in langs]
                langs = {k: langs.count(k) for k in set(langs)}
                print(f"{langs}")
                time.sleep(0.5)

            if i == 0:
                if not nano.mpt:
                    trimmed_x = x[0].tolist()[:1024]
                    trimmed_y = y[0].tolist()[:1024]

                    decoded_x = nano.tokenizer.decode(
                        trimmed_x, include_special_tokens=True
                    )
                    decoded_y = nano.tokenizer.decode(
                        trimmed_y, include_special_tokens=True
                    )

                    print("-" * 80)
                    print(trimmed_x)
                    print("-" * 80)
                    print(decoded_x)
                    print("-" * 80)
                    print(trimmed_y)
                    print("-" * 80)
                    print(decoded_y)
                    print("-" * 80)
                else:
                    print(f"{x.shape}")
                    print(f"{y.shape}")

                    tokenizer = nano.tokenizer.unwrap()
                    assert isinstance(tokenizer, TokenGeeXTokenizer)

                    random_offset = random.randint(16, nano.sequence_length - 16)

                    context = x[0, random_offset - 3 : random_offset + 3].tolist()
                    random_x = x[0, random_offset].item()
                    random_y = y[0, random_offset, :].tolist()

                    if tokenizer.id_to_special_token(random_x):
                        print("Unlucky!")
                        continue

                    non_zero_random_y_indices = [
                        i for i, v in enumerate(random_y) if v != 0
                    ]

                    context = tokenizer.decode(context, include_special_tokens=True)
                    print(f"Context: {repr(context)}")
                    print(
                        f"At position {random_offset} of {nano.sequence_length} ",
                        end="",
                    )
                    print(
                        f"the randomly selected token is ({repr(tokenizer.id_to_token(random_x)[0])}) ",  # type: ignore
                        end="",
                    )
                    print(
                        f"and here is what must be predicted after it: ({[repr(tokenizer.id_to_token(v)[0]) for v in non_zero_random_y_indices]})"  # type: ignore
                    )

            if i > 10000:
                break


if __name__ == "__main__":
    unittest.main()
