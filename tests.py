import time
import unittest

import torch
from tqdm import tqdm

from nano import Nano800M32KHFDeepSeekCoder


class TestNano32K800M(unittest.TestCase):
    def test_dataloader(self):
        emma = Nano800M32KHFDeepSeekCoder()
        lang = emma.tokenizer.special_token_to_id("<|lang|>")

        for i, (args, kwargs) in tqdm(enumerate(emma.dataloader(torch.device("cpu")))):
            x, y = args
            assert x.shape == (
                emma.micro_batch_size,
                emma.sequence_length,
            ), f"{x.shape} != {(emma.micro_batch_size, emma.sequence_length)}"
            assert y.shape == (
                emma.micro_batch_size,
                emma.sequence_length,
            ), f"{y.shape} != {(emma.micro_batch_size, emma.sequence_length)}"

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
                    langs.append(emma.tokenizer.id_to_token(next_token))

            langs = [v.decode("utf-8") for v in langs]
            langs = {k: langs.count(k) for k in set(langs)}
            print(f"{langs}")
            time.sleep(0.5)

            if i == 0:
                trimmed_x = x[0].tolist()[:1024]
                trimmed_y = y[0].tolist()[:1024]

                decoded_x = emma.tokenizer.decode(
                    trimmed_x, include_special_tokens=True
                )
                decoded_y = emma.tokenizer.decode(
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

            if i > 10000:
                break


if __name__ == "__main__":
    unittest.main()
