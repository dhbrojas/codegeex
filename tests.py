import os
import time
import unittest

from tokengeex import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datasets import BinaryFileDataset


def file_size_bytes(file_path):
    return os.path.getsize(file_path)


class TestBinaryFileDataset(unittest.TestCase):
    def test_read(self):
        dataset = BinaryFileDataset(
            "/workspace/datasets/tokenized/tokengeex/exact-32k-merged/train.bin", 4096
        )

        assert (
            len(dataset)
            == file_size_bytes(
                "/workspace/datasets/tokenized/tokengeex/exact-32k-merged/train.bin"
            )
            // 4
            // 4096
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            drop_last=True,
        )

        tokenizer = Tokenizer.from_file("resources/tokengeex/exact-32k-merged.json")

        elapsed = []

        it = iter(dataloader)

        for i in tqdm(range(256)):
            start = time.perf_counter_ns()
            x, y = next(it)
            elapsed.append(time.perf_counter_ns() - start)

            if i == 0:
                trimmed_x = x[0].tolist()[:256]
                trimmed_y = y[0].tolist()[:256]

                decoded_x = tokenizer.decode(trimmed_x, include_special_tokens=True)
                decoded_y = tokenizer.decode(trimmed_y, include_special_tokens=True)

                print("-" * 80)
                print(trimmed_x)
                print("-" * 80)
                print(decoded_x)
                print("-" * 80)
                print(trimmed_y)
                print("-" * 80)
                print(decoded_y)
                print("-" * 80)

            self.assertEqual(x.shape, (32, 4096))
            self.assertEqual(y.shape, (32, 4096))

            # Sleep for the forward + backward pass time (~50ms)
            time.sleep(0.05)

        print(f"Average time: {sum(elapsed) / len(elapsed):.2f}ns")


if __name__ == "__main__":
    unittest.main()
