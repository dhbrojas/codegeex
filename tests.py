import os
import time
import unittest

from tokengeex import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from codegeex.datasets import BinaryFileDataset


def file_size_bytes(file_path):
    return os.path.getsize(file_path)


class TestBinaryFileDataset(unittest.TestCase):
    def test_read(self):
        filepath = "/workspace/datasets/tokenized/tokengeex/exact-32k-merged/train.bin"
        batch_size = 256
        sequence_length = 4096

        dataset = BinaryFileDataset(filepath, sequence_length)

        print(f"Size: {len(dataset)}")
        print(f"File size: {file_size_bytes(filepath)}")
        print(f"Mmap size: {len(dataset.mmap)}")

        assert len(dataset.mmap) == file_size_bytes(filepath) // 4
        assert len(dataset) == file_size_bytes(filepath) // 4 // sequence_length

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=1,
            prefetch_factor=2,
            pin_memory=False,
        )

        tokenizer = Tokenizer.from_file("resources/tokengeex/exact-32k-merged.json")

        elapsed = []

        for i, (x, y) in tqdm(
            enumerate(dataloader),
            total=len(dataset) // batch_size,
            desc=f"Size: {len(dataset)}",
        ):
            start = time.perf_counter_ns()
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

            self.assertEqual(x.shape, (batch_size, sequence_length))
            self.assertEqual(y.shape, (batch_size, sequence_length))

        print(f"Average time: {sum(elapsed) / len(elapsed):.2f}ns")


if __name__ == "__main__":
    unittest.main()
