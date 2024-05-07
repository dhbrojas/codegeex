import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryFileDataset(Dataset):
    """
    Loads a token dataset from a NumPy binary file.
    """

    def __init__(self, file: str, sequence_length: int):
        self.file = file
        self.sequence_length = sequence_length
        self.mmap = np.memmap(self.file, mode="r", dtype=np.uint32)

    def __len__(self):
        return len(self.mmap) // self.sequence_length

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        seq = torch.from_numpy(
            self.mmap[
                index * self.sequence_length : (index + 1) * self.sequence_length + 1
            ].copy(),
        ).long()

        return seq[:-1], seq[1:]


class FakeDataset(Dataset):
    def __init__(self, vocab_size, sequence_length, num_elements):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_elements = num_elements

    def __len__(self):
        return self.num_elements

    def __getitem__(self, index):
        if index >= self.num_elements:
            raise IndexError
        return (
            torch.randint(0, self.vocab_size, (self.sequence_length,)),
            torch.randint(0, self.vocab_size, (self.sequence_length,)),
        )
