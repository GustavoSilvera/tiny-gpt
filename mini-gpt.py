from typing import List, Dict
import numpy as np
import torch
import os

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M1 Metal GPU acceleration!")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVidia Cuda acceleration!")
print()

# Karpathy's tiny-shakespeare
dataset = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filename = "input.txt"  # name of final text dataset

if not os.path.exists(filename):
    print(f'Downloading text-dataset from "{dataset}"')
    print()
    os.system(f"wget {dataset}")

assert os.path.exists(filename)
print(f'Using text dataset from "{filename}"')

dataset = ""
with open(filename, "r") as f:
    dataset = f.read()

# print metadata about this dataset
characters = sorted(list(set(dataset)))  # get the ordered unique characters in the text
vocabulary_size: int = len(characters)
print(f"Available characters ({vocabulary_size}) in dataset: {characters}")
print()

# create encoder/decoder (character-level tokenizer) for mapping between int <-> char
stoi: Dict[str, int] = {ch: i for i, ch in enumerate(characters)}
itos: Dict[int, str] = {i: ch for ch, i in stoi.items()}
assert len(stoi) == len(itos)


def encoder(s: str) -> List[int]:
    # given a string, return its encoding as a per-character list of integers
    return [stoi[c] for c in s]


def decoder(ints: List[int]) -> str:
    # inverse of encoder, take a list of integers and create its corresponding str
    return "".join([itos[i] for i in ints])


print("Encoder test: ", encoder("Hello There!"))
print("Decoder test: ", decoder(encoder("Hello There!")))
print()

# encoding the entire dataset and using Torch:
dataset_t: torch.Tensor = torch.tensor(
    encoder(dataset), dtype=torch.int8, device=device
)
print(f"Loaded entire dataset into torch Tensor: {dataset_t.shape}, {dataset_t.dtype}")

# split dataset into training and validation
percentage_for_training: float = 0.9  # % used for training, remaining used for val
cutoff: int = int(len(dataset_t) * percentage_for_training)
train: torch.Tensor = dataset_t[:cutoff]
val: torch.Tensor = dataset_t[cutoff:]
print(f"Split dataset into training: {train.shape} and val: {val.shape}")
