from typing import List, Dict, Tuple
import torch
import numpy as np
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
dataset_t: torch.Tensor = torch.tensor(encoder(dataset), dtype=torch.int, device=device)
print(f"Loaded entire dataset into torch Tensor: {dataset_t.shape}, {dataset_t.dtype}")

# split dataset into training and validation
percentage_for_training: float = 0.9  # % used for training, remaining used for val
cutoff: int = int(len(dataset_t) * percentage_for_training)
train: torch.Tensor = dataset_t[:cutoff]
val: torch.Tensor = dataset_t[cutoff:]
print(f"Split dataset into training: {train.shape} and val: {val.shape}")
print()

# begin sampling blocks from the dataset
block_size: int = 8  # size of the context that we train our transformer on
batch_size: int = 4  # number of training instances happening at once (in parallel)
seed: int = 1  # to fix the randomness
torch.manual_seed(seed)


def sample_batch(type: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    # sample a small segment of data from the dataset at random
    data = train if type == "train" else val
    start_idx = torch.randint(low=0, high=len(data) - block_size, size=(batch_size, 1))
    x: torch.Tensor = torch.stack([data[i : i + block_size] for i in start_idx])
    # offset x by 1 to get the "targets"
    y: torch.Tensor = torch.stack([data[i + 1 : i + block_size + 1] for i in start_idx])
    return x, y


# example random chunks
x, y = sample_batch()
print(f"Example input batch ({x.shape}): \n{x}")
print(f"Example output targets ({y.shape}): \n{y}")
print()

# begin a BigramLanguageModel impl


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, C: int):
        super().__init__()
        # create an embedding table to map the tokens to the "next" tokens
        self.vocab_size = C
        self.embedding = torch.nn.Embedding(C, C, device=device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)
        assert x.dtype == torch.int
        logits: torch.Tensor = self.embedding(x)
        assert logits.shape == (batch_size, block_size, self.vocab_size)
        B, T, C = logits.shape

        # negative log likelihood loss
        # cross entropy operates on BxCxT (we have BxTxC) so we need to reshape
        logits: torch.Tensor = logits.view(B * T, C)
        targets: torch.Tensor = y.view(B * T)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss


m = BigramLanguageModel(C=vocabulary_size)
logits, loss = m.forward(x, y)
expected_loss: float = -np.log(1.0 / vocabulary_size)
print(f"Using BigramLanguageModel on initial (untrained sample batch)")
print(f"Initial prediction has loss: {loss:.2f}")
print(f"Expected loss with uniform weights: {expected_loss:.2f}")
