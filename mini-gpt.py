from typing import List, Dict, Tuple, Optional
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

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = x.shape
        assert y is None or y.shape == (B, T)
        assert x.dtype == torch.int
        logits: torch.Tensor = self.embedding(x)
        assert logits.shape == (B, T, self.vocab_size)
        B, T, C = logits.shape

        if y is not None:
            # negative log likelihood loss
            # cross entropy operates on BxCxT (we have BxTxC) so we need to reshape
            logits: torch.Tensor = logits.view(B * T, C)
            targets: torch.Tensor = y.view(B * T)
            loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, x: torch.Tensor, maximum_out_len: int) -> torch.Tensor:
        B, T = x.shape
        C: int = self.vocab_size
        # generate more sequences (up to maximum_out_len) by concatenating into a running stream
        for _ in range(maximum_out_len):
            logits, _ = self.forward(x)  # don't care about loss, not training
            logits = logits[:, -1, :]  # only take last of block_size
            assert logits.shape == (B, C)  # new shape is (B, C)
            dist = torch.nn.functional.softmax(logits, dim=1)  # to probabilities
            assert dist.shape == logits.shape
            next_x = torch.multinomial(dist, num_samples=1)  # generate a next char
            assert next_x.shape == (B, 1)  # only a single sample
            x = torch.cat((x, next_x), dim=1).type(x.dtype)  # concat to running stream
            assert x.shape == (B, T + 1)
            T += 1  # for the future predictions
        return x


m = BigramLanguageModel(C=vocabulary_size)
m = m.to(device)
logits, loss = m.forward(x, y)
expected_loss: float = -np.log(1.0 / vocabulary_size)
print(f"Using BigramLanguageModel on initial (untrained sample batch)")
print(f"Initial prediction has loss: {loss:.2f}")
print(f"Expected loss with uniform weights: {expected_loss:.2f}")
initial_i: int = 1  # space
initial_s: str = decoder([initial_i])
num: int = 10
x0: torch.Tensor = (
    torch.ones(size=(1, 1), dtype=torch.int, device=device) * initial_i
)  # 1x1 block
pred_s: str = decoder(m.generate(x0, maximum_out_len=num)[0].tolist())
print(f'Initial ({num}) prediction starting with "{initial_s}" is "{pred_s}"')
