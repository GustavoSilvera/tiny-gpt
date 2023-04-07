from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
import os
from bow import compute_xBOW_softmax
from utils import *


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
batch_size: int = 32  # number of training instances happening at once (in parallel)
seed: int = 1  # to fix the randomness
torch.manual_seed(seed)
epochs: int = 1000
eval_iter: int = 200
n_embed: int = 32
lr: float = 1e-3


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
num_print: int = min(batch_size, 8)
first_str = f"first {num_print} out of " if num_print != batch_size else ""
print(f"Example input batch ({first_str}{x.shape}): \n{x[:num_print]}")
print(f"Example output targets ({first_str}{y.shape}): \n{y[:num_print]}")
print()


# create a head of self attention following the Attention is All You Need paper
# https://arxiv.org/pdf/1706.03762.pdf
# "Scaled Dot-Product Attention": Attention(Q, K, V) = softmax((Q @ K.T)/sqrt(C)) @ V


class AttHead(torch.nn.Module):
    # one "Head" of self attention
    def __init__(self, head_size: int):
        super().__init__()
        # keys represent what the current token knows about itself (ex. is a vowel)
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)
        # queries represent what information the token wants from the aggregations
        # ex. wants constanants or wants 2-positions ahead, etc.
        self.queries = torch.nn.Linear(n_embed, head_size, bias=False)
        self.values = torch.nn.Linear(n_embed, head_size, bias=False)
        # self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.tril = torch.tril(torch.ones((block_size, block_size), device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        K = self.key.forward(x)
        assert K.shape == (B, T, C)
        Q = self.queries.forward(x)  # self attention bc x is used for both K and Q
        assert Q.shape == (B, T, C)
        # compute attention scores ("scaled affinities")
        # the Q and K peform batched matrix multiply: (B,T,C) @ (B,C,T)->(B,T,T)
        # divide by sqrt for scaled attention (normalizes variance)
        W = Q @ K.transpose(dim0=-2, dim1=-1) * (C**-0.5)
        assert W.shape == (B, T, T)
        # effectively doing the same thing as xBOW-softmax
        W[:, self.tril[:T, :T] == 0] = -float("inf")  # prohibiting future comms
        W = torch.nn.functional.softmax(W, dim=-1)
        assert W.shape == (B, T, T)
        V = self.values.forward(x)
        assert V.shape == (B, T, C)
        out = W @ V  # (B,T,T) @ (B,T,C) -> (B,T,C)
        assert out.shape == (B, T, C)
        return out


# begin a BigramLanguageModel impl


class BigramLanguageModel(torch.nn.Module):
    def __init__(self, C: int, T: int, n_embed: int):
        super().__init__()
        # create an embedding table to map the tokens to the "next" tokens
        self.vocab_size = C
        self.n_embed = n_embed
        self.token_embedding = torch.nn.Embedding(C, n_embed)
        self.posn_embedding = torch.nn.Embedding(T, n_embed)
        self.sa_head = AttHead(n_embed)  # self-attention-head
        self.lm_head = torch.nn.Linear(n_embed, C)  # language-model-head

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = x.shape
        C = self.n_embed
        assert y is None or y.shape == (B, T)
        assert x.dtype == torch.int
        # add a layer of indirection to get from x -> embedding -> linear layer -> logits
        tok_emb: torch.Tensor = self.token_embedding(x)
        assert tok_emb.shape == (B, T, C)
        pos_emb: torch.Tensor = self.posn_embedding(torch.arange(T, device=device))
        assert pos_emb.shape == (T, C)
        x = tok_emb + pos_emb  # broadcast the addition of pos_emb throughout batches
        x = self.sa_head.forward(x)  # install self-attention
        assert x.shape == (B, T, C)
        logits: torch.Tensor = self.lm_head(x)
        assert logits.shape == (B, T, self.vocab_size)
        B, T, C = logits.shape

        if y is not None:
            # negative log likelihood loss
            # cross entropy operates on BxCxT (we have BxTxC) so we need to reshape
            logits: torch.Tensor = logits.view(B * T, C)
            targets: torch.Tensor = y.view(B * T).type(torch.long)
            loss: torch.Tensor = torch.nn.functional.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, x: torch.Tensor, maximum_out_len: int) -> torch.Tensor:
        B, T = x.shape
        C: int = self.vocab_size
        # generate more sequences (up to maximum_out_len) by concatenating into a running stream
        for _ in range(maximum_out_len):
            x_safe = x[:, -block_size:]  # crop to last {block_size} tokens (to fit)
            logits, _ = self.forward(x_safe)  # don't care about loss, not training
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


m = BigramLanguageModel(C=vocabulary_size, T=block_size, n_embed=n_embed)
m = m.to(device)
logits, loss = m.forward(x, y)
expected_loss: float = -np.log(1.0 / vocabulary_size)
print(f"Using BigramLanguageModel on initial (untrained sample batch)")
print(f"Initial prediction has loss: {loss:.2f}")
print(f"Expected loss with uniform weights: {expected_loss:.2f}")
initial_i: int = 1  # space
initial_s: str = decoder([initial_i])
x0: torch.Tensor = torch.ones((1, 1), dtype=torch.int, device=device) * initial_i  # 1x1
pred_s: str = decoder(m.generate(x0, maximum_out_len=10)[0].tolist())
print(f'Initial random (10) prediction starting with "{initial_s}" is "{pred_s}"')
print()

# create a loss estimator for averaging training and val loss
def estimate_loss(num_iters: int = 200) -> Tuple[float, float]:
    losses: Dict[str, float] = {}
    with torch.no_grad():  # no need to track gradients (lower memory footprint)
        m.eval()  # switch to evaluation mode
        for split in ["train", "val"]:
            cumulative_loss: float = 0
            for _ in range(num_iters):
                xb, yb = sample_batch(split)
                _, loss = m.forward(xb, yb)
                cumulative_loss += loss.item()
            losses[split] = cumulative_loss / num_iters
        m.train()  # back to training phase
    return losses["train"], losses["val"]


# train the model so its not just purely random
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

train_loss: Optional[float] = float("nan")
val_loss: Optional[float] = float("nan")
for epoch in range(epochs):
    xb, yb = sample_batch()
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(
        f"Training {epoch}/{epochs} \t ({100 * epoch / epochs}%) \t Train loss: {train_loss:.2f} \t Val loss: {val_loss:.2f}",
        end="\r",
        flush=True,
    )
    if epoch % eval_iter == 0:
        train_loss, val_loss = estimate_loss(num_iters=eval_iter)
        print()
print()
print()
print(f"Total loss after {epochs} epochs: {loss.item():.2f}")
pred_s = decoder(m.generate(x0, maximum_out_len=100)[0].tolist())
print(f'Trained prediction starting with "{initial_s}" is "{pred_s}"')
print()
