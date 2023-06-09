import torch
from typing import Optional, Tuple, Dict
from utils import device, ckpt_dir
import os
from data import train, val, encoder, decoder, vocabulary_size
import numpy as np

# create a head of self attention following the Attention is All You Need paper
# https://arxiv.org/pdf/1706.03762.pdf
# "Scaled Dot-Product Attention": Attention(Q, K, V) = softmax((Q @ K.T)/sqrt(C)) @ V

# begin sampling blocks from the dataset
block_size: int = 256  # size of the context that we train our transformer on
batch_size: int = 64  # number of training instances happening at once (in parallel)
seed: int = 1  # to fix the randomness
torch.manual_seed(seed)
epochs: int = 4000
eval_iter: int = 150
n_embed: int = 384  # should be a multiple of num_heads
lr: float = 0.001
n_layer: int = 6
num_heads: int = 8  # every head is {n_embed//num_heads}-dimensional
dropout: float = 0.2  # percent of indermediate calculations that are disabled


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
        # registered as a buffer to be a "const" parameter that is not learned/trained
        # in the training step for this model. Ie. it is not a parameter that torch cares about
        # see https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/3
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        # self.tril = torch.tril(torch.ones((block_size, block_size), device=device)) # equivalent
        assert hasattr(self, "tril") and self.tril is not None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        xC = C  # keep track of the original C
        # in case head_size < C from multi-head attention partitioning
        C = self.key.out_features if self.key.out_features < C else C
        K = self.key.forward(x)
        assert K.shape == (B, T, C)
        Q = self.queries.forward(x)  # self attention bc x is used for both K and Q
        assert Q.shape == (B, T, C)
        # compute attention scores ("scaled affinities")
        # the Q and K peform batched matrix multiply: (B,T,C) @ (B,C,T)->(B,T,T)
        # divide by sqrt for scaled attention (normalizes variance)
        W = Q @ K.transpose(dim0=-2, dim1=-1) * (xC**-0.5)
        assert W.shape == (B, T, T)
        # effectively doing the same thing as xBOW-softmax
        W[:, self.tril[:T, :T] == 0] = -float("inf")  # prohibiting future comms
        W = torch.nn.functional.softmax(W, dim=-1)
        assert W.shape == (B, T, T)
        # dropout is a regularization technique to randomly disable some nodes in the
        # network and act like training several smaller ensembles of networks at once
        # then at test time, averaging them to get an extra ~2% boost
        W = self.dropout.forward(W)  # (randomly prevent some nodes from communicating)
        V = self.values.forward(x)
        assert V.shape == (B, T, C)
        out = W @ V  # (B,T,T) @ (B,T,C) -> (B,T,C)
        assert out.shape == (B, T, C)
        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, N: int, head_size: int):
        super().__init__()
        # create multiple (N) equal attention heads to run in parallel
        self.heads = torch.nn.ModuleList([AttHead(head_size) for _ in range(N)])
        # projection is linear transformation of the concatenated attentions
        self.proj = torch.nn.Linear(n_embed, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # concatenate the result of running all the attention heads ("in parallel")
        out = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        return self.dropout.forward(self.proj.forward(out))


class FeedForward(torch.nn.Module):  # simple MLP
    def __init__(self, N: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N, 4 * N),  # making deeper residual block
            torch.nn.ReLU(),
            torch.nn.Linear(4 * N, N),  # projection layer back into residual pathway
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x)


# same as MultiHeadAttention but vectorized over heads
class CausalSelfAttention(torch.nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        # create attention layer (note 3x out dim for key, query, values)
        self.attn = torch.nn.Linear(n_embed, 3 * n_embed)
        # projection is linear transformation of the concatenated attentions
        self.proj = torch.nn.Linear(n_embed, n_embed)
        self.num_heads = num_heads
        # regularization
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        assert hasattr(self, "tril") and self.tril is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C % self.num_heads == 0  # so division can take place evenly
        Q, K, V = self.attn(x).split(n_embed, dim=2)  # extract 3 quantities at 2nd dim
        nh = self.num_heads
        # split the last dimension (C) into nh and C//nh
        Q = Q.view(B, T, nh, C // nh).transpose(1, 2)
        K = K.view(B, T, nh, C // nh).transpose(1, 2)
        V = V.view(B, T, nh, C // nh).transpose(1, 2)
        assert K.shape == Q.shape == V.shape == (B, nh, T, C // nh)
        # compute attention scores ("scaled affinities")
        # the Q and K peform batched matrix multiply: (B,T,C) @ (B,C,T)->(B,T,T)
        # divide by sqrt for scaled attention (normalizes variance)
        W = Q @ K.transpose(dim0=-2, dim1=-1) * (K.shape[-1] ** -0.5)
        assert W.shape == (B, nh, T, T)
        # effectively doing the same thing as xBOW-softmax
        W[:, :, self.tril[:T, :T] == 0] = -float("inf")  # prohibiting future comms
        W = torch.nn.functional.softmax(W, dim=-1)
        assert W.shape == (B, nh, T, T)
        # dropout is a regularization technique to randomly disable some nodes in the
        # network and act like training several smaller ensembles of networks at once
        # then at test time, averaging them to get an extra ~2% boost
        W = self.attn_dropout.forward(W)  # (prevent some nodes from communicating)
        out = W @ V  # (B,nh,T,T) @ (B,nh,T,C) -> (B,nh,T,C)
        assert out.shape == (B, nh, T, C // nh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        assert out.shape == (B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


# attention block combining the MultiHeadAttention and a feed-forward layer
class AttnBlock(torch.nn.Module):
    def __init__(self, n_embed: int, num_heads: int):
        super().__init__()
        assert n_embed % num_heads == 0  # to ensure equality
        # self.sa_heads = MultiHeadAttention(N=num_heads, head_size=n_embed // num_heads)
        self.sa_heads = CausalSelfAttention(num_heads=num_heads)
        self.feed_forward = FeedForward(n_embed)
        # layer norm is like batch normalization (make unit gaussian distribution of weights)
        # along the layers rather than the individual weights/biases
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using addition to encode residual connections (gradient-super-highway)
        x = x + self.sa_heads.forward(self.ln1.forward(x))
        x = x + self.feed_forward.forward(self.ln2.forward(x))
        return x


class BigramLanguageModel(torch.nn.Module):
    def __init__(
        self,
        C: int = vocabulary_size,
        T: int = block_size,
        n_embed: int = n_embed,
        n_layer: int = n_layer,
    ):
        super().__init__()
        # create an embedding table to map the tokens to the "next" tokens
        self.vocab_size = C
        self.n_embed = n_embed
        self.token_embedding = torch.nn.Embedding(C, n_embed)
        self.posn_embedding = torch.nn.Embedding(T, n_embed)
        self.attn_blocks = torch.nn.Sequential(
            *(
                [
                    AttnBlock(n_embed=n_embed, num_heads=num_heads)
                    for _ in range(n_layer)  # how many attention block layers
                ]
                + [torch.nn.LayerNorm(n_embed)]  # finalize with a layernorm
            )
        )
        self.lm_head = torch.nn.Linear(n_embed, C)  # language-model-head

        print(
            f"Created Bigram language model with {n_layer} layers with {n_embed} embeddings and {num_heads} heads"
        )

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
        x = self.attn_blocks.forward(x)  # install self-attention
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
        for i in range(maximum_out_len):
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
            print(
                f"({100 * i/maximum_out_len:.1f}%)",
                end="\r",
                flush=True,
            )
        return x

    def load(self, ckpt: str) -> None:
        with open(os.path.join(ckpt_dir, ckpt), "rb") as f:
            self.load_state_dict(torch.load(f))
            print(f'Loaded state dict from "{ckpt}" successfully!')
            print()

    def example(self) -> None:
        logits, loss = self.forward(x, y)
        expected_loss: float = -np.log(1.0 / vocabulary_size)
        print(f"Using BigramLanguageModel on initial (untrained sample batch)")
        print(f"Initial prediction has loss: {loss:.2f}")
        print(f"Expected loss with uniform weights: {expected_loss:.2f}")
        initial_i: int = 1  # space
        initial_s: str = decoder([initial_i])
        x0: torch.Tensor = (
            torch.ones((1, 1), dtype=torch.int, device=device) * initial_i
        )  # 1x1
        pred_s: str = decoder(self.generate(x0, maximum_out_len=10)[0].tolist())
        print(
            f'Initial random (10) prediction starting with "{initial_s}" is "{pred_s}"'
        )
        print()

    # create a loss estimator for averaging training and val loss
    def estimate_loss(self, num_iters: int = 200) -> Tuple[float, float]:
        losses: Dict[str, float] = {}
        with torch.no_grad():  # no need to track gradients (lower memory footprint)
            self.eval()  # switch to evaluation mode
            for split in ["train", "valid"]:
                cumulative_loss: float = 0
                for i in range(num_iters):
                    xb, yb = sample_batch(split)
                    _, loss = self.forward(xb, yb)
                    cumulative_loss += loss.item()
                    print(
                        f"({split}) Eval: {100 * i / num_iters:.0f}%",
                        end="\r",
                        flush=True,
                    )
                losses[split] = cumulative_loss / num_iters
            self.train()  # back to training phase
        return losses["train"], losses["valid"]

    def start_training(self) -> None:
        self.train()
        # train the model so its not just purely random
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        train_loss: Optional[float] = float("nan")
        val_loss: Optional[float] = float("nan")
        for epoch in range(epochs):
            xb, yb = sample_batch()
            logits, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            est_loss: bool = epoch % eval_iter == eval_iter - 1
            if est_loss:
                train_loss, val_loss = self.estimate_loss(num_iters=eval_iter)
                torch.save(
                    self.state_dict(), os.path.join("saves", f"state_dict_{epoch}.pt")
                )
                scheduler.step(loss)
            print(
                f"Training {epoch:>4}/{epochs} \t ({100 * epoch / epochs:.1f}%) \t Train loss: {train_loss:.2f} \t Val loss: {val_loss:.2f}",
                end="\r",
                flush=True,
            )
            if est_loss:
                print()
        print()
        print()
        print(f"Total loss after {epochs} epochs: {loss.item():.2f}")

    def evaluate(self, prompt: str, out_len: int = 100) -> str:
        prompt_enc = torch.tensor([encoder(prompt)], dtype=torch.int, device=device)
        encoded_out = self.generate(prompt_enc, maximum_out_len=out_len)
        assert encoded_out.shape[0] == 1
        out: str = decoder(encoded_out[0].tolist())
        return out


def sample_batch(type: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    # sample a small segment of data from the dataset at random
    data = train if type == "train" else val
    start_idx = torch.randint(low=0, high=len(data) - block_size, size=(batch_size, 1))
    # TODO: vectorize
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
