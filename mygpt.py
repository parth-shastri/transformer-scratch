import torch 
import torch.nn as nn
import torch.nn.functional as F
from model_utils import Decoder, ProjectionLayer

class MyGPT(nn.Module):
    def __init__(self, n_layers: int, seq_len: int, vocab_size: int,  d_model: int, h: int, d_ff:int, mha_dropout: float, ff_dropout: float, res_dropout: float, prenorm=True):
        super().__init__()
        # init the decoder and the proj blocks
        self.decoder = Decoder(n_layers, seq_len, vocab_size, d_model, h, d_ff, mha_dropout, ff_dropout, res_dropout, prenorm)
        self.proj = ProjectionLayer(d_model, vocab_size)

    def forward(self, x: torch.Tensor, targets=None) -> torch.Tensor:
        # (B, seq_len, 1) --> (B, seq_len, vocab_size)
        x = self.decoder(x)
        logits = self.proj(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


def build_gpt(vocab_size, seq_len, n_layers: int=6, d_model: int=512, mha_dropout: float=0.1, ff_dropout: int=0.1, res_dropout: float=0.1, h:int=8, d_ff: int=2048, prenorm=True) -> MyGPT:
    # init and return the model
    return MyGPT(n_layers, seq_len, vocab_size, d_model, h, d_ff, mha_dropout, ff_dropout, res_dropout)

def _test_model_init():
    myGPT = build_gpt(50_000, 350)

    # init random input batch
    seq_len = 350
    batch_size = 8
    vocab_size = 50_000
    x = torch.randint(low=0, high=vocab_size, size=(8, 350)) # (B, seq_len) int

    print(myGPT)

    print("*" * 50)
    print("Testing the forward pass")
    print("*" * 50)

    print("test seq_len: {}".format(seq_len))
    print("test batch_size: {}".format(batch_size))

    #mask simulation
    mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1)[:, None, ...].type(torch.int) == 0
    out, _ = myGPT(x)
    print("Output shape: {}".format(out.shape))


if __name__ == "__main__":
    _test_model_init()
