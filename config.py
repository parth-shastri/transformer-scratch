from dataclasses import dataclass
from os import PathLike
from typing import Union


@dataclass
class MyGPTConfig:
    # Training and optimization args
    batch_size: int = 4   # use gradient accumulation for weight updates Larger batch size
    num_epochs: int = 20
    lr: float = 10**-4
    weight_decay = 0.01
    betas = (0.9, 0.99)
    accumulation_steps = 8

    # model args and hyperparameters
    seq_len: int = 1024
    d_model: int = 192
    vocab_size: int = 50_257
    num_layers: int = 6
    num_heads: int = 6
    embed_dropout: float = 0.1  # dropout for the initial embed layer
    attn_dropout: float = 0.1  # dropout after the calculation of attn scores
    ff_dropout: float = 0.1  # dropout in the feed-forward projections
    res_dropout: float = 0.1  # dropout after the feed-forward projections
    prenorm: bool = True
    d_ff: int = 768   # A general norm to use 4x proj dim
    eos_token: str = "[EOS]"
    sos_token: str = "[SOS]"
    pad_token: str = "[PAD]"

    # saving and loading args
    datasource = "wikitext/wikitext-103-raw-v1"
    model_folder: Union[str, PathLike] = "weights-gptmini"
    model_basename: str = "gptmini_wiki_"
    preload: Union[str, int] = "latest"
    tokenizer_file = "data/tokenizer"
    experiment_name: str = f"runs/model_gptmini_{datasource.split('/')[-1]}"
