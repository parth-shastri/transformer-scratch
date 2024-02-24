import torch
import datasets
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import processors
from tokenizers import decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from pathlib import Path
from transformers import AutoTokenizer
import os
from typing import Optional, Union, Any
from config import MyGPTConfig


import logging

LOGFORMAT = (
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)

logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_and_load_data(
    data_name: str,
    sub_data_name: str,
) -> tuple[datasets.Dataset]:
    raw_ds = datasets.load_dataset(data_name, sub_data_name)
    train_ds = raw_ds["train"]
    val_ds = raw_ds["validation"]

    return train_ds, val_ds


def data_iterator(ds):
    for item in ds:
        yield item["text"]


def tokenize(inputs: list, tokenizer: Tokenizer) -> dict:
    """
    Tokenize the inputs usint the tokenizer
    """
    batch_encoding = tokenizer.encode_batch(inputs["text"], add_special_tokens=False)
    return {
        "input_ids": [encoding.ids for encoding in batch_encoding],
        "attention_mask": [encoding.attention_mask for encoding in batch_encoding],
    }


def prepare_tokenizer(tokenizer_path: Union[str, os.PathLike], ds, config: MyGPTConfig):
    if Path(tokenizer_path).exists():
        logger.info("Tokenizer found at {}".format(tokenizer_path))
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        # Train the tokenizer on the input dataset
        # https://huggingface.co/docs/tokenizers/quicktour
        logger.info("Tokenizer Not found")
        logger.info("Training a BPE tokenizer...")
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(   
            vocab_size=config.vocab_size,
            special_tokens=[
                config.sos_token,
                config.eos_token,
                config.pad_token
            ],
            show_progress=True,
        )
        tokenizer.train_from_iterator(data_iterator(ds), trainer=trainer)
        tokenizer.save(tokenizer_path)

    return tokenizer


def process_chunks(batch, max_len):
    """
    function to divide the big input examples into chunks of smaller length,
    NOTE: Always use batched processing to perform chunking
    """
    # code from https://huggingface.co/docs/datasets/v2.14.5/en/process#batch-processing
    chunks = []
    for text in batch["text"]:
        chunks += [text[i : i + max_len] for i in range(0, len(text), max_len)]

    return {"text": chunks}


def mapper(inputs):
    # for item in train_ds:
    # split the dataset into seq_len long chunks
    input_ids = inputs["input_ids"].split(350)  # use the seq len
    attention_masks = inputs["attention_mask"].split(350)
    return {
        "input_ids": [input_id for input_id in input_ids],
        "attention_mask": [attention_mask for attention_mask in attention_masks],
    }


# Define a function to flatten the "input_ids" column


def patches_to_examples(batch):
    return {
        "input_ids": [
            input_id for input_ids in batch["input_ids"] for input_id in input_ids
        ],
        "attention_mask": [
            attention_mask
            for attention_masks in batch["input_ids"]
            for attention_mask in attention_masks
        ],
    }


def causal_mask(size: int):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_weights_path(config: MyGPTConfig, epoch: int):
    model_folder = f'{config.datasource.split("/")[0]}-{config.model_folder}'
    model_filename = f"{config.model_basename}{epoch}"
    return str(Path(".") / model_folder / model_filename)


# find the latest weights for the model
def get_latest_weights_filepath(config: MyGPTConfig):
    model_folder = f'{config.datasource.split("/")[0]}-{config.model_folder}'
    model_filename = f"{config.model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort(key=lambda x: int(str(x).split("_")[-1]))
    return str(weights_files[-1])

