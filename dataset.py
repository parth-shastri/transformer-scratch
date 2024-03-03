import torch
from typing import Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
from data_utils import process_chunks, get_and_load_data
from data_utils import prepare_tokenizer
import datasets
import tokenizers
from config import MyGPTConfig

import logging

LOGFORMAT = (
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)

logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiText(Dataset):
    def __init__(
        self, ds: datasets.Dataset, tokenizer: tokenizers.Tokenizer, seq_len: int
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Filter the dataset
        self.ds = self.ds.filter(lambda x: len(x["text"]) > 150)
        # Chunk the data set into sequences of max_len equal to seq_len
        self.ds = self.ds.map(
            lambda x: process_chunks(x, 350),
            batched=True,
            remove_columns=self.ds.column_names,
            desc="Splitting into seq_len",
        )

        self.sos_token = torch.tensor(
            [self.tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [self.tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> Any:
        # Get the next item
        data = self.ds[index]
        # tokenize the text
        data_tokenized = self.tokenizer.encode(
            data["text"], add_special_tokens=False
        ).ids
        # add the special tokens
        pad_len = (
            self.seq_len - len(data_tokenized) - 1
        )  # subract 1 for the special EOS or BOS token


        if pad_len < 0:
            data_tokenized = data_tokenized[:(self.seq_len - 1)]   # truncating the dataset

        input_seq = torch.cat(
            [
                self.sos_token,
                torch.tensor(data_tokenized, dtype=torch.int64),
                torch.tensor([self.pad_token] * pad_len, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(data_tokenized, dtype=torch.int64),
                self.eos_token,
                torch.tensor([-100] * pad_len, dtype=torch.int64),   # replace pad token to -100 in the label
            ],
            dim=0,
        )
        return {
            "input_ids": input_seq,  # (seq_len)
            "labels": label,  # (seq_len)
            "attention_mask": (input_seq != self.pad_token)[
                None, :
            ].int(),  # (1, seq_len),   
            "text": data["text"],
        }


if __name__ == "__main__":
    train_ds, val_ds = get_and_load_data("wikitext", "wikitext-103-raw-v1")

    # build the tokenizer
    config = MyGPTConfig()
    tokenizer = prepare_tokenizer("./data/tokenizer", train_ds, config)

    # # Filter the dataset
    # train_ds = train_ds.filter(lambda x: len(x["text"]) > 100)
    # # Chunk the data set into sequences of max_len equal to seq_len
    # train_ds = train_ds.map(
    #     lambda x: process_chunks(x, 350),
    #     batched=True,
    #     remove_columns=train_ds.column_names,
    # )
    # # tokenize the dataset using the tokenizer that we trained
    # train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    # # set the format to torch
    # train_ds.set_format("torch")

    # check our dataset
    train_ds = WikiText(train_ds, tokenizer, seq_len=350)
    val_ds = WikiText(val_ds, tokenizer=tokenizer, seq_len=350)

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size)

    # look at 10 examples 
    print(train_ds[0])

    # look at one batch
    for data in train_dataloader:
        print(data)
        break

    from data_utils import data_iterator
    val_iterator = data_iterator(val_ds)

    # for item in val_iterator:
    #     print(item)   

    # print(len(val_ds))
