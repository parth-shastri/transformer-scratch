from mygpt import build_gpt, MyGPT
from mingpt import GPT
from dataset import WikiText
from tokenizers import Tokenizer
from data_utils import causal_mask
from data_utils import prepare_tokenizer, get_and_load_data
from data_utils import get_weights_path, get_latest_weights_filepath
from config import MyGPTConfig

# import the torch necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from typing import Union
import torchmetrics
import warnings
from tqdm import tqdm
import os
from pathlib import Path

import logging

LOGFORMAT = (
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)

logging.basicConfig(format=LOGFORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


def greedy_decoding(
    model: Union[MyGPT, GPT],
    context: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    config: MyGPTConfig,
    device,
):
    sos_idx = tokenizer.token_to_id(config.sos_token)
    eos_idx = tokenizer.token_to_id(config.eos_token)

    # Init the decoder input
    decoder_input = context if context is not None else torch.tensor([sos_idx], dtype=torch.int64).unsqueeze(0)

    # the decoding loop
    while True:
        if decoder_input.size(1) == max_len:
            break

        # calculate the output
        out, _ = model(decoder_input)

        # probs of the next token
        prob = F.softmax(out[:, -1, :], dim=-1)
        # greedy decode (select the argmax)
        _, next_word = torch.max(prob, dim=-1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(-1)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def validation_fn(
    model: Union[MyGPT, GPT],
    dataloader: DataLoader,
    tokenizer: Tokenizer,
    writer: SummaryWriter,
    max_len: int,
    print_fn,
    device,
    config,
    print_max=2,
):
    """
    Simple validation loop to validate the model predictions at the end of each epoch
    # TODO: add perplexity and validation loss as metrics
    """
    # Put the model on eval mode
    model.eval()

    count = 0

    context_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in dataloader:
            count += 1
            context_len = torch.randint(low=4, high=25, size=())
            input_ids = batch["input_ids"]
            context = input_ids[:, :context_len].to(device)  # (B, context_len)

            # check if the batch size is 1
            assert context.size(0) == 1, "validation Batch size must be 1"

            # perform decoding
            model_out = greedy_decoding(
                model, context, tokenizer, max_len, config, device
            )

            # decode the output
            context_text = tokenizer.decode(context.squeeze(0).detach().cpu().numpy())
            expected_text = tokenizer.decode(
                input_ids[:, context_len:].squeeze(0).detach().cpu().numpy()
            )
            model_completion = tokenizer.decode(
                model_out[context_len:].detach().cpu().numpy()
            )

            # appendd
            context_texts.append(context_text)
            expected.append(expected_text)
            predicted.append(model_completion)

            # print the predictions
            print_fn("-" * 60)
            print_fn("CONTEXT/PROMPT: {}".format(context_text))
            print_fn("EXPECTED: {}".format(expected_text))
            print_fn("MODEL COMPLETION: {}".format(model_completion))

            if count == print_max:
                print_fn("-" * 60)
                break


def get_ds(config: MyGPTConfig):
    """
    Get the datasets and load them as a Torch Dataloader
    """
    train_ds, val_ds = get_and_load_data(
        data_name=config.datasource.split("/")[0],
        sub_data_name=config.datasource.split("/")[1],
    )

    # Get the Tokenizer
    tokenizer = prepare_tokenizer(config.tokenizer_file, train_ds, config=config)

    # prepare the dataset
    train_ds = WikiText(train_ds, tokenizer, seq_len=config.seq_len)
    val_ds = WikiText(val_ds, tokenizer, seq_len=config.seq_len)

    # Prepare the dataloader objects
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer


def get_model(config: MyGPTConfig) -> MyGPT:
    return build_gpt(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        n_layers=config.num_layers,
        d_model=config.d_model,
        mha_dropout=config.dropout,
        ff_dropout=config.dropout,
        res_dropout=config.dropout,
        h=config.num_heads,
        d_ff=config.d_ff,
        prenorm=config.prenorm,
    )


def train(config: MyGPTConfig):
    # get the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: {}".format(device))
    if device == "cuda":
        logger.info(f"Device name: {torch.cuda.get_device_name(device.index)}")
        logger.info(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    else:
        logger.info("NOTE: If you have a GPU, consider using it for training.")
        logger.info(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        logger.info(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )

    device = torch.device(device)

    # make sure the weights folder exists
    Path(f"{config.datasource.split('/')[0]}-{config.model_folder}").mkdir(
        parents=True, exist_ok=True
    )
    # equivalent to os' mkdirs after checking os.path.exists('path')

    #####################
    # LOAD THE DATASETS #
    #####################
    train_dataloader, val_dataloader, tokenizer = get_ds(config)

    #####################
    # LOAD THE MODEL    #
    #####################
    logger.info("Loading the model...")
    # model = get_model(config).to(device)
    model = GPT(config).to(device)
    # log the number of parameters in the model
    logger.info(
        "Model loaded with total parameters : {} M".format(
            sum(p.numel() for p in model.parameters()) // 10**6
        )
    )
    logger.info(
        "Trainalble parameters : {} M".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**6
        )
    )

    ######################
    # LOAD THE OPTIMIZER #
    ######################
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
    optimizer = model.configure_optimizers(config)

    ####################
    # PRELOAD AND INIT #
    ####################
    # summary writer
    # writer = SummaryWriter(
    #     config.experiment_name,
    #     comment=f"LR_{config.lr}_BATCH_{config.batch_size}_ADAMW",
    # )
    # weight loading
    initial_epoch = 0
    global_step = 0
    preload = config.preload

    model_filename = (
        get_latest_weights_filepath(config)
        if preload == "latest"
        else get_weights_path(config, preload)
        if preload
        else None
    )

    if model_filename:
        logger.info(
            "Preloading the weights from epoch {}: from {}".format(
                preload, model_filename
            )
        )
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        logger.info("No Model to preload, starting from scratch")

    # loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id(config.pad_token), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iter = tqdm(
            train_dataloader, desc="Processing Epoch {:02d}".format(epoch)
        )

        ##############
        # TRAIN LOOP #
        ##############
        # for batch_idx, batch in enumerate(batch_iter):
        #     model_input = batch["input_ids"].to(device)  # (B, seq_len)
        #     attention_mask = batch[
        #         "attention_mask"
        #     ]  # (B, 1, seq_len) dont put this on GPU as this is not to be used

        #     # label
        #     label = batch["labels"].to(device)  # (B, seq_len)

        #     # cau_msk = causal_mask(attention_mask.size(-1))  # (1, seq_len, seq_len)
        #     # decoder_mask = (attention_mask & cau_msk)[:, None, ...].to(
        #     #     device
        #     # )  # mask out the padding token and the next token (B, 1, seq_len, seq_len)

        #     # output
        #     model_out, _ = model(model_input)

        #     # Calculate theloss
        #     loss = loss_fn(model_out.view(-1, config.vocab_size), label.view(-1), ignore_index=-100)
        #     # modify the loss for the accumulation steps
        #     loss = loss / config.accumulation_steps
        #     batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

        #     # Log the loss
        #     writer.add_scalar("train_loss", loss.item(), global_step)
        #     writer.flush()

        #     # Backpropagate the gradients
        #     loss.backward()

        #     # Update the weights according to the gradient accumulation steps
        #     if not batch_idx % config.accumulation_steps:
        #         optimizer.step()
        #         optimizer.zero_grad(set_to_none=True)

        #     global_step += 1

        # Run validation
        writer = None
        validation_fn(
            model,
            val_dataloader,
            tokenizer,
            writer,
            max_len=config.seq_len,
            print_fn=lambda msg: batch_iter.write(msg),
            device=device,
            config=config,
        )

        # Save the model at the end of each epoch
        model_filename = get_weights_path(config, epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = MyGPTConfig()
    train(config)
