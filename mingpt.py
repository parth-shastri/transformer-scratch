## Inspired from Karpathys minGPT implementation

# This implementation somewhat mimics the huggingface implementation
# so that we can directly use the weights from the pretrained models from HF

import torch
import torch.nn as nn
from torch.nn import functional as F

import math


class GELU(nn.Module):
    """
    Implementation of the GELU activation from the paper
    https://arxiv.org/abs/1606.08415
    GELU is defined as the x * cum_pdf_gauss(x)
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalMultiHeadAttention(nn.Module):
    """
    An quick reimplementation of the SDPA with a causal mask
    This is in alignment with the GPT-2 implementation
    """

    def __init__(self, config):
        super().__init__()
        # query, key and value projections but all in one
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # projection of the post attention vector
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.residual_dropout = nn.Dropout(config.res_dropout)
        # causal attention mask
        self.register_buffer(
            "masked_bias",
            torch.tril(
                torch.ones((config.seq_len, config.seq_len)).view(
                    1, 1, config.seq_len, config.seq_len
                )
            ),
        )
        self.n_head = config.num_heads
        self.n_embed = config.d_model

    def forward(self, x):
        B, T, C = x.size()  # (B, seq_len, d_model)

        # calculate the query, key and values
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # fill the causal mask
        att = att.masked_fill(
            self.masked_bias[:, :, :T, :T] == 0, float("-inf")
        )  # fill with -ve inf vals
        # softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection w_o
        y = self.residual_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.d_model, config.d_ff),
                c_proj=nn.Linear(config.d_ff, config.d_model),
                act=GELU(),
                dropout=nn.Dropout(config.ff_dropout),
            )
        )

        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # prenorm
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.seq_len is not None
        assert config.d_model is not None
        self.seq_len = config.seq_len

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.seq_len, config.d_model),
                drop=nn.Dropout(config.embed_dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
                ln_f=nn.LayerNorm(config.d_model),
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # init all the weights as normal distribution N(0 , 0.02)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

        # report number of parameters
        n_parameters = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.seq_len
        ), f"The max seq_len is exceeded, Found len {t} expected {self.seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # forward the model
        token_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(token_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if the targets are given compute the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    @classmethod
    def load_pretrained(cls, model_name):
        """
        Kind of like the huggingface, we load a model from hugging face
        match the weights and then copy them over to our little GPT
        """

        assert model_name in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        from config import MyGPTConfig

        # create a desired model from our code
        kwargs = {
            "gpt2": dict(num_layers=12, num_heads=12, d_model=768),
            "gpt2-medium": dict(num_layers=24, num_heads=16, d_model=1024),
            "gpt2-large": dict(num_layers=36, num_heads=20, d_model=1280),
            "gpt2-xl": dict(num_layers=48, num_heads=25, d_model=1600),
        }[model_name]

        config = MyGPTConfig(**kwargs)
        model = GPT(config)

        sd = model.state_dict()

        # init the huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        keys = [k for k in sd if not k.endswith("attn.masked_bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(keys) == len(sd_hf)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # transpose the conv weights in the original GPT model
                assert (
                    sd_hf[k].shape[::-1] == sd[k].shape
                ), f"Transpose shape mismatch for {k}. Found {sd_hf[k].shape} and {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy for all the others
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"Shape mismatch for {k}. Found {sd_hf[k].shape} and {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, config):
        """
        Configure optimizers for the GPT model,
         1. The LayerNorm should not be added to the weight decay
         2. Embedding layers wont experience weight decay
         3. Biases should be excluded as well
        """

        decay = set()
        no_decay = set()
        include_modules = (nn.Linear,)
        exclude_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():

                full_name = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    # all biases are excluded
                    no_decay.add(full_name)
                elif pn.endswith("weight") and isinstance(m, include_modules):
                    decay.add(full_name)
                elif pn.endswith("weight") and isinstance(m, exclude_modules):
                    no_decay.add(full_name)

        # check if the whole spectrum of params is covered
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_params = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_params, lr=config.lr, betas=config.betas)

        return optimizer

    @torch.no_grad
    def generate(
        self, idx, max_new_tokens, temperature=0.1, do_sample=True, top_k=None
    ):
        """
        This is the function to generate max_new_tokens len of tokens given idx (LongTensor of shape (b, seq_len))
        either greedy or sampling from a distribution, with top_k, use in model.eval()
        """
        for _ in range(max_new_tokens):
            # if the input sequence is too long crop it from the left
            idx_cond = idx if idx.size(1) <= self.seq_len else idx[:, : -self.seq_len]
            # forward pass through the model
            logits, _ = self(idx_cond)
            # take the last logit
            logits = logits[:, -1, :] / temperature
            # crop the logits to the top_k options
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            # do softmax
            probs = F.softmax(logits, dim=-1)
            # either greedy or sample from the multinomial dist
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                _, idx_next = torch.max(probs, dim=-1)

            idx = torch.cat((idx, idx_next.unsqueeze(-1)), dim=1)

        return idx
