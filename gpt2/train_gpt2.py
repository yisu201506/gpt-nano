from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # attention is a "reduce" operation, which different tokens communicate with each other
        x = x + self.mlp(self.ln_2(x)) # mlp is a "map" operation, which the same token reflect on itself
        return x
    

class CausalAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # key, query, value prediction for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to prevent attending to future tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # self.c_attn outputs a tensor of shape (B,T,3*n_embd)
        # split it into 3 equal parts along dim=2 (embedding dimension)
        # each part has size n_embd, giving us separate query, key and value tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        wei = q @ k.transpose(-2, -1) * (1.0 / (k.size(-1) ** 0.5)) # (B, nh, T, T)
        wei = wei.masked_fill(self.bias[:, :, :T, :T]==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(out)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024 # sequence length
    vocab_size: int = 50257 # number of tokens: 50000 for tokens + 256 bytes tokens + 1 for <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size() # batch size, sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emd = self.transformer.wpe(pos) # shape (T, n_embd)
        tok_emd = self.transformer.wte(idx) # shape (B, T, n_embd)
        x = tok_emd + pos_emd # broadcasted addition
        # forward transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # shape (B, T, n_embd)
        logits = self.lm_head(x) # shape (B, T, vocab_size)
        return logits
        

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

if __name__ == "__main__":

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model=GPT.from_pretrained("gpt2")
    # print("did not crash!!!")

    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to("cpu")


    prompt = "Hello, I'm a language model,"
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)

    with torch.no_grad():
        sample_tokens = tokens.repeat(num_return_sequences, 1)
        for _ in range(max_length):
            logits = model(sample_tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=100, dim=-1)
            sampled_indices = torch.multinomial(top_probs, num_samples=1)  
            next_token_indices = torch.gather(top_indices, dim=1, index=sampled_indices)
            sample_tokens = torch.cat([sample_tokens, next_token_indices], 1)

        for sample_token in sample_tokens:
            print(tokenizer.decode(sample_token))
            print("-"*40)
