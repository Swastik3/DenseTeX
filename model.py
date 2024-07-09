"""
entire gpt2-124M model architecture
"""

import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    

class CrossAttention(nn.Module):

    def __init__(self, config):

        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout


    def forward(self, x, encoder_output): 

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # _, T_enc, _ = encoder_output.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # q = self.c_attn(x).split(self.n_embd, dim=2)[0]
        # k, v = self.c_attn(encoder_output).split(self.n_embd, dim=2)[1:]
        
        # k = k.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_enc, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = v.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_enc, hs)

        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection 
        y = self.resid_dropout(self.c_proj(y))

        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, encoder_output):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), encoder_output)
        x = x + self.mlp(self.ln_2(x))

        return x



@dataclass
class GPTConfig:
    # block_size: int = 1024
    # vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster    
    block_size: int = 300  # Set to match the first dimension of our embeddings
    vocab_size: int = 50257   # Set this to the number of classes in your task basically means the number of tokens in your vocabulary


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("witiko/mathberta")
        
        # Update vocab_size based on the tokenizer
        config.vocab_size = len(self.tokenizer)
        
        # Update the lm_head and token_embedding_layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding_layer = nn.Embedding(config.vocab_size, config.n_embd)
        
        # weight tying
        self.token_embedding_layer.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))



    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """

        return sum(p.numel() for p in self.parameters())


    
    def _init_weights(self, module):
        """ intialise weights for training """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    # def crop_block_size(self, block_size):
    #     # model surgery to decrease the block size if necessary
    #     # but want to use a smaller block size for some smaller, simpler model

    #     assert block_size <= self.config.block_size
    #     self.config.block_size = block_size

    #     for block in self.transformer.h:
    #         if hasattr(block.attn, 'bias'):
    #             block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]



    def forward(self, input_embd, targets=None):
        b, t, e = input_embd.size()
        assert t == self.config.block_size and e == self.config.n_embd, f"Input should be of shape [batch_size, {self.config.block_size}, {self.config.n_embd}], but got {input_embd.shape}"

        # forward the GPT model itself
        x = self.transformer.drop(input_embd)

        for block in self.transformer.h:
            x = block(x, encoder_output = input_embd)

        x = self.transformer.ln_f(x)

        # Calculate logits for all positions
        logits = self.lm_head(x)

        # loss calculation for targets will change according to tokenizer
        if targets is not None:
            if isinstance(targets, str):
                targets = self.tokenizer.encode(targets, add_special_tokens=True, return_tensors='pt').squeeze(0)
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss
    

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = self.get_num_params()
    #     cfg = self.config
    #     L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    #     flops_per_token = 6*N + 12*L*H*Q*T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0/dt) # per second
    #     flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu

    @torch.no_grad()
    def generate(self, image_embedding, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate LateX tokens given image embeddings

        Take a conditioning sequence of input embeddings (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        """
    
        self.eval()
        assert image_embedding.size() == (1, self.config.block_size, self.config.n_embd), \
            f"Expected image embedding of shape (1, {self.config.block_size}, {self.config.n_embd}), but got {image_embedding.size()}"
        
        device = image_embedding.device
        
        # Start with just the image embedding
        text_seq = torch.zeros((1, 0, self.config.n_embd), device=device)

        
        # Initialize an empty tensor to store generated tokens
        generated_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
        
        for i in range(max_new_tokens):

            # Concatenate the image embedding with the current sequence
            current_seq = torch.cat([image_embedding, text_seq], dim=1)

            # ensure the sequence does not exceed block size
            if current_seq.size(1) > self.config.block_size:
                current_seq = current_seq[:, -self.config.block_size:]

            # Forward pass through the model
            logits, _ = self(current_seq)
            
            # Get logits for the next token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the sampled token to our generated sequence
            generated_tokens[:, i] = next_token.view(-1)
            
            # Convert the generated token to its string representation
            generated_text = self.tokenizer.decode(generated_tokens[0, :i+1])
            
            # Prepare for the next iteration:
            # Embed the newly generated token
            next_token_embedding = self.token_embedding(next_token).view(1, 1, -1)
            
            # Concatenate with the current sequence
            current_seq = torch.cat([current_seq, next_token_embedding], dim=1)
            
            # If the sequence is getting too long, remove the oldest token embedding
            if current_seq.size(1) > self.config.block_size:
                current_seq = current_seq[:, 1:]

            # If we've generated an end token or reached max length, stop
            if self.tokenizer.eos_token in generated_text or i == max_new_tokens - 1:
                break
            
        return self.tokenizer.decode(generated_tokens[0, :i+1])
    

    def token_embedding(self, tokens):
        """
        Convert token indices to embeddings.
        This method needs to be implemented based on how we're handling token embeddings in your model.
        """
        # This is a placeholder. You need to implement this based on your model's architecture.
        # It might involve using the weights from self.lm_head or a separate embedding layer.
        
        # If tokens are not already tensor ids, encode them
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, add_special_tokens=True, return_tensors='pt').squeeze(0)
        return self.token_embedding_layer(tokens)