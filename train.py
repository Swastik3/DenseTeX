"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from contextlib import nullcontext
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Dataset  
from model import GPTConfig, GPT
from torchvision import models
from DenseNet import PositionalEncoding2D, InputEmbeddings
from transformers import AutoTokenizer
from torchvision.models import DenseNet169_Weights
from torch.utils.data import Subset
from DataLoader import CustomDataset
import tqdm
from DataLoader import get_dataloader

# HYPERPARAMETERS

# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1
log_interval = 1
eval_iters = 5
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # 5 * 8 # used to simulate larger batch sizes
batch_size = 5 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 300 # max token length
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster


# --------------------------DO NOT CHANGE---------------------------------------
# configuration parameters that are allowed to be overridden from command line
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# --------------------------DO NOT CHANGE---------------------------------------
# setting up the environment for distributed data parallel training 
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# -----------------------------------------------------------------------------


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9



# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=300,
                  bias=bias, dropout=dropout) # start with model_args from command line


# --------------------------DO NOT CHANGE---------------------------------------

# initialization of model based on the arg init_from (resume, scratch, gpt2, etc.)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)


elif init_from == 'resume':

    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']


# ----------------------------------------------------------------------------


# # crop down the model block size if desired, using model surgery
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value


# move the model to the correct device
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# ---------------------------------------------------------------------------


# Define the DenseNet169 model
densenet_model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1) # change to DEFAULT HERE

# Remove the final fully connected layer to get the final feature maps
densenet_model = nn.Sequential(*list(densenet_model.children())[:-1])
densenet_model.add_module('PositionalEncoding2D', PositionalEncoding2D(1664, 12, 25)) # hardcoded this based on denseNet output size
densenet_model.add_module('InputEmbeddings', InputEmbeddings(1664, 768))

# Move the DenseNet model to the correct device
densenet_model = densenet_model.to(device)

# Wrap the original model to include the DenseNet model as the first layers
class CombinedModel(nn.Module):
    def __init__(self, densenet_model, original_model):
        super(CombinedModel, self).__init__()
        self.densenet_model = densenet_model
        self.original_model = original_model

    def forward(self, images, targets):
        embeddings = self.densenet_model(images)
        outputs = self.original_model(input_embd=embeddings, targets=targets)
        return outputs

# Replace the original model with the combined model
num_epochs = 3
global_step = 0
max_length = 300

# INITALIZE THE TOKENIZER HERE !!!!!!
tokenizer = model.tokenizer
# tokenizer = AutoTokenizer.from_pretrained("witiko/mathberta")

def tokenize_latex(latex_text, max_length):
    toks = tokenizer(latex_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = toks['input_ids']
    attention_mask = toks['attention_mask']
    
    targets = input_ids.clone()
    # Shift targets to the right, filling in with pad token
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = tokenizer.pad_token_id
    
    return input_ids, attention_mask, targets

# def tokenize_latex(latex_text, max_length) :

#     toks = OurTokenizer(latex_text, padding = 'max_length', truncation = True, max_length = max_length, return_tensors = 'pt')
#     input_ids = toks.input_ids # shape : (batch_size, sequence_length)

#     attention_mask = toks.attention_mask # 1 for real tokens, 0 for padding tokens

#     targets = input_ids.clone()

#     targets[:, :-1] = input_ids[:, 1:]
#     targets[:, -1] = -100 # ignore the last token in prediction

#     return input_ids, attention_mask, targets


# get the dataloader
train_loader = get_dataloader(batch_size=batch_size, image_dir='../data/UniMER-1M/images/', label_file='../data/UniMER-1M/train.txt')
# val_loader = get_dataloader(batch_size=batch_size, image_dir='../data/UniMER-Test/spe/', label_file='../data/UniMER-Test/spe.txt')

# get a very small subset of the entire dataset 
# subset_size = 1024

# def get_subset_dataloader(batch_size=batch_size, image_dir = '../data/UniMER-1M/images/', label_file = '../data/UniMER-1M/train.txt', subset_size = subset_size) :

#     full_dataset = CustomDataset(image_dir = image_dir, label_file = label_file, transform = None, cache_file = 'valid_indices_cache.pkl')

#     # random select a subset of indices
#     subset_indices = random.sample(range(len(full_dataset)), subset_size)
#     #create a subset dataset
#     subset_dataset = Subset(full_dataset, subset_indices)
#     # create a dataloader for the subset
#     subset_dataloader = DataLoader(subset_dataset, batch_size = batch_size, shuffle = True)

#     return subset_dataloader

# # subset train_loader
# train_loader = get_subset_dataloader(batch_size=batch_size, image_dir='../data/UniMER-1M/images/', label_file='../data/UniMER-1M/train.txt', subset_size=subset_size)


# Evaluation function
@torch.no_grad()

# # calculate loss on train and val sets
# def evaluate(inp_model , train_loader, val_loader, device, eval_iters=100):
#     inp_model.eval()
#     results = {}

#     for split, loader in [("train", train_loader), ("val", val_loader)]:
#         total_loss = 0
#         for i, (images, latex_labels) in enumerate(loader):
#             if i >= eval_iters:
#                 break
            
#             images = images.to(device)
            
#             # Tokenize LaTeX labels
#             input_ids, attention_mask, targets = tokenize_latex(latex_labels, max_length=300)
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             targets = targets.to(device)
            
#             # Forward pass
#             # inp_model = CombinedModel(densenet_model, model)
#             outputs = inp_model(images=images, targets=targets)
#             loss = outputs[1] if isinstance(outputs, tuple) else outputs.loss
            
#             total_loss += loss.item()
        
#         avg_loss = total_loss / min(eval_iters, len(loader))
#         results[split] = avg_loss

#     inp_model.train()
#     return results


# ----------------- DO NOT CHANGE -------------------------------------
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -------------------------------------------------------------------------

# training loop 
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0


# max_batches = 100

model = CombinedModel(densenet_model, model)
model.train()

for epoch in range(num_epochs):

    for batch_idx, (images, latex_labels) in enumerate(train_loader):

    # for batch_idx, (images, latex_labels) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):

        # if batch_idx >= max_batches:
        #     break
        
        # Get the image embeddings and the latex labels

        # print('1')
        images = images.to(device)
        
        # Tokenize LaTeX labels
        input_ids, attention_mask, targets = tokenize_latex(latex_labels, max_length=300)

        # print('2')

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate

        # print('3')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing


        # # if iter_num % eval_interval == 0 and master_process:
        # model.eval()
        # with torch.no_grad():
        #         outputs = model(images=images, targets=targets)
        #         loss = outputs[1]
        #     losses = evaluate(model, train_loader, val_loader, device = device, eval_iters = eval_iters)
        #     print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #     model.train()

            
        #     if wandb_log:
        #         wandb.log({
        #             "iter": iter_num,
        #             "train/loss": losses['train'],
        #             "val/loss": losses['val'],
        #             "lr": lr,
        #         })
            
        #     # if losses['val'] < best_val_loss or always_save_checkpoint:
        #     #     best_val_loss = losses['val']
        #     #     if iter_num > 0:
        #     #         checkpoint = {
        #     #             'model': model.state_dict(),
        #     #             'optimizer': optimizer.state_dict(),
        #     #             'model_args': model_args,
        #     #             'iter_num': iter_num,
        #     #             'best_val_loss': best_val_loss,
        #     #             'config': config,
        #     #         }
        #     #         print(f"saving checkpoint to {out_dir}")
        #     #         torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        # Forward backward update, with optional gradient accumulation
        for micro_step in range(gradient_accumulation_steps):

            if ddp:

                # print('6')
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable


                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)


            # Forward pass
            with ctx:

                outputs = model(images=images, targets=targets)

                if isinstance(outputs, tuple):
                    logits, loss = outputs

                else :
                    logits, loss = outputs, None


                sample_prediction = torch.multinomial(logits[0].softmax(dim=-1), num_samples=1)

                non_pad_mask = sample_prediction != tokenizer.pad_token_id
                decoded_prediction = tokenizer.decode(sample_prediction[non_pad_mask])

                print(f"Sample non-padded prediction: {decoded_prediction}")
                print(f"Actual label: {latex_labels[0]}")


                loss = outputs[1] / gradient_accumulation_steps 

        
            # Backward pass
            scaler.scale(loss).backward()
        
        # Clip the gradient
        if grad_clip != 0.0:


            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Step the optimizer and scaler if training in fp16

        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1


        if iter_num % log_interval == 0 and master_process:


            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)

            lossf = loss.item() * gradient_accumulation_steps


            # if local_iter_num >= 5 :
                # mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)

                # running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        
        iter_num += 1
        local_iter_num += 1
        
        # termination condition
        if iter_num > max_iters:
            break
    

print(f"Training completed. Total iterations: {iter_num}")


if ddp:
    destroy_process_group()