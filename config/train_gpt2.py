# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# logging
log_file = 'training_log.txt'
sample_interval = 100  # Log sample predictions every 100 iterations
# I/O
out_dir = 'out'
eval_interval = 1200
log_interval = 20
eval_iters = 90
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'image2latex'
gradient_accumulation_steps = 1 #8*4 for 8 GPUs # used to simulate larger batch sizes
batch_size = 8   # if gradient_accumulation_steps > 1, this is the MICRO-BATCH SIZE
block_size = 300 # max token length
# model
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 60000 # total number of training iterations
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 45000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
num_epochs = 100
max_length = 300
max_n = 4 # max n-gram for BLEU score
subset_size = 150000