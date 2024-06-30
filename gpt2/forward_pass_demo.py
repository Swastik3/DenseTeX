import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

# Set random seed for reproducibility
torch.manual_seed(42)

# Define model configuration
config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=300,
    vocab_size=50257,
    dropout=0.1
)

# Initialize the model
model = GPT(config)

# Generate random input data
batch_size = 4
seq_length = config.block_size
input_shape = (batch_size, seq_length, config.n_embd)
random_input = torch.randn(input_shape)

# Generate random target data
target_shape = (batch_size, seq_length)
random_target = torch.randint(0,config.vocab_size, target_shape)

# Set the model to evaluation mode
model.eval()

# Perform a forward pass
with torch.no_grad():
    logits, loss = model(random_input, targets=random_target)

# Print the results
print(f'sequence length : {seq_length}')
print(f"Input shape: {random_input.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Loss: {loss.item()}")

# Calculate probabilities
probs = F.softmax(logits, dim=-1)

print(logits)

# Print the probability distribution for the first token of the first sequence
print("\nProbability distribution for the first token of the first sequence:")
top_probs, top_indices = torch.topk(probs[0, 0], k=50)
for prob, idx in zip(top_probs, top_indices):
    print(f"Token {idx}: {prob.item():.4f}")
