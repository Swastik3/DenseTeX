import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer
from DenseNet import PositionalEncoding2D, InputEmbeddings
from DataLoader import get_dataloader
from model import GPTConfig, GPT

def test_training_setup():
    print("Starting training setup test...")

    # Test device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test data loading
    try:
        train_loader = get_dataloader(batch_size=2, image_dir='/Users/marmik/UniMER-1M/images/', label_file='/Users/marmik/UniMER-1M/train.txt')
        images, latex_labels = next(iter(train_loader))
        print(f"Data loading successful. Image shape: {images.shape}, Labels: {latex_labels[:2]}")
    except Exception as e:
        print(f"Error in data loading: {e}")
        return

    # Test tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("witiko/mathberta")
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Test tokenize_latex function
    def tokenize_latex(latex_text, max_length):
        toks = tokenizer(latex_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = toks['input_ids']
        attention_mask = toks['attention_mask']
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]
        targets[:, -1] = tokenizer.pad_token_id
        return input_ids, attention_mask, targets

    try:
        input_ids, attention_mask, targets = tokenize_latex(latex_labels[:2], max_length=300)
        
        print(f"Tokenization successful. Input IDs shape: {input_ids.shape}")
    except Exception as e:
        print(f"Error in tokenization: {e}")
        return

    # Test model initialization
    try:
        gptconf = GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=300, bias=False, dropout=0.1)
        gpt_model = GPT(gptconf)
        print("GPT model initialized successfully")

        densenet_model = models.densenet169(pretrained=True)
        densenet_model = nn.Sequential(*list(densenet_model.children())[:-1])
        densenet_model.add_module('PositionalEncoding2D', PositionalEncoding2D(1664, 800, 400))
        densenet_model.add_module('InputEmbeddings', InputEmbeddings(1664, 768))
        print("DenseNet model initialized successfully")
    except Exception as e:
        print(f"Error in model initialization: {e}")
        return
    
    # Test CombinedModel
    class CombinedModel(nn.Module):
        def __init__(self, densenet_model, original_model):
            super(CombinedModel, self).__init__()
            self.densenet_model = densenet_model
            self.original_model = original_model

        def forward(self, images, targets):
            embeddings = self.densenet_model(images)
            outputs = self.original_model(input_embd=embeddings, targets=targets)
            return outputs

    try:
        combined_model = CombinedModel(densenet_model, gpt_model)
        combined_model.to(device)
        print("Combined model created and moved to device successfully")
    except Exception as e:
        print(f"Error in creating combined model: {e}")
        return

    # Test forward pass
    try:
        images = images.to(device)
        targets = targets.to(device)
        outputs = combined_model(images, targets=targets)
        print(f"Forward pass successful. Output shape: {outputs.loss.shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return

    # Test optimizer
    try:
        optimizer = torch.optim.AdamW(combined_model.parameters(), lr=1e-4)
        print("Optimizer created successfully")
    except Exception as e:
        print(f"Error in creating optimizer: {e}")
        return

    # Test mini training loop
    try:
        combined_model.train()
        for i in range(3):
            optimizer.zero_grad()
            outputs = combined_model(images=images, targets=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Mini-batch {i+1}, Loss: {loss.item()}")
        print("Mini training loop completed successfully")
    except Exception as e:
        print(f"Error in mini training loop: {e}")
        return

    print("All tests completed successfully!")

if __name__ == "__main__":
    test_training_setup()