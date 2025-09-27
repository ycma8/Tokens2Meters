import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipForImageTextRetrieval, AutoProcessor
from torch.optim import AdamW
import torch.nn.functional as F
from lora_layer import add_lora_to_linear_layer, get_lora_params, save_lora_weights, load_lora_weights
import json
from datetime import datetime
import os
PROJECT_ROOT = os.getenv("PYTHONPATH")

class NuScenesDataset(Dataset):
    def __init__(self, data_file, processor, image_dir):
        self.data = []
        # Read jsonl file
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.image_dir = image_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Get image path - from first element of images field
        image_path = os.path.join(self.image_dir, item["images"][0])
        # Get text - extract from assistant's response
        text = item["messages"][1]["content"]  # assistant's response
        
        try:
            # Open and convert image
            image = Image.open(image_path).convert('RGB')
            # Multi-modal processor call
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Remove batch dimension
            for k,v in inputs.items():
                inputs[k] = v.squeeze(0)
                
            return inputs
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return empty sample or skip this sample
            # Here simply recursively call next index
            if idx + 1 < len(self):
                return self.__getitem__((idx + 1) % len(self))
            else:
                raise e

def collate_fn(batch):
    # Get maximum length in batch
    max_length = max([b['input_ids'].size(0) for b in batch])
    
    processed_batch = {
        'input_ids': [],
        'attention_mask': [],
        'pixel_values': []
    }
    
    for item in batch:
        # Process input_ids
        padding_length = max_length - item['input_ids'].size(0)
        padded_input_ids = F.pad(item['input_ids'], (0, padding_length), value=processor.tokenizer.pad_token_id)
        processed_batch['input_ids'].append(padded_input_ids)
        
        # Process attention_mask
        padded_attention_mask = F.pad(item['attention_mask'], (0, padding_length), value=0)
        processed_batch['attention_mask'].append(padded_attention_mask)
        
        # Process pixel_values
        processed_batch['pixel_values'].append(item['pixel_values'])
    
    # Convert lists to tensors
    processed_batch['input_ids'] = torch.stack(processed_batch['input_ids'])
    processed_batch['attention_mask'] = torch.stack(processed_batch['attention_mask'])
    processed_batch['pixel_values'] = torch.stack(processed_batch['pixel_values'])
    
    return processed_batch

def compute_loss(vision_embeds, text_embeds, temperature=0.07):
    """Compute contrastive loss"""
    # Normalize features
    vision_embeds = F.normalize(vision_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(vision_embeds, text_embeds.transpose(0, 1)) / temperature
    
    # Create labels (diagonal as positive examples)
    labels = torch.arange(len(logits), device=logits.device)
    
    # Compute image-to-text and text-to-image loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
    
    # Total loss is average of two directional losses
    total_loss = (loss_i2t + loss_t2i) / 2
    return total_loss

def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    # Record metrics for each batch
    batch_metrics = []
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # Get visual features
        vision_outputs = model.vision_model(batch["pixel_values"])
        # Get last hidden state of visual features, i.e., image features, 768 dimensions
        vision_embeds = vision_outputs.last_hidden_state[:, 0, :]
        
        # Get text features
        text_outputs = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        # Get last hidden state of text features, i.e., text features, 768 dimensions
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        
        # Use projection layer to convert both image and text vectors to 256-dimensional vectors
        vision_embeds = model.vision_proj(vision_embeds) 
        text_embeds = model.text_proj(text_embeds)
        
        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(vision_embeds, text_embeds, dim=1)
        avg_cosine_similarity = cosine_similarity.mean().item()
        
        # Compute loss
        loss = compute_loss(vision_embeds, text_embeds)
        # Zero gradients to prevent gradient accumulation
        optimizer.zero_grad()
        # Backward propagation
        loss.backward()
        # Update parameters
        optimizer.step()
        
        batch_loss = loss.item() # Loss value
        total_loss += batch_loss # Accumulate loss value
        
        # Record current batch metrics
        batch_metrics.append({
            'epoch': epoch + 1,
            'batch': batch_idx + 1,
            'loss': batch_loss,
            'cosine_similarity': avg_cosine_similarity
        })
        
        progress_bar.set_postfix({
            'loss': batch_loss,
            'cosine_sim': f'{avg_cosine_similarity:.4f}'
        })
    
    return total_loss / len(train_loader), batch_metrics # Return average loss and metrics for each batch

if __name__ == "__main__":
    # Set random seed, multiple runs will produce identical results
    # Convenient for debugging and verification: exclude random interference, focus on algorithm itself
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed(42)
    
    # Create directory to save results
    results_dir = os.path.join(PROJECT_ROOT, "output", "blip_finetune")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for model and LoRA weight saving
    model_weights_dir = os.path.join(results_dir, "model_weights")
    lora_weights_dir = os.path.join(results_dir, "lora_weights")
    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(lora_weights_dir, exist_ok=True)
    
    # Data files and image directory
    data_file = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "rag.jsonl")
    image_dir = os.path.join(PROJECT_ROOT, "dataset", "nuscenes")  # Image directory path
    
    # Save training configuration
    config = {
        'num_epochs': 50,
        'learning_rate': 5e-4,
        'batch_size': 64,
        'lora_rank': 16,
        'lora_alpha': 32,
        'data_file': data_file,
        'image_dir': image_dir,
        'seed': 42
    }
    
    with open(f'{results_dir}/training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Model and processor paths
    blip_path = os.path.join(PROJECT_ROOT, "models", "blip")
    
    # Initialize model and processor
    model = BlipForImageTextRetrieval.from_pretrained(blip_path)
    processor = AutoProcessor.from_pretrained(blip_path, use_fast=True)
    
    # Add LoRA layers
    target_modules = [
        "query",
        "value",
        "vision_proj",
        "text_proj"
    ]
    
    model = add_lora_to_linear_layer(
        model,
        target_modules,
        rank=16,
        alpha=32
    )
    
    # Move entire model to CUDA
    model = model.cuda()
    
    # Freeze original parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Only train LoRA parameters
    lora_params = get_lora_params(model)
    for param in lora_params:
        param.requires_grad = True
    
    # Create dataset and data loader
    train_dataset = NuScenesDataset(data_file, processor, image_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4  # Use multiprocessing to load data
    )
    
    # Training configuration
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']  # Can use larger learning rate for LoRA parameters
    
    # Optimizer (only optimize LoRA parameters)
    optimizer = AdamW(lora_params, lr=learning_rate)
    
    # Training loop
    history = {
        'epoch': [],
        'train_loss': [],
        'batch_metrics': [],
    }

    # Only keep best model
    best_loss = float('inf')
    best_model_path = os.path.join(model_weights_dir, "best_model.pth")
    best_lora_path = os.path.join(lora_weights_dir, "best_lora.pth")

    # Print dataset size
    print(f"Training dataset size: {len(train_dataset)}")
    
    for epoch in range(config['num_epochs']):
        # Train one epoch
        train_loss, batch_metrics = train_one_epoch(model, train_loader, optimizer, epoch)
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['batch_metrics'].extend(batch_metrics)
        
        # Save batch metrics for this epoch (including cosine similarity)
        batch_df = pd.DataFrame(batch_metrics)
        batch_df.to_csv(f'{results_dir}/batch_metrics_epoch_{epoch+1}.csv', index=False)
        
        # Only save best model weights
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)
            save_lora_weights(model, best_lora_path)
            print(f"New best model saved at epoch {epoch+1} loss value {train_loss:.4f}")
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'Training loss: {train_loss:.4f}')
        print('-' * 50)
    
    # Save summary metrics for each epoch
    epoch_df = pd.DataFrame({
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
    })
    epoch_df.to_csv(f'{results_dir}/epoch_metrics.csv', index=False)
    
    print("Training completed!")
    print(f"All training results saved to: {results_dir}/")
