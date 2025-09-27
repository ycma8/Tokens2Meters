
import os 

from datasets import load_dataset,Dataset,load_from_disk
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from tqdm import tqdm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from datasets import load_dataset,Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info
from trl import GRPOConfig, GRPOTrainer
from transformers import (
AutoModelForCausalLM, 
AutoTokenizer, 
Qwen2_5_VLForConditionalGeneration, 
AutoProcessor,
BitsAndBytesConfig
)
import torch
import deepspeed
from datasets import load_dataset, Features, Sequence, Value, Image as HFImage
from PIL import Image
import numpy as np
import re
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import torch.nn.functional as F
PROJECT_ROOT = os.getenv("PYTHONPATH")

compute_dtype = getattr(torch, "float16")
processor = AutoProcessor.from_pretrained(os.path.join(PROJECT_ROOT, "models", "vanilla"), use_fast=True, padding_side="left")

LOAD_DIR = os.path.join(PROJECT_ROOT, "dataset", "rl", "rft_nothink")
ds = load_from_disk(LOAD_DIR)

def object_reward_func(completions, solution, **kwargs):
    """
    Simplified GRPO reward function
    Based on weighted combination of distance(70%), size(20%), orientation(10%)
    """
    
    def parse_objects_new_format(text):
        """Parse object description"""
        count_pattern = r'There (?:is|are) (\d+) objects? in the current view'
        count_match = re.search(count_pattern, text, re.IGNORECASE)
        
        declared_count = 0
        if count_match:
            declared_count = int(count_match.group(1))
        
        objects = []
        pattern = r'\[([^,]+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
        
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                category = match[0].strip()
                x, y, z = float(match[1]), float(match[2]), float(match[3])
                length, width, height = float(match[4]), float(match[5]), float(match[6])
                yaw = float(match[7])
                
                objects.append({
                    "category": category,
                    "position": np.array([x, y, z], dtype=np.float32),
                    "dimensions": np.array([abs(length), abs(width), abs(height)], dtype=np.float32),
                    "yaw": yaw
                })
            except (ValueError, IndexError):
                continue
        
        return declared_count, objects
    
    def compute_similarity(pred_obj, gt_obj):
        """
        Compute similarity, ensure all components are in 0-1 range
        Weights: distance 70%, size 20%, orientation 10%
        """
        # 1. Distance similarity (0-1)
        distance = np.linalg.norm(pred_obj["position"] - gt_obj["position"])
        distance_sim = np.exp(-distance/10)  # 10-meter characteristic distance
        
        # 2. Size similarity (0-1) 
        pred_dims = pred_obj["dimensions"]
        gt_dims = gt_obj["dimensions"] 
        # Compute similarity for each dimension: min(a/b, b/a), ensure 0-1
        dim_sims = np.minimum(pred_dims / (gt_dims + 1e-6), gt_dims / (pred_dims + 1e-6))
        size_sim = np.mean(dim_sims)
        
        # 3. Orientation similarity (0-1)
        yaw_diff = abs(pred_obj["yaw"] - gt_obj["yaw"])
        # Handle angle periodicity
        yaw_diff = min(yaw_diff, 2*np.pi - yaw_diff) 
        orientation_sim = np.exp(-yaw_diff/(np.pi/6))  # π/6 characteristic angle
        
        # Weighted combination: 7:2:1
        total_sim = 0.7 * distance_sim + 0.2 * size_sim + 0.1 * orientation_sim
        
        return total_sim

    # ---- Main loop ----
    rewards = []
    
    for comp_msg, gt_txt in zip(completions, solution):
        pred_declared_count, pred_objs = parse_objects_new_format(comp_msg)
        gt_declared_count, gt_objs = parse_objects_new_format(gt_txt)
        
        # Special case: both have 0 objects
        if len(pred_objs) == 0 and len(gt_objs) == 0:
            rewards.append(1.0)
            continue
        
        # If prediction or ground truth is empty
        if len(pred_objs) == 0 or len(gt_objs) == 0:
            rewards.append(0.0)
            continue
        
        # Group by class for matching
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for i, obj in enumerate(pred_objs):
            pred_by_class[obj["category"].lower()].append(i)
        for j, obj in enumerate(gt_objs):
            gt_by_class[obj["category"].lower()].append(j)
        
        all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())
        all_similarities = []
        
        for cls in all_classes:
            pred_ids = pred_by_class.get(cls, [])
            gt_ids = gt_by_class.get(cls, [])
            
            if len(pred_ids) == 0 or len(gt_ids) == 0:
                continue  # Skip unmatched classes

            # Build similarity matrix
            similarity_matrix = np.zeros((len(pred_ids), len(gt_ids)))
            for i, pred_idx in enumerate(pred_ids):
                for j, gt_idx in enumerate(gt_ids):
                    similarity_matrix[i, j] = compute_similarity(pred_objs[pred_idx], gt_objs[gt_idx])

            # Hungarian algorithm to find optimal matching
            cost_matrix = 1.0 - similarity_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # Collect matched similarities
            for i, j in zip(row_indices, col_indices):
                all_similarities.append(similarity_matrix[i, j])
        
        # Compute average similarity
        if len(all_similarities) == 0:
            rewards.append(0.0)
        else:
            rewards.append(float(np.mean(all_similarities)))
    
    return rewards

output_dir=os.path.join(PROJECT_ROOT, "output", "nothink")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    os.path.join(PROJECT_ROOT, "models", "vanilla"),
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, os.path.join(PROJECT_ROOT, "output", "sft", "adapter"))
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in
    num_train_epochs=3,
    bf16=True,
    # Parameters that control the data preprocessing
    per_device_train_batch_size=16,
    max_completion_length=1024,  # default: 256
    num_generations=8,  # default: 8
    gradient_accumulation_steps=1,
    max_prompt_length=None,
    # Parameters related to reporting and saving
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    temperature=0.95,
    top_p=0.7,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[object_reward_func],
    args=training_args,
    train_dataset=ds,
)

trainer.train()