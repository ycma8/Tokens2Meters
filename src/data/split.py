# Category-balanced dataset splitting strategy
# Ensure each category has sufficient samples in eval dataset

import json
import os
from collections import defaultdict, Counter
PROJECT_ROOT = os.getenv("PYTHONPATH")

# Data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "all_data.jsonl")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "rag.jsonl")
SFT_PATH = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "sft.jsonl")
RFT_PATH = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "rft.jsonl")
EVAL_PATH = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "eval.jsonl")

# Category balance configuration
MIN_EVAL_PER_CATEGORY = 15  # Minimum 15 samples per category in eval
TARGET_EVAL_SIZE = 500      # Target eval dataset size

def extract_categories_from_sample(sample):
    """Extract all categories from sample"""
    categories = set()
    assistant_content = sample['messages'][1]['content']
    
    for line_content in assistant_content.split('\n')[1:]:
        if line_content.strip() and line_content.startswith('['):
            parts = line_content.strip()[1:].split(',')
            if len(parts) > 0:
                category = parts[0].strip()
                categories.add(category)
    
    return categories

def stratified_split():
    """Stratified sampling to split dataset"""
    
    # 1. Read all data
    print("Reading data...")
    with open(DATA_PATH, "r") as f:
        all_data = [json.loads(line) for line in f]
    
    print(f"Total data size: {len(all_data)}")
    
    # 2. Organize samples by category
    category_to_samples = defaultdict(list)
    sample_to_categories = {}
    
    for idx, sample in enumerate(all_data):
        categories = extract_categories_from_sample(sample)
        sample_to_categories[idx] = categories
        
        # Each category records this sample
        for category in categories:
            category_to_samples[category].append(idx)
    
    # 3. Statistics of category distribution
    print("\nOriginal dataset category distribution:")
    for category in sorted(category_to_samples.keys()):
        count = len(category_to_samples[category])
        print(f"  {category}: {count} samples")
    
    # 4. Stratified sampling to select eval data
    eval_indices = set()
    category_eval_count = defaultdict(int)
    
    # First ensure minimum samples for each category
    print(f"\nStarting stratified sampling, minimum {MIN_EVAL_PER_CATEGORY} samples per category...")
    
    for category in sorted(category_to_samples.keys()):
        available_samples = category_to_samples[category]
        
        # Calculate required eval samples for this category
        if len(available_samples) < MIN_EVAL_PER_CATEGORY:
            # If too few samples, add all to eval
            needed = len(available_samples)
            print(f"  Warning: {category} only has {len(available_samples)} samples, adding all to eval")
        else:
            needed = MIN_EVAL_PER_CATEGORY
        
        # Select samples, avoid duplicates
        selected = 0
        for sample_idx in available_samples:
            if sample_idx not in eval_indices and selected < needed:
                eval_indices.add(sample_idx)
                category_eval_count[category] += 1
                selected += 1
    
    # 5. If not reached target eval size, continue adding samples
    remaining_samples = [i for i in range(len(all_data)) if i not in eval_indices]
    
    while len(eval_indices) < TARGET_EVAL_SIZE and remaining_samples:
        sample_idx = remaining_samples.pop()
        eval_indices.add(sample_idx)
        
        # Update category count
        categories = sample_to_categories[sample_idx]
        for category in categories:
            category_eval_count[category] += 1
    
    # 6. Split dataset
    eval_data = [all_data[i] for i in eval_indices]
    remaining_data = [all_data[i] for i in range(len(all_data)) if i not in eval_indices]
    
    total_remaining = len(remaining_data)
    train_ratio = 7.0 / (7.0 + 2.5 + 0.4)  # ≈ 0.707
    sft_ratio = 2.5 / (7.0 + 2.5 + 0.4)    # ≈ 0.253
    
    train_size = int(total_remaining * train_ratio)
    sft_size = int(total_remaining * sft_ratio)
    
    train_data = remaining_data[:train_size]
    sft_data = remaining_data[train_size:train_size + sft_size]
    rft_data = remaining_data[train_size + sft_size:]
    
    return train_data, sft_data, rft_data, eval_data, category_eval_count


# Execute stratified sampling
train_data, sft_data, rft_data, eval_data, category_eval_count = stratified_split()

# Save datasets
datasets = [
    (TRAIN_PATH, train_data, "RAG training"),
    (SFT_PATH, sft_data, "SFT training"), 
    (RFT_PATH, rft_data, "RFT training"),
    (EVAL_PATH, eval_data, "Evaluation")
]

for path, data, name in datasets:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{name} dataset: {len(data)} samples -> {path}")

# Output eval dataset category distribution
print(f"\nEval dataset category distribution (Total: {len(eval_data)} samples):")
for category in sorted(category_eval_count.keys()):
    count = category_eval_count[category]
    print(f"  {category}: {count} samples")

# Check if any category has insufficient samples
missing_categories = []
for category, count in category_eval_count.items():
    if count < MIN_EVAL_PER_CATEGORY:
        missing_categories.append((category, count))

if missing_categories:
    print(f"\n⚠️  Warning: The following categories still have insufficient samples:")
    for category, count in missing_categories:
        print(f"    {category}: {count} < {MIN_EVAL_PER_CATEGORY}")
else:
    print(f"\n✅ All categories meet the minimum {MIN_EVAL_PER_CATEGORY} samples requirement!")
