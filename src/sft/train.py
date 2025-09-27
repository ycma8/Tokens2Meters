import os, json
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, set_seed
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
PROJECT_ROOT = os.getenv("PYTHONPATH")
MODEL = os.path.join(PROJECT_ROOT, "models", "vanilla")
OUT   = os.path.join(PROJECT_ROOT, "output", "sft")
DATA  = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "sft.jsonl")
BF16  = True
CUT   = 20000
BATCH = 4
ACCUM = 8
EPOCH = 3.0
LR    = 5e-5
SCHED = "cosine"
OPTIM = "adamw_torch"
MAX_N = 100000
LOG   = 5
SAVE  = 100
WARM  = 0
PACK  = False
FREEZE_VISION = True
FREEZE_PROJECTOR = True
SEED = 42

LORA_R, LORA_A, LORA_D, LORA_T = 8, 16, 0.0, "all-linear"

set_seed(SEED)

class JsonlChat(Dataset):
    def __init__(self, data_dir: str, max_n: int):
        self.rows=[]
        if data_dir.endswith(".jsonl"):
            with open(data_dir, "r", encoding="utf-8") as f:
                for line in f:
                    self.rows.append(json.loads(line))
                    if len(self.rows)>=max_n: break
        if not self.rows: raise FileNotFoundError("no *.jsonl in dataset/")
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def load_images(ex: Dict[str,Any]):
    imgs = []
    if "images" in ex and ex["images"]:
        for img_path in ex["images"]:
            full_path = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", img_path)
            if os.path.exists(full_path):
                imgs.append(Image.open(full_path).convert("RGB"))
            else:
                print(f"warning: image file not found: {full_path}")
    return imgs

def to_chat(msgs: List[Dict[str,Any]], has_images: bool = False):
    new = []
    for m in msgs:
        items = []
        content = m["content"]
        
        if isinstance(content, str):
            items.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    items.append({"type": "text", "text": c.get("text", "")})
                elif c.get("type") == "image":
                    items.append({"type": "image"})
        
        if has_images and m["role"] == "user" and not any(item.get("type") == "image" for item in items):
            items.append({"type": "image"})
            
        new.append({"role": m["role"], "content": items})
    return new

class Collator:
    def __init__(self, processor, cutoff): self.p, self.cut = processor, cutoff
    def __call__(self, batch):
        texts, imgs = [], []
        for ex in batch:
            msgs = ex["messages"]
            has_images = "images" in ex and ex["images"]
            
            chat_format = to_chat(msgs, has_images)
            text = self.p.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=False)[:self.cut]
            texts.append(text)
            
            if has_images:
                imgs.append(load_images(ex))
            else:
                imgs.append([])
        
        processed_texts, processed_imgs = [], []
        for text, img_list in zip(texts, imgs):
            if img_list:
                processed_texts.append(text)
                processed_imgs.append(img_list)
        
        if not processed_texts:
            raise ValueError("no images in batch")
            
        out = self.p(text=processed_texts, images=processed_imgs, return_tensors="pt", add_special_tokens=False, padding=True)
        out["labels"] = out["input_ids"].clone()
        return out

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    trust_remote_code=True, low_cpu_mem_usage=True,
)

for n,p in model.named_parameters():
    if FREEZE_VISION and ("vision" in n or "visual" in n): p.requires_grad=False
    if FREEZE_PROJECTOR and ("projector" in n): p.requires_grad=False

peft_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_A, lora_dropout=LORA_D,
                      target_modules=LORA_T, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_cfg)

dataset = JsonlChat(DATA, MAX_N)
collate = Collator(processor, CUT)
args = SFTConfig(
    output_dir=OUT, per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
    learning_rate=LR, num_train_epochs=EPOCH, lr_scheduler_type=SCHED, optim=OPTIM,
    warmup_steps=WARM, logging_steps=LOG, save_steps=SAVE,
    bf16=BF16, fp16=not BF16, max_grad_norm=1.0,
    remove_unused_columns=False, max_length=CUT, packing=PACK
)

trainer = SFTTrainer(
    model=model, args=args, train_dataset=dataset, data_collator=collate
)

trainer.train()
trainer.model.save_pretrained(os.path.join(OUT, "adapter"))
processor.save_pretrained(OUT)