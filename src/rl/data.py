import os
from datasets import load_dataset, Features, Sequence, Value, Image as HFImage
from PIL import Image
from transformers import AutoProcessor
PROJECT_ROOT = os.getenv("PYTHONPATH")

SYSTEM_PROMPT = """
You are a vision model for single-image 3D object detection. The answer format is as follows:
- First line: `There are N objects in the current view, from nearest to farthest:`
- Then N lines: `[category, x, y, z, width, length, height, yaw]`
- Right-handed. Origin at the camera center.
- +X points to the image right; +Y points to the image bottom; +Z points forward along the optical axis.
- Rotation around the vertical axis (looking from above, i.e., along −Y).
- yaw = 0 when the box faces +Z; yaw > 0 rotates toward +X (counter-clockwise when viewed from above).Range: (−π, π].
- x,y,z,width,length,height in meters; yaw in radians.
- Categories: {pedestrian, animal, car, motorcycle, bicycle, bus, truck, construction, emergency, trailer, barrier, trafficcone, pushable_pullable, debris, bicycle_rack}. Don’t invent new ones.
- Sort by Euclidean distance, nearest to farthest.
- If none: `There are 0 objects in the current view.`
"""

SAVE_DIR  = os.path.join(PROJECT_ROOT, "dataset", "rl", "rft_nothink")
ds = load_dataset(
    "json",
    data_files=os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "rft.jsonl"),
    split="train",
)
ds = ds.cast_column("images", Sequence(HFImage(decode=True)))
print(ds)
print(ds[0]["images"][0])
model_id = os.path.join(PROJECT_ROOT, "models", "vanilla")
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
def resize_keep_aspect(img, target_size=448):
    w, h = img.size
    scale = target_size / max(w, h)
    return img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
def to_trl_examples(example):
    msgs = example["messages"]
    user_text = msgs[0]["content"]
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation)
    img = example["images"][0]

    ans = msgs[1]["content"]
    return {
        "prompt": prompt,
        "image": img,
        "solution": ans,
    }

dataset_train = ds.map(
    to_trl_examples,
    remove_columns=ds.column_names,  # Drop original columns, keep only new columns
)
print(dataset_train[0])

dataset_train.save_to_disk(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")


