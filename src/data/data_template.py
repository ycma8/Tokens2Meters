#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from tqdm.auto import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility
PROJECT_ROOT = os.getenv("PYTHONPATH")

def simplify_category_name(full_category_name):
    """
    Simplify category name: use second-level category name if exists, otherwise use first-level
    Examples:
    - "vehicle.bus.bendy" -> "bus"
    - "vehicle.car" -> "car"  
    - "animal" -> "animal"
    """
    parts = full_category_name.split('.')
    if len(parts) >= 2:
        return parts[1]  # Second-level category name
    else:
        return parts[0]  # Original name
DATAROOT = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "v1.0-trainval_meta")
VERSION  = "v1.0-trainval"
IMG_ROOT_OUT = "samples"
SPLITS = ["train", "val"]
OUT_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "all_data.jsonl")

CLS2ID = {
    "pedestrian": 0,  # human.pedestrian.* all map to this
    "animal": 1,
    "car": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "bus": 5,  # vehicle.bus.* all map to this
    "truck": 6,
    "construction": 7,  # vehicle.construction
    "emergency": 8,  # vehicle.emergency.*
    "trailer": 9,
    "barrier": 10,  # movable_object.barrier
    "trafficcone": 11,  # movable_object.trafficcone
    "pushable_pullable": 12,  # movable_object.pushable_pullable
    "debris": 13,  # movable_object.debris
    "bicycle_rack": 14,  # static_object.bicycle_rack
}


def ann_to_extra(ann_list):
    num_obj = len(ann_list)
    cls_ids = [CLS2ID[a["category"]] for a in ann_list]
    dists   = [a["distance"] for a in ann_list]
    sizes   = [a["size"]     for a in ann_list]  # [ [l,w,h], ... ]
    return num_obj, cls_ids, dists, sizes


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def build_annotation_dict(box: Box,
                          ego_translation: np.ndarray,
                          ego_quat: Quaternion):
    obj_center_global = np.array(box.center)
    center_ego = ego_quat.inverse.rotate(obj_center_global - ego_translation)

    dist = float(np.linalg.norm(center_ego[:2]))

    return {
        "category": simplify_category_name(box.name),
        "distance": round(dist, 3),
        "size": [round(x, 3) for x in box.wlh],
    }

def compute_yaw_cam(box):
    """
    Return yaw relative to camera coordinate system (in radians).
    Definition: In camera coordinates, X axis points right, Z axis points forward,
    yaw is the angle from +Z to forward vector in XZ plane (counterclockwise).
    Note: Using atan2(fwd_x, fwd_z), 0 means facing +Z, +pi/2 means facing +X (right side of image).
    """
    fwd_local = np.array([1.0, 0.0, 0.0], dtype=float)       # Object local forward (along length direction)
    fwd_cam   = box.orientation.rotate(fwd_local)            # Rotate to camera coordinate system
    yaw = np.arctan2(fwd_cam[0], fwd_cam[2])                 # atan2(x, z)
    return float(wrap_angle(yaw))

def build_annotation_dict_cam(box: Box):
    # Boxes are already in camera coordinate system, no need to transform again!
    center_cam = np.array(box.center, dtype=float)

    # Distance definition: if you want "true 3D Euclidean distance", use the line below;
    # if you want "camera forward depth" (more intuitive for CV "depth"), use center_cam[2] as distance.
    dist_euclid = float(np.linalg.norm(center_cam))
    # dist_depth  = float(center_cam[2])  # Optional: forward depth
    yaw_cam = compute_yaw_cam(box)   
    return {
        "category": simplify_category_name(box.name),
        "distance": round(dist_euclid, 3),       # or replace with dist_depth
        "size":     [round(x, 3) for x in box.wlh],
        "cam_coords": [round(x, 3) for x in center_cam],
        "yaw_cam": round(yaw_cam, 6)
    }


def convert_sample_cam(nusc: NuScenes, cam_sd_token: str):
    sd_rec = nusc.get('sample_data', cam_sd_token)
    _, boxes, cam_intrinsic = nusc.get_sample_data(cam_sd_token, box_vis_level=BoxVisibility.ANY)

    ann_list = [build_annotation_dict_cam(box) for box in boxes]

    rel_img = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", IMG_ROOT_OUT, "/".join(sd_rec["filename"].split("/")[1:]))
    return rel_img, ann_list

def convert_to_answer_with_tags(ann_list, keep_text=True, include_coords=False):
    n = len(ann_list)
    if n == 0:
        return f"There are 0 objects in the current view."

    # Sort by distance from nearest to farthest
    sorted_anns = sorted(ann_list, key=lambda a: a["distance"])
    
    lines = [f"There are {n} objects in the current view, from nearest to farthest:"]
    for idx, ann in enumerate(sorted_anns, 1):
        l, w, h = ann["size"]
        x, y, z = ann["cam_coords"]
        line = (
            f"[{ann['category']}, {x}, {y}, {z}, {l}, {w}, {h}, {ann['yaw_cam']}]"
        )
        lines.append(line)
    return "\n".join(lines)

def ann_to_extra(ann_list, max_obj=20, pad_val=-1):
    """
    ann_list: [{"category": str, "distance": float, "size": [l,w,h]}, ...]
    max_obj : Maximum number of objects to keep
    pad_val : Padding value (default -1)

    Returns:
        cls_ids  : [max_obj]
        reg      : [max_obj * reg_per_obj]   (here reg_per_obj = 1(dist) + len(size))
        cls_mask : [max_obj]                 (1=valid, 0=padded/cropped)
        reg_mask : [max_obj * reg_per_obj]
    """
    if len(ann_list) == 0:
        reg_per_obj = 4  # Fallback
    else:
        reg_per_obj = 1 + len(ann_list[0]["size"])  # 1 distance + size dimensions

    # You can customize filtering strategy: here example prioritizes by distance
    ann_sorted = sorted(ann_list, key=lambda a: a.get("distance", 1e9))
    sel = ann_sorted[:max_obj]  # Crop if exceeds limit

    # --- cls ---
    cls_ids = [CLS2ID[a["category"]] for a in sel]
    cls_mask = [1] * len(sel)
    if len(sel) < max_obj:
        cls_ids += [pad_val] * (max_obj - len(sel))
        cls_mask += [0] * (max_obj - len(sel))

    # --- reg (dist + size) ---
    dists = [a["distance"] for a in sel]
    sizes = [dim for a in sel for dim in a["size"]]          # Flatten
    reg = dists + sizes                                      # [len(sel)*reg_per_obj]
    reg_mask = [1] * len(reg)

    total_reg_dim = max_obj * reg_per_obj
    if len(reg) < total_reg_dim:
        pad_len = total_reg_dim - len(reg)
        reg += [pad_val] * pad_len
        reg_mask += [0] * pad_len
    else:
        # Theoretically if sel equals max_obj, len(reg)==total_reg_dim, won't enter here
        reg = reg[:total_reg_dim]
        reg_mask = reg_mask[:total_reg_dim]

    return cls_ids, reg

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

def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    all_items = []
    for sample in tqdm(nusc.sample, desc="Iterate samples"):
        # 6 camera tokens
        for cam in ["CAM_FRONT"]:
        # for cam in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        #             "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"]:
            cam_token = sample["data"][cam]
            rel_img, ann_list = convert_sample_cam(nusc, cam_token)
            if len(ann_list) > 12:
                continue

            user_prompt = "<image>List all objects in the image along with their category, distance from the camera, and 3D dimensions."
            # Set include_coords=True to include coordinates in camera coordinate system
            assistant_answer = convert_to_answer_with_tags(ann_list, include_coords=True)
            cls_ids, reg = ann_to_extra(ann_list)
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_answer}
                ],
                "images": [rel_img]
            }
            all_items.append(item)

    train_items = all_items

    def dump_jsonl(path, items):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump_jsonl(OUT_TRAIN, train_items)

    print(f"Done! Train: {len(train_items)} -> {OUT_TRAIN}")

if __name__ == "__main__":
    main()
