#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate RAG-based SFT data
Retrieve images similar to input images from vector database and generate new SFT data
"""

import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from pymilvus import MilvusClient

from src.rag.vector_db.model_vectorization import FashionEmbeddingModel
# No longer use functions from retrieval_util, directly use MilvusClient
PROJECT_ROOT = os.getenv("PYTHONPATH")
def main():
    # Connect to Milvus vector database
    uri = os.path.join(PROJECT_ROOT, "output", "vector_db", "nu.db")
    
    client = MilvusClient(
        uri=uri,
    )
    
    # Configure model paths
    model_path = os.path.join(PROJECT_ROOT, "output", "blip_finetune", "model_weights", "best_model.pth")
    processor_path = os.path.join(PROJECT_ROOT, "models", "blip")
    
    # Initialize model
    print("Loading model...")
    model = FashionEmbeddingModel(model_path, processor_path)
    print("Model loading completed!")

    # Read jsonl file
    input_file = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "eval.jsonl")
    output_file = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "rag_eval.jsonl")
    image_dir = os.path.join(PROJECT_ROOT, "dataset", "nuscenes")  # Base directory for images
    
    # Read data
    print(f"Reading data file: {input_file}")
    data_items = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # Ensure line is not empty
                data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} data items")
    
    # Generate new SFT data
    new_data_items = []
    
    for idx, item in enumerate(tqdm(data_items, desc="Processing data")):
        try:
            # Get original image path
            if not item.get("images") or len(item["images"]) == 0:
                print(f"Skipping data item without image: {idx}")
                continue
                
            original_image_path = os.path.join(image_dir, item["images"][0])
            
            # Check if image file exists
            if not os.path.exists(original_image_path):
                print(f"Original image file does not exist: {original_image_path}")
                continue
            
            # Get original answer
            if not item.get("messages") or len(item["messages"]) < 2:
                print(f"Skipping data item with insufficient messages: {idx}")
                continue
                
            
            original_system = item["messages"][0]["content"]
            original_answer = item["messages"][2]["content"]
            
            # Load image
            image = Image.open(original_image_path).convert('RGB')
            
            # Generate image vector
            image_embedding = model.encode_images_batch([image])[0]
            
            # Retrieve most similar images from vector database
            # Directly use MilvusClient for retrieval, specify correct metric type
            search_params = {
                "collection_name": "nu",
                "data": [image_embedding.tolist()],
                "anns_field": "image_embedding",
                "search_params": {"metric_type": "COSINE"},  # Use COSINE metric type
                "limit": 1,
                "output_fields": ["pk", "image_path", "ans"]
            }
            search_results = client.search(**search_params)[0]  # Get result list of first query
            
            if not search_results:
                print(f"Search results empty: {idx}")
                continue
            
            # Get image path and answer from search results
            retrieval_result = search_results[0]
            retrieval_image_path = retrieval_result.get('image_path', '')
            retrieval_answer = retrieval_result.get('ans', '')
            
            if not retrieval_image_path or not retrieval_answer:
                print(f"Search results missing necessary information: {idx}")
                continue
            
            # Create new SFT data format
            new_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": original_system
                    },
                    {
                        "role": "user", 
                        "content": f"<image>\n<image>\nThere are two images: let A be the first image (images[0]) and B be the second image (images[1]).\nI already know the 3D detections for A:\n{retrieval_answer}\n\nUsing both images as context, <answer>answer ONLY for B (second image)</answer>. Do not repeat A's results.Your reasoning process must less than 1024 words."
                    },
                    {
                        "role": "assistant", 
                        "content": original_answer
                    }
                ],
                "images": [retrieval_image_path, original_image_path]
            }
            
            new_data_items.append(new_item)
            
            # Save every 100 processed items
            if (idx + 1) % 100 == 0:
                with open(output_file, 'w') as f:
                    for new_item in new_data_items:
                        f.write(json.dumps(new_item) + '\n')
                print(f"Saved {len(new_data_items)} data items")
            
        except Exception as e:
            print(f"Error processing data item {idx}: {str(e)}")
    
    # Save final results
    print(f"Saving final results to: {output_file}")
    with open(output_file, 'w') as f:
        for new_item in new_data_items:
            f.write(json.dumps(new_item) + '\n')
    
    print(f"Processing completed! Generated {len(new_data_items)} new SFT data items")

if __name__ == "__main__":
    main()