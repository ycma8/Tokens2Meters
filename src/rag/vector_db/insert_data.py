#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NuScenes data vector database insertion tool
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pymilvus import MilvusClient, DataType
from model_vectorization import FashionEmbeddingModel
PROJECT_ROOT = os.getenv("PYTHONPATH")

def ensure_collection(client: MilvusClient, coll: str):

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field(field_name="image_embedding", datatype=DataType.FLOAT_VECTOR, dim=256)
    schema.add_field(field_name="text_embedding", datatype=DataType.FLOAT_VECTOR, dim=256)
    # 也可以把元数据显式建字段（或走动态字段 $meta）：
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="ans", datatype=DataType.VARCHAR, max_length=2048)

    index = client.prepare_index_params()
    index.add_index(field_name="image_embedding", index_type="FLAT", metric_type="COSINE")
    index.add_index(field_name="text_embedding", index_type="FLAT", metric_type="COSINE")

    client.create_collection(collection_name=coll, schema=schema, index_params=index)


def main():
    client = MilvusClient(
        uri=os.path.join(PROJECT_ROOT, "output", "vector_db", "nu.db"),
    )
    # Configure model paths
    model_path = os.path.join(PROJECT_ROOT, "output", "blip_finetune", "model_weights", "best_model.pth")
    processor_path = os.path.join(PROJECT_ROOT, "models", "blip")
    
    # Initialize model
    print("Loading model...")
    model = FashionEmbeddingModel(model_path, processor_path)
    print("Model loading completed!")

    # Read jsonl file
    data_file = os.path.join(PROJECT_ROOT, "dataset", "nuscenes", "rag.jsonl")
    image_dir = os.path.join(PROJECT_ROOT, "dataset", "nuscenes")  # Base directory for images
    ensure_collection(client, "nu")
    # Read data
    print(f"Reading data file: {data_file}")
    data_items = []
    with open(data_file, 'r') as f:
        for line in f:
            data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} data items")
    
    # Batch processing parameters
    batch_size = 32  # Batch size, can be adjusted based on GPU memory
    
    # Iterate through data and insert into vector database
    success_count = 0
    error_count = 0
    
    # Process data in batches
    for i in range(0, len(data_items), batch_size):
        batch_items = data_items[i:i+batch_size]
        batch_images = []
        batch_texts = []
        batch_metadata = []
        
        # Collect batch data
        for item in batch_items:
            try:
                # Get image path - from first element of images field
                if not item.get("images") or not item["images"]:
                    print(f"Skipping data item without image: {item.get('id', 'unknown')}")
                    continue

                image_path = os.path.join(image_dir, item["images"][0])
                
                # Get text content - extract from assistant's response
                if not item.get("messages") or len(item["messages"]) < 2:
                    print(f"Skipping data item with insufficient messages: {item.get('id', 'unknown')}")
                    continue
                    
                text_content = item["messages"][1]["content"]  # assistant's response
                
                # Generate primary key - using filename (without path and extension)
                file_name = os.path.basename(image_path)
                primary_key = os.path.splitext(file_name)[0]  # Remove extension
                
                # Check if image file exists
                if not os.path.exists(image_path):
                    print(f"Image file does not exist: {image_path}")
                    error_count += 1
                    continue
                
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Add to batch
                batch_images.append(image)
                batch_texts.append(text_content)
                batch_metadata.append({
                    'pk': primary_key,
                    'image_path': image_path,
                    'ans': text_content
                })
                    
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                error_count += 1
        
        # If batch has valid data
        if batch_images:
            try:
                # Batch encode images
                image_embeddings = model.encode_images_batch(batch_images)
                
                # Batch encode texts
                text_embeddings = model.encode_texts_batch(batch_texts)
                
                # Prepare batch insertion data
                batch_data = []
                for i, metadata in enumerate(batch_metadata):
                    batch_data.append({
                        'pk': metadata['pk'],
                        'image_embedding': image_embeddings[i],
                        'text_embedding': text_embeddings[i],
                        'image_path': metadata['image_path'],
                        'ans': metadata['ans']
                    })
                
                # Batch insert into vector database
                if batch_data:
                    res = client.insert(
                        collection_name="nu",
                        data=batch_data
                    )
                    
                    success_count += len(batch_data)
                    print(f"Successfully inserted {success_count} records")
                    
            except Exception as e:
                print(f"Error during batch processing: {str(e)}")
                error_count += len(batch_images)
    
    # Output summary information
    print("\nData insertion completed!")
    print(f"Total data: {len(data_items)}")
    print(f"Successfully inserted: {success_count}")
    print(f"Processing failed: {error_count}")

if __name__ == "__main__":
    main()