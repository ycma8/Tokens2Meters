import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForImageTextRetrieval, AutoProcessor
import os
import sys
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
sys.path.append('/root/autodl-tmp/multimodel_fasion_retrieval')
from src.rag.blip_finetune.lora_layer import add_lora_to_linear_layer, load_lora_weights


class FashionEmbeddingModel:
    def __init__(self, model_path: str, processor_path: str, device: str = 'cuda'):
        """
        初始化时尚向量化模型
        
        Args:
            model_path: 训练好的模型权重路径
            processor_path: BLIP处理器路径
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
        
        # 加载原始模型
        self.model = BlipForImageTextRetrieval.from_pretrained(processor_path)
        
        # 添加LoRA层（需要与训练时保持一致）
        target_modules = ["query", "value", "vision_proj", "text_proj"]
        self.model = add_lora_to_linear_layer(
            self.model,
            target_modules,
            rank=16,
            alpha=32
        )
        
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"模型已加载到 {device} 设备")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        对单张图片进行向量化编码
        
        Args:
            image: 图片路径或PIL Image对象
            
        Returns:
            numpy数组，形状为 (256,) 的向量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 预处理图像
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移动到设备
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            # 获取视觉特征
            vision_outputs = self.model.vision_model(pixel_values)
            vision_embeds = vision_outputs.last_hidden_state[:, 0, :]  # 取CLS token
            
            # 使用投影层降维到256维
            vision_embeds = self.model.vision_proj(vision_embeds)
            
            # 归一化
            vision_embeds = F.normalize(vision_embeds, dim=-1)
        
        return vision_embeds.cpu().numpy().squeeze()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        对文本进行向量化编码
        
        Args:
            text: 输入文本
            
        Returns:
            numpy数组，形状为 (256,) 的向量
        """
        # 预处理文本
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移动到设备
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # 获取文本特征
            text_outputs = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeds = text_outputs.last_hidden_state[:, 0, :]  # 取CLS token
            
            # 使用投影层降维到256维
            text_embeds = self.model.text_proj(text_embeds)
            
            # 归一化
            text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds.cpu().numpy().squeeze()
    
    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        批量对PIL图片对象进行向量化编码
        
        Args:
            images: PIL Image对象列表
            
        Returns:
            numpy数组，形状为 (len(images), 256)
        """
        if not images:
            return np.array([])
            
        # 预处理批次
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            # 获取视觉特征
            vision_outputs = self.model.vision_model(pixel_values)
            vision_embeds = vision_outputs.last_hidden_state[:, 0, :]
            
            # 使用投影层
            vision_embeds = self.model.vision_proj(vision_embeds)
            
            # 归一化
            vision_embeds = F.normalize(vision_embeds, dim=-1)
            
        return vision_embeds.cpu().numpy()
    
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量对文本进行向量化编码
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy数组，形状为 (len(texts), 256)
        """
        if not texts:
            return np.array([])
            
        # 预处理批次
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # 获取文本特征
            text_outputs = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            
            # 使用投影层
            text_embeds = self.model.text_proj(text_embeds)
            
            # 归一化
            text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds.cpu().numpy()
    
    def compute_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """
        计算图片和文本嵌入之间的相似度
        
        Args:
            image_embedding: 图片嵌入向量
            text_embedding: 文本嵌入向量
            
        Returns:
            相似度分数 (0-1之间)
        """
        # 确保向量是归一化的
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        
        # 计算余弦相似度
        similarity = np.dot(image_embedding, text_embedding)
        return float(similarity)
    
    def find_similar_images(self, query_text: str, image_embeddings: np.ndarray, 
                           image_paths: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据文本查询查找最相似的图片
        
        Args:
            query_text: 查询文本
            image_embeddings: 图片嵌入矩阵
            image_paths: 图片路径列表
            top_k: 返回前k个结果
            
        Returns:
            [(图片路径, 相似度分数), ...] 的列表
        """
        # 编码查询文本
        query_embedding = self.encode_text(query_text)
        
        # 计算相似度
        similarities = np.dot(image_embeddings, query_embedding)
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((image_paths[idx], float(similarities[idx])))
        
        return results
    
    def find_similar_texts(self, query_image: Union[str, Image.Image], 
                          text_embeddings: np.ndarray, texts: List[str], 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据图片查询查找最相似的文本
        
        Args:
            query_image: 查询图片路径或PIL Image对象
            text_embeddings: 文本嵌入矩阵
            texts: 文本列表
            top_k: 返回前k个结果
            
        Returns:
            [(文本, 相似度分数), ...] 的列表
        """
        # 编码查询图片
        query_embedding = self.encode_image(query_image)
        
        # 计算相似度
        similarities = np.dot(text_embeddings, query_embedding)
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((texts[idx], float(similarities[idx])))
        
        return results


def create_dataset_embeddings(csv_path: str, model_path: str, processor_path: str, 
                            output_dir: str = "embeddings_output"):
    """
    为整个数据集创建嵌入向量
    
    Args:
        csv_path: 数据集CSV文件路径
        model_path: 训练好的模型权重路径
        processor_path: BLIP处理器路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"数据集大小: {len(df)}")
    
    # 初始化模型
    model = FashionEmbeddingModel(model_path, processor_path)
    
    # 生成图片嵌入
    print("生成图片嵌入...")
    image_paths = df['path'].tolist()
    image_embeddings = model.encode_batch_images(image_paths, batch_size=16)
    
    # 生成文本嵌入
    print("生成文本嵌入...")
    texts = df['caption'].tolist()
    text_embeddings = model.encode_batch_texts(texts, batch_size=32)
    
    # 保存嵌入
    print("保存嵌入...")
    np.save(os.path.join(output_dir, 'image_embeddings.npy'), image_embeddings)
    np.save(os.path.join(output_dir, 'text_embeddings.npy'), text_embeddings)
    
    # 保存元数据
    metadata = {
        'image_paths': image_paths,
        'texts': texts,
        'image_ids': df['image_id'].tolist(),
        'product_ids': df['product_id'].tolist(),
        'genders': df['gender'].tolist(),
        'product_types': df['product_type'].tolist(),
        'image_types': df['image_type'].tolist()
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"嵌入已保存到 {output_dir}")
    print(f"图片嵌入形状: {image_embeddings.shape}")
    print(f"文本嵌入形状: {text_embeddings.shape}")
    
    return image_embeddings, text_embeddings, metadata


if __name__ == "__main__":
    # 配置路径
    model_path = "../../training_results_20250705_165112/model_weights/best_model.pth"
    processor_path = "/root/autodl-fs/blip-image-captioning-base/"
    csv_path = "../../dataset/DeepFashion-MultiModal/labels_front.csv"
    
    # 创建数据集嵌入
    image_embeddings, text_embeddings, metadata = create_dataset_embeddings(
        csv_path, model_path, processor_path
    )
    
    # 示例使用
    model = FashionEmbeddingModel(model_path, processor_path)
    
    # 文本到图片检索示例
    print("\n=== 文本到图片检索示例 ===")
    query_text = "red dress"
    similar_images = model.find_similar_images(
        query_text, image_embeddings, metadata['image_paths'], top_k=3
    )
    
    print(f"查询文本: {query_text}")
    for i, (image_path, score) in enumerate(similar_images):
        print(f"  {i+1}. {image_path} (相似度: {score:.4f})")
    
    # 图片到文本检索示例
    print("\n=== 图片到文本检索示例 ===")
    query_image = metadata['image_paths'][0]  # 使用第一张图片作为查询
    similar_texts = model.find_similar_texts(
        query_image, text_embeddings, metadata['texts'], top_k=3
    )
    
    print(f"查询图片: {query_image}")
    for i, (text, score) in enumerate(similar_texts):
        print(f"  {i+1}. {text[:100]}... (相似度: {score:.4f})") 