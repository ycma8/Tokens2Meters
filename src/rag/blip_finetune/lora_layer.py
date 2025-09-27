import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA矩阵A和B
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # LoRA路径: x -> A -> B
        # 确保所有张量在同一设备上
        device = x.device
        self.lora_A = self.lora_A.to(device)
        self.lora_B = self.lora_B.to(device)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return lora_output

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=32):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        # 原始路径 + LoRA路径
        return self.linear(x) + self.lora(x)

def add_lora_to_linear_layer(model, target_module_names, rank=4, alpha=32):
    """
    将模型中的指定线性层替换为带有LoRA的层
    
    Args:
        model: 原始模型
        target_module_names: 要替换的模块名称列表
        rank: LoRA的秩
        alpha: LoRA的缩放因子
    """
    for name, module in model.named_modules():
        if any(target_name in name for target_name in target_module_names):
            if isinstance(module, nn.Linear):
                # 保存原始权重
                original_weights = module.weight.data.clone()
                original_bias = module.bias.data.clone() if module.bias is not None else None
                
                # 创建新的带LoRA的层
                new_layer = LinearWithLoRA(
                    nn.Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    ),
                    rank=rank,
                    alpha=alpha
                )
                
                # 恢复原始权重
                new_layer.linear.weight.data = original_weights
                if original_bias is not None:
                    new_layer.linear.bias.data = original_bias
                
                # 替换原始层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                setattr(parent_module, child_name, new_layer)
    
    return model

def get_lora_params(model):
    """获取所有LoRA参数"""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params

def save_lora_weights(model, path):
    """保存LoRA权重"""
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    torch.save(lora_state_dict, path)

def load_lora_weights(model, path):
    """加载LoRA权重"""
    lora_state_dict = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    return model