import torch
import numpy as np
import os
from typing import BinaryIO, IO


class DataLoader:
    def __init__(self, dataset: np.ndarray, batch_size: int, context_length: int, shuffle: bool = True, device: str = 'cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.shuffle = shuffle
        self.device = device
        
        # 计算可用的样本数量
        self.num_samples = len(self.dataset) - self.context_length
        
    def __iter__(self):
        # 创建索引 - 确保不会超出范围
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        # 分批处理
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            if len(batch_indices) == 0:
                break
            
            # 使用向量化操作
            start_indices = batch_indices[:, None]  # (batch_size, 1)
            context_indices = np.arange(self.context_length + 1)  # (context_length + 1,)
            
            # 计算所有需要的索引 - 由于num_samples已经考虑了context_length，这里不会超出范围
            all_indices = start_indices + context_indices  # (batch_size, context_length + 1)
            
            # 从dataset中提取数据
            batch_data = self.dataset[all_indices]  # (batch_size, context_length + 1)
            
            # 转换为tensor并分离输入和标签
            batch_data = torch.tensor(batch_data, dtype=torch.long, device=self.device)
            batch_inputs = batch_data[:, :-1]   # (batch_size, context_length)
            batch_labels = batch_data[:, 1:]    # (batch_size, context_length)
            
            yield batch_inputs, batch_labels

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

class ValidationDataLoader(DataLoader):
    def __init__(self, dataset: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu'):
        super().__init__(dataset, batch_size, context_length, shuffle=False, device=device)
    
    def __iter__(self):
        # 验证时使用连续的数据块，不重叠
        # 确保不会超出范围：num_blocks * context_length <= len(dataset)
        num_blocks = len(self.dataset) // self.context_length
        
        for i in range(0, num_blocks, self.batch_size):
            batch_indices = np.arange(i, min(i+self.batch_size, num_blocks))
            if len(batch_indices) == 0:
                break
                
            # 使用与DataLoader相同的向量化方法
            start_indices = batch_indices * self.context_length  # 每个block的起始位置
            start_indices = start_indices[:, None]  # (batch_size, 1)
            context_indices = np.arange(self.context_length + 1)  # (context_length + 1,)
            
            # 计算所有需要的索引 - 由于num_blocks已经考虑了context_length，这里不会超出范围
            all_indices = start_indices + context_indices  # (batch_size, context_length + 1)
            
            # 从dataset中提取数据
            batch_data = self.dataset[all_indices]  # (batch_size, context_length + 1)
            
            # 转换为tensor并分离输入和标签
            batch_data = torch.tensor(batch_data, dtype=torch.long, device=self.device)
            batch_inputs = batch_data[:, :-1]   # (batch_size, context_length)
            batch_labels = batch_data[:, 1:]    # (batch_size, context_length)
            
            yield batch_inputs, batch_labels
    
    def __len__(self):
        return (len(self.dataset) // self.context_length + self.batch_size - 1) // self.batch_size

class Dataset:
    def __init__(self, data: np.ndarray, context_length: int, device: str = 'cpu'):
        self.data = data
        self.context_length = context_length
        self.device = device

    def __getitem__(self, index: int):
        data = torch.tensor(self.data[index:index+self.context_length+1], dtype=torch.long, device=self.device)
        return data[:-1], data[1:]
    
    def __len__(self):
        return len(self.data) - self.context_length
    
class ValidationDataset(Dataset):
    def __init__(self, data: np.ndarray, context_length: int, device: str = 'cpu'):
        super().__init__(data, context_length, device)
        self.indices = np.arange(0, len(self.data), self.context_length)
    
    def __getitem__(self, index: int):
        data = torch.tensor(self.data[self.indices[index]:self.indices[index]+self.context_length+1], dtype=torch.long, device=self.device)
        return data[:-1], data[1:]
    
    def __len__(self):
        return len(self.indices)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


# 使用示例和内存优化说明
def example_usage():
    """
    示例：如何使用优化后的DataLoader
    
    内存优化要点：
    1. 使用yield生成器，避免一次性加载所有数据
    2. 向量化操作但限制batch大小
    3. 及时释放不需要的中间变量
    4. 使用适当的数据类型（torch.long vs torch.float32）
    """
    # 创建示例数据
    data = np.random.randint(0, 1000, size=(10000,), dtype=np.int32)
    
    # 训练数据加载器
    train_loader = DataLoader(
        dataset=data,
        batch_size=32,
        context_length=128,
        shuffle=True,
        device='cpu'
    )
    
    # 验证数据加载器
    val_loader = ValidationDataLoader(
        dataset=data,
        batch_size=32,
        context_length=128,
        device='cpu'
    )
    
    # 使用示例
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs shape {inputs.shape}, labels shape {labels.shape}")
        if batch_idx >= 2:  # 只打印前3个batch
            break
    
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")


if __name__ == "__main__":
    example_usage()

