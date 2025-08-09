import argparse
import torch
import numpy as np
import os
import itertools
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

import cs336_basics.mynn as mynn
from cs336_basics.data_utils import Dataset, ValidationDataset, save_checkpoint, load_checkpoint
torch.set_float32_matmul_precision('high')

def get_memory_info(device: str = 'cuda') -> dict:
    """获取内存使用信息"""
    if device == 'cuda' and torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {}

def print_memory_info(device: str = 'cuda', prefix: str = ""):
    """打印内存使用信息"""
    if device == 'cuda' and torch.cuda.is_available():
        mem_info = get_memory_info(device)
        print(f"{prefix}GPU Memory - Allocated: {mem_info['allocated']:.2f}GB, "
              f"Reserved: {mem_info['reserved']:.2f}GB, "
              f"Max Allocated: {mem_info['max_allocated']:.2f}GB")

def create_model(vocab_size: int, context_length: int, d_model: int, num_layers: int, 
                num_heads: int, d_ff: int, rope_theta: float = 10000.0) -> mynn.TransformerLM:
    """创建一个简单的Transformer语言模型"""
    model = mynn.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    return model


def train_step(model: mynn.TransformerLM, batch: tuple, optimizer: torch.optim.Optimizer, 
               loss_fn: mynn.CrossEntropyLoss, device: str = 'cpu', 
               writer: Optional[SummaryWriter] = None, global_step: int = 0, 
               lr_scheduler: mynn.LinearWarmupCosineAnnealingLR = None, 
               gradient_clipper: mynn.GradientClipper = None) -> tuple[float, int]:
    """训练一个batch"""
    model.train()
    optimizer.zero_grad()
    
    inputs, targets = batch
    
    # 前向传播
    logits = model(inputs)  # shape: (batch_size, seq_len, vocab_size)
    
    # 计算损失
    # 将logits和targets重塑为适合交叉熵损失的形式
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch_size * seq_len)
    
    loss = loss_fn(logits_flat, targets_flat)
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    gradient_clipper(model.parameters())
    optimizer.step()
    lr = lr_scheduler.step()

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    # 记录到TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        # 记录梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        writer.add_scalar('Gradient_Norm', total_norm, global_step)
    
    return loss.item(), global_step + 1


def evaluate(model: mynn.TransformerLM, dataloader: DataLoader, 
            loss_fn: mynn.CrossEntropyLoss, device: str = 'cpu',
            writer: Optional[SummaryWriter] = None, global_step: int = 0) -> float:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # 限制评估的batch数量，避免无限循环
        dataloader_iter = iter(dataloader)
        pbar = tqdm(range(len(dataloader)), desc="Evaluating", leave=False, dynamic_ncols=True)
        
        for batch_idx in pbar:
            inputs, targets = next(dataloader_iter)
            
            logits = model(inputs)
            
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            
            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
    avg_loss = total_loss / num_batches
    if writer is not None:
        writer.add_scalar('Loss/Val_Epoch', avg_loss, global_step)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train a simple Transformer language model")
    parser.add_argument('--data_path', type=str, default='data/tinystory/', 
                       help='Path to training data')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feed-forward dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--total_tokens', type=int, default=327680000, help='Total tokens to process (will override epochs if specified)')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_l2_norm', type=float, default=10.0, help='Max L2 norm')
    parser.add_argument('--log_dir', type=str, default='runs', 
                       help='Directory to save TensorBoard logs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval in batches')
    parser.add_argument('--print_interval', type=int, default=1000, help='Print interval in batches')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint interval in batches')
    parser.add_argument('--eval_batches', type=int, default=10, help='Number of batches to evaluate on')
    
    args = parser.parse_args()
    
    # 生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_vocab{args.vocab_size}_ctx{args.context_length}_d{args.d_model}_layers{args.num_layers}_heads{args.num_heads}_ff{args.d_ff}_bs{args.batch_size}_lr{args.lr}"
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.data_path, "checkpoints")
    
    # 创建checkpoint和log目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # 加载数据
    print(f"Loading data from {args.data_path}")
    train_data = np.memmap(os.path.join(args.data_path, "train_token_ids.raw"), dtype=np.uint16, mode="r")
    train_dataset = Dataset(train_data, args.context_length, device=args.device)
    # print(f"Data loaded, shape: {data.shape}")
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # 创建验证数据加载器（使用部分训练数据）
    # val_data = data[:len(data)//10]  # 使用10%的数据作为验证集
    val_data = np.memmap(os.path.join(args.data_path, "val_token_ids.raw"), dtype=np.uint16, mode="r")
    val_dataset = ValidationDataset(val_data, args.context_length, device=args.device)
    val_dataloader = DataLoader(val_dataset, args.batch_size, drop_last=True)
    
    # 计算每个batch处理的token数量
    tokens_per_batch = args.batch_size * args.context_length
    print(f"Tokens per batch: {tokens_per_batch:,}")
    print(f"Dataset size: {len(data):,} tokens")
    print(f"Batches per epoch: {len(train_dataloader):,}")
    
    # 如果指定了total_tokens，计算所需的总batch数
    if args.total_tokens > 0:
        total_batches_needed = args.total_tokens // tokens_per_batch
        print(f"Target total tokens: {args.total_tokens:,}")
        print(f"Total batches needed: {total_batches_needed:,}")
        print(f"Actual total tokens to be processed: {total_batches_needed * tokens_per_batch:,}")
    else:
        # 如果没有指定total_tokens，使用默认的epoch数
        total_batches_needed = len(train_dataloader) * 10  # 默认训练10个epoch
    
    # 创建模型
    print("Creating model...")
    model = create_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff
    )
    model = model.to(args.device)
    
    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
        model = model.to(args.device)
    
    # 使用torch.compile加速
    if hasattr(torch, 'compile'):
        print("Using torch.compile for model acceleration...")
        model = torch.compile(model)
    
    # 记录模型参数到TensorBoard
    writer.add_text('Model/Architecture', str(model), 0)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_scalar('Model/Total_Parameters', total_params, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
    
    # 创建优化器和损失函数
    optimizer = mynn.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = mynn.LinearWarmupCosineAnnealingLR(warmup_steps=args.warmup_steps, cosine_annealing_steps=total_batches_needed, max_lr=args.lr, min_lr=args.lr/10)
    gradient_clipper = mynn.GradientClipper(max_l2_norm=args.max_l2_norm)
    loss_fn = mynn.CrossEntropyLoss()
    
    # 恢复训练（如果指定）
    global_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")
        global_step = start_epoch * len(train_dataloader)
    
    # 训练循环 - 基于batch数量
    print("Starting training...")
    print(f"Training for {total_batches_needed} batches")
    print(f"Evaluation interval: {args.eval_interval} batches")
    print(f"Print interval: {args.print_interval} batches")
    print(f"Save interval: {args.save_interval} batches")
    
    # 创建无限迭代器
    train_iter = itertools.cycle(train_dataloader)
    batch_count = 0
    
    # 使用tqdm显示总体进度
    pbar = tqdm(total=total_batches_needed, desc="Training Progress", dynamic_ncols=True)
    
    # 用于计算平均训练损失
    running_loss = 0.0
    running_count = 0
    
    while batch_count < total_batches_needed:
        batch = next(train_iter)
        
        # 训练一个batch
        loss, global_step = train_step(
            model, batch, optimizer, loss_fn, args.device, 
            writer, global_step, lr_scheduler, gradient_clipper
        )
        
        batch_count += 1
        running_loss += loss
        running_count += 1
        
        # 更新tqdm显示
        avg_loss = running_loss / running_count
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        pbar.update(1)
        
        # 打印损失
        if batch_count % args.print_interval == 0:
            tqdm.write(f"Batch {batch_count}/{total_batches_needed}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            # 重置running loss用于下一个interval
            running_loss = 0.0
            running_count = 0
        
        # 评估
        if batch_count % args.eval_interval == 0:
            tqdm.write(f"\nEvaluating at batch {batch_count}...")
            tqdm.write(f"Current Train Loss: {avg_loss:.4f}")
            
            # 打印评估前的内存使用情况
            # print_memory_info(args.device, "Before evaluation: ")
            
            val_loss = evaluate(model, val_dataloader, loss_fn, device=args.device, writer=writer, global_step=global_step)
            tqdm.write(f"Validation Loss: {val_loss:.4f}")
            
            # 打印评估后的内存使用情况
            # print_memory_info(args.device, "After evaluation: ")
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Val_Step', val_loss, global_step)
        
        # 保存checkpoint
        if batch_count % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_batch_{batch_count}.pt')
            save_checkpoint(model, optimizer, batch_count, checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")
    
    pbar.close()
    
    # 保存最终checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_final.pt')
    save_checkpoint(model, optimizer, batch_count, final_checkpoint_path)
    tqdm.write(f"Final checkpoint saved: {final_checkpoint_path}")
    del optimizer, lr_scheduler, gradient_clipper, train_dataloader, data
    
    # 最终评估
    tqdm.write("\nFinal evaluation...")
    final_val_loss = evaluate(model, val_dataloader, loss_fn, device=args.device, writer=writer, global_step=global_step)
    tqdm.write(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # 关闭TensorBoard writer
    writer.close()
    tqdm.write("Training completed!")
    tqdm.write(f"TensorBoard logs saved to: {log_dir}")
    tqdm.write(f"Run 'tensorboard --logdir={log_dir}' to view the logs")


if __name__ == "__main__":
    main()
