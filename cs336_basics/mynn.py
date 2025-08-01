import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = Linear(input_dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, input_dim, bias=False)
        self.w3 = Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x):
        # 分割输入为两部分（假设输入维度为偶数）
        # assert x.shape[-1] % 2 == 0, "输入维度需为偶数"
        # a, b = x.chunk(2, dim=-1)
        
        # Swish门控计算
        gate = F.silu(self.w1(x))  # F.silu等价于Swish(β=1)
        filtered = gate * self.w3(x)
        return self.w2(filtered)

def scaled_dot_product_attention(Q, K, V, mask = None):
    attn_score = Q @ K.transpose(-2, -1)
    attn_score = attn_score / math.sqrt(K.shape[-1])
    if mask is not None:
        attn_score = attn_score.masked_fill(mask == 0, float("-inf"))
    attn_score = F.softmax(attn_score, dim=-1)
    return attn_score @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_in, d_k, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_in, d_model, bias=False)
        self.k_proj = Linear(d_in, d_model, bias=False)
        self.v_proj = Linear(d_in, d_model, bias=False)
        self.o_proj = Linear(d_model, d_in, bias=False)

