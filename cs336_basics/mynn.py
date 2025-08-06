import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einx

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, bias = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

def SiLU(x):
    return torch.sigmoid(x) * x

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
        gate = SiLU(self.w1(x))  # F.silu等价于Swish(β=1)
        filtered = gate * self.w3(x)
        return self.w2(filtered)

def scaled_dot_product_attention(Q, K, V, mask = None):
    seq_len = Q.shape[-2]
    attn_score = torch.zeros(Q.shape[0], Q.shape[1], seq_len, seq_len, device=Q.device)
    for i in range(seq_len):
        for j in range(seq_len):
            if mask is not None and mask[i, j] == 0:
                attn_score[:, :, i, j] = torch.tensor(float("-inf"), device=Q.device)
            else:
                temp = einx.dot("... h s d, ... h s d -> ... h s", Q[:, :, i, :], K[:, :, j, :])
                attn_score[:, :, i, j] = temp
    attn_score = attn_score / math.sqrt(K.shape[-1])
    attn_score = F.softmax(attn_score, dim=-1)
    return attn_score @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_in, num_heads, rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = Linear(d_in, d_model, bias=False)
        self.k_proj = Linear(d_in, d_model, bias=False)
        self.v_proj = Linear(d_in, d_model, bias=False)
        self.o_proj = Linear(d_model, d_in, bias=False)
        self.rope = rope
    
    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = einx.rearrange("... s (h d) -> ... h s d", q, h=self.num_heads)
        k = einx.rearrange("... s (h d) -> ... h s d", k, h=self.num_heads)
        v = einx.rearrange("... s (h d) -> ... h s d", v, h=self.num_heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = torch.triu(torch.ones(x.shape[-2], x.shape[-2], device=x.device), diagonal=1)
        causal_mask = torch.ones_like(causal_mask) - causal_mask
        causal_mask = causal_mask.bool()
        attn_score = scaled_dot_product_attention(q, k, v, causal_mask)
        attn_score = einx.rearrange("... h s d -> ... s (h d)", attn_score, h=self.num_heads)
        return self.o_proj(attn_score)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + self.eps)
        output = x / rms
        output = output * self.scale
        return output.to(in_dtype)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        inv_freqs = 1 / (theta **(torch.arange(0, d_k, 2, device=device) / d_k))
        freqs = torch.einsum("i,j->ij", torch.arange(max_seq_len, device=device), inv_freqs)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        x_even, x_odd = einx.rearrange("... (d p) -> ... p d", x, p=2).unbind(dim=-2) 
        rot_even = einx.subtract("..., ... -> ...", x_even * cos, x_odd * sin)
        rot_odd = einx.add("..., ... -> ...", x_even * sin, x_odd * cos)
        x_rot = torch.stack((rot_even, rot_odd), dim=-1)
        x_rot = einx.rearrange("... d p -> ... (d p)", x_rot, p=2)
        return x_rot

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    s = e.sum(dim=dim, keepdim=True)
    return e / s


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope=None):
        super().__init__()
        self.rope = rope
        self.mha = MultiHeadAttention(d_model, d_model, num_heads, rope=rope)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, token_positions=None):
        residual = x
        x = self.norm1(x)
        x = self.mha(x, token_positions=token_positions)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.embedding = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, self.rope) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, token_positions=None):
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.norm(x)
        return self.lm_head(x)
    

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, weight_decay=0):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                t = state.get("step", 0)
                p.data -= lr/math.sqrt(t+1) * p.grad
                state["step"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("step", 0) + 1
                m = betas[0] * m + (1 - betas[0]) * p.grad
                v = betas[1] * v + (1 - betas[1]) * p.grad ** 2
                alpha = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= alpha * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * p.data * weight_decay
                state["step"] = t
                state["m"] = m
                state["v"] = v
        return loss


class LinearWarmupCosineAnnealingLR:
    def __init__(self, warmup_steps, cosine_annealing_steps, max_lr, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.cosine_annealing_steps = cosine_annealing_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            lr = self.max_lr * epoch / self.warmup_steps
        elif epoch < self.cosine_annealing_steps:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / (self.cosine_annealing_steps - self.warmup_steps))) / 2
        else:
            lr = self.min_lr
        self.last_epoch = epoch
        return lr

class GradientClipper:
    def __init__(self, max_l2_norm):
        self.max_l2_norm = max_l2_norm

    def __call__(self, parameters):
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.norm(2) ** 2
        total_norm = math.sqrt(total_norm)
        if total_norm > self.max_l2_norm:
            scale = self.max_l2_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad = p.grad * scale