import torch
FP16_BYTES = 2
FP32_BYTES = 4

def count_flops(vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int) -> int:
    # 1. Embedding
    flops = vocab_size * d_model

    # 2. Positional Encoding
    flops += context_length * d_model

    # 3. Transformer Blocks
    for _ in range(num_layers):
        # 3.1 Multi-Head Attention
        flops += 12 * context_length * d_model * d_model
        # 3.2 Feed-Forward Network
        flops += 2 * context_length * d_model * d_ff
        # 3.3 RMSNorm
        flops += 2 * context_length * d_model

    # 4. Final RMSNorm
    flops += 2 * context_length * d_model

    return flops

def count_model_size(vocab_size: int, num_layers: int, d_model: int, d_ff: int) -> int:
    embedding_size = vocab_size * d_model
    attention_size = 3 * d_model * d_model
    feed_forward_size = 2 * d_ff * d_model
    rmsnorm_size = 2 * d_model
    transformer_block_size = attention_size + feed_forward_size + rmsnorm_size
    size = embedding_size + transformer_block_size * num_layers + rmsnorm_size
    return size

def count_activation_memory_per_sample(vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int) -> int:
    memory = 0
    embedding_memory = vocab_size * d_model
    transformer_block_memory = 0
    attention_memory = context_length * 3 * d_model
    rmsnorm_memory = context_length * d_model
    feed_forward_memory = context_length * d_model + context_length * d_ff
    transformer_block_memory = attention_memory + feed_forward_memory + rmsnorm_memory
    memory = embedding_memory + transformer_block_memory * num_layers + rmsnorm_memory
    return memory

def count_adamw_memory(vocab_size: int, batch_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int) -> int:
    model_weights_memory = count_model_size(vocab_size, num_layers, d_model, d_ff)
    optimizer_memory = 2 * model_weights_memory
    memory = model_weights_memory + optimizer_memory
    memory += count_activation_memory_per_sample(vocab_size, context_length, num_layers, d_model, d_ff) * batch_size
    return memory

if __name__ == "__main__":
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    d_ff = 6400
    batch_size = 1
    model_weights_memory = count_model_size(vocab_size, num_layers, d_model, d_ff) * FP32_BYTES / 1e9
    optimizer_memory = 2 * model_weights_memory
    activation_memory = count_activation_memory_per_sample(vocab_size, context_length, num_layers, d_model, d_ff) * batch_size * FP32_BYTES / 1e9
    adamw_memory = model_weights_memory + optimizer_memory + activation_memory
    print(count_flops(vocab_size, context_length, num_layers, d_model, d_ff)*FP32_BYTES/1e12, 'TFLOPS')
    print(model_weights_memory, 'GB')
    print(activation_memory, 'GB')
    print(adamw_memory, 'GB')
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        adamw_memory = count_adamw_memory(vocab_size, batch_size, context_length, num_layers, d_model, d_ff) * FP32_BYTES / 1e9
        print(f"adamw memory: {model_weights_memory:.2f} + {optimizer_memory:.2f} + {activation_memory:.2f} * {batch_size} = {adamw_memory:.2f}")
