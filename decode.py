import torch
from cs336_basics.mynn import TransformerLM
from cs336_basics.tokenizer import Tokenizer
import os


def generate_text(model, tokenizer, text, max_new_tokens=100):
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long, device="cuda")
    tokens = tokens.unsqueeze(0)
    while tokens.shape[1] < max_new_tokens:
        output = model(tokens)
        next_token = torch.argmax(output[0, -1, :], dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    return tokenizer.decode(tokens[0].tolist())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feed-forward dimension')
    args = parser.parse_args()
    
    tokenizer = Tokenizer.from_files(Tokenizer, os.path.join(args.data_path, "vocab.pkl"), os.path.join(args.data_path, "merges.pkl"), ["<|endoftext|>"])
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0
    )
    checkpoint = torch.load(args.model_path)
    model = torch.compile(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to("cuda")

    while True:
        text = input("Enter a text: ")
        if text == "":
            break
        generated_text = generate_text(model, tokenizer, text, max_new_tokens=100)
        print(generated_text)