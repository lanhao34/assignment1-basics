import numpy as np
NUM_SSD_CHANNELS = 8
SSD_CHANNEL_SIZE = 16 # KB
from pprint import pprint

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--num_layers', type=int, default=32)
    args = parser.parse_args()
    KV_SIZE = args.hidden_size * 2 * 2 / 1024 # KB
    num_channels_on_layer = {i: [] for i in range(args.num_layers)}
    current_channel = 0
    current_offset = 0
    num_tokens = 16
    for _ in range(num_tokens):
        for i in range(args.num_layers):
            span_channels = int((current_offset + KV_SIZE) // SSD_CHANNEL_SIZE)
            current_offset = (current_offset + KV_SIZE) % SSD_CHANNEL_SIZE
            used_channels = [channel % NUM_SSD_CHANNELS for channel in range(current_channel, current_channel + span_channels + (1 if current_offset else 0))]
            num_channels_on_layer[i] += [used_channels]
            current_channel += span_channels
            # print("layer", i, "current_channel", current_channel, "current_offset", current_offset, "span_channels", span_channels, "used_channels", used_channels)
            # if current_offset == 0:
            #     print("-"*10)
    # pprint(num_channels_on_layer)
    channel_count = np.zeros((args.num_layers, NUM_SSD_CHANNELS, num_tokens))
    print(channel_count.shape)
    for layer_idx, layer in enumerate(num_channels_on_layer.values()):
        for token_idx, token_channels in enumerate(layer):
            for channel in token_channels:
                channel_count[layer_idx][channel][token_idx] += 1
    # pprint(channel_count)
    for i in range(args.num_layers):
        # sum in token dim
        per_channel_usage = channel_count[i].sum(axis=-1)
        per_channel_usage = per_channel_usage / per_channel_usage.sum()
        print(i, per_channel_usage)