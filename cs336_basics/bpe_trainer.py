import regex as re
import multiprocessing as mp
from collections import Counter, OrderedDict
from functools import reduce, partial
from tqdm import tqdm, trange
import time
import psutil
import os
from typing import BinaryIO, Optional, Iterable
from queue import Empty

mp.set_start_method('spawn', force=True)
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pre_tokenize_chunk(chunk):
    # print(len(chunk))
    texts = re.split("<|endoftext|>", chunk.decode("utf-8", errors="ignore"))
    word_counts = Counter()
    for text in texts:
        word_iter = re.finditer(PAT, text)
        for word in word_iter:
            word_bytes = bytes(word.group().encode('utf-8'))
            word_counts[word_bytes] = word_counts.get(word_bytes, 0) + 1
    return word_counts

def get_new_word(input_tuple):
    word_tuple, pair, word = input_tuple
    new_word = []
    new_pairs = []
    old_pairs = []
    i = 0
    merged_bytes = pair[0]+pair[1]
    while i+2 <= len(word_tuple):
        if pair == word_tuple[i:i+2]:
            new_word.append(merged_bytes)
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    if i == len(word_tuple)-1:
        new_word.append(word_tuple[i])
    new_word = tuple(new_word)
    for i in range(len(new_word)):
        if new_word[i] == merged_bytes:
            if i>=1:
                new_pairs.append(new_word[i-1:i+1])
                old_pairs.append((new_word[i-1], pair[0]))
            if i+2 <= len(new_word):
                new_pairs.append(new_word[i:i+2])
                old_pairs.append((pair[1], new_word[i+1]))
                if new_word[i+1] == merged_bytes:
                    old_pairs.append((pair[1], pair[0]))

    # print(word, word_tuple, left_new_pairs, right_new_pairs)
    return word, new_word, new_pairs, old_pairs

def read_chunk(input_path):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 16, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            yield chunk

def pre_tokenize_chunk_with_boundaries(input_tuple):
    input_path, start, end = input_tuple
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    return pre_tokenize_chunk(chunk)

K = 16
def pre_tokenize(input_path, num_processes):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes*K, b"<|endoftext|>")
    pool = mp.Pool(num_processes)
    input_tuples = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    bar = tqdm(total=len(input_tuples))
    word_counters = Counter()
    
    # 使用imap逐个处理结果
    for result in pool.imap(pre_tokenize_chunk_with_boundaries, input_tuples):
        word_counters += result
        bar.update(1)
    
    bar.close()
    return word_counters

def train_bpe(input_path, vocab_size, special_tokens, num_processes=8):    
    word_counters = pre_tokenize(input_path, num_processes)

    vocab = dict()
    merges = []
    pair_counter = OrderedDict()
    pair_to_idx = dict()
    word_to_tuple = dict()
    idx = 0
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
        pair_counter[chr(i)] = 0
    for token in special_tokens:
        vocab[len(vocab)] = bytes(token.encode('utf-8'))
    for word, count in word_counters.items():
        word_tuple = tuple([bytes([i]) for i in word])
        word_to_tuple[word] = word_tuple
        for i in range(len(word)-1):
            pair = word_tuple[i:i+2]
            pair_counter[pair] = pair_counter.get(pair, 0) + count
            if pair not in pair_to_idx:
                pair_to_idx[pair] = set()
            pair_to_idx[pair].add(idx)
        idx += 1
    idx_to_word = list(word_counters.keys())
    
    # sorted_pair = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
    # print(len(sorted_pair))
    # print(sorted_pair[:10])
    # pair_counter = OrderedDict(sorted_pair)
    pool = mp.Pool(num_processes)
    for _ in trange(vocab_size - len(vocab)):
        # get most frequent pair
        # max_pair = max(pair_counter.items(), key=lambda x: x[1])
        # merged_pair = max_pair[0]
        sorted_pair = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
        # print(len(sorted_pair))
        if len(sorted_pair) > vocab_size:
            pair_counter = dict(sorted_pair[:vocab_size])
        # print(sorted_pair[:3])
        max_freq = sorted_pair[0][1]
        candidate_pairs = [sorted_pair[0][0]]
        i = 1
        while i < len(sorted_pair) and sorted_pair[i][1] == max_freq:
            candidate_pairs.append(sorted_pair[i][0])
            i += 1
        merge_pair = max(candidate_pairs)
        # print(candidate_pairs, repr(merge_pair))
        merges.append(merge_pair)
        word_idxs_need_update = pair_to_idx[merge_pair]
        # print(word_idxs_need_update)
        assert len(word_idxs_need_update) == len(set(word_idxs_need_update))
        input_tuples = []
        for word_idx in list(word_idxs_need_update):
            word = idx_to_word[word_idx]
            word_tuple = word_to_tuple[word]
        #     input_tuples.append((word_tuple, merge_pair, word))
        # output_tuples = pool.map(get_new_word, input_tuples)
        # for word, new_word, new_pairs, old_pairs in output_tuples:
            word, new_word, new_pairs, old_pairs = get_new_word((word_tuple, merge_pair, word))
            word_count = word_counters[word]
            # print(new_pairs)
            for pair in new_pairs:
                pair_counter[pair] = pair_counter.get(pair, 0) + word_count
                if pair not in pair_to_idx:
                    pair_to_idx[pair] = set()
                pair_to_idx[pair].add(word_idx)
            for pair in old_pairs:
                if pair in pair_counter:
                    pair_counter[pair] -= word_count
            for i in range(len(new_word)-1):
                if new_word[i:i+2] in old_pairs:
                    old_pairs.remove(new_word[i:i+2])
            for pair in old_pairs:
                if pair in pair_to_idx:
                    pair_to_idx[pair].discard(word_idx)
            word_to_tuple[word] = new_word

        vocab[len(vocab)] = merge_pair[0]+merge_pair[1]
        pair_counter.pop(merge_pair)
        pair_to_idx.pop(merge_pair)
        # print(repr(merged_pair), len(word_counters), len(vocab), len(merges), merged_pair_bytes.decode('utf-8'))
        # break
    return vocab, merges

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--input_path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    # Track training start time and memory
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    print(f"Starting BPE training...")
    print(f"Initial memory usage: {start_memory:.2f} MB")
    
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens, args.num_processes)
    if args.save_path is not None:
        import json
        # 确保保存目录存在
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Created/ensured directory: {args.save_path}")
        
        # 保存vocab
        import pickle
        with open(os.path.join(args.save_path, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
        with open(os.path.join(args.save_path, "merges.pkl"), "wb") as f:
            pickle.dump(merges, f)

    # Calculate training statistics
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    training_hours = (end_time - start_time) / 3600
    memory_used = end_memory - start_memory
    
    # Find the longest token in the vocabulary
    longest_token = max(vocab.values(), key=len)
    longest_token_length = len(longest_token)
    
    print(f"\n=== Training Statistics ===")
    print(f"Training time: {training_hours:.4f} hours ({training_hours*60:.2f} minutes)")
    print(f"Memory usage: {memory_used:.2f} MB (started at {start_memory:.2f} MB, ended at {end_memory:.2f} MB)")
    print(f"Longest token in vocabulary: {repr(longest_token)} (length: {longest_token_length} bytes)")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    
    