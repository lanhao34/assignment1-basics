import regex as re
import multiprocessing as mp
from collections import Counter, OrderedDict
from functools import reduce, partial

import os
from typing import BinaryIO

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
def pre_tokenize_chunk(filename, start, end):
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        texts = re.split("<|endoftext|>", chunk)
        word_counts = Counter()
        for text in texts:
            word_iter = re.finditer(PAT, text)
            for word in word_iter:
                word_bytes = bytes(word.group().encode('utf-8'))
                word_counts[word_bytes] = word_counts.get(word_bytes, 0) + 1
        return word_counts

def get_new_word(word_to_tuple, pair, word):
    word_tuple = word_to_tuple[word]
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

def train_bpe(input_path, vocab_size, special_tokens, num_processes=8):    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    worker_pool = mp.Pool(num_processes)
    pre_tokenize_chunk_partial = partial(pre_tokenize_chunk, input_path)
    word_counters = worker_pool.starmap(pre_tokenize_chunk_partial, zip(boundaries[:-1], boundaries[1:]))
    word_counters = reduce(lambda x, y: x + y, word_counters)
    # print(merged)

    vocab = dict()
    merges = []
    pair_counter = OrderedDict()
    pair_to_word = dict()
    word_to_tuple = dict()  
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
            if pair not in pair_to_word:
                pair_to_word[pair] = set()
            pair_to_word[pair].add(word)
    sorted_pair = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_pair[:10])
    # pair_counter = OrderedDict(sorted_pair)
    while len(vocab) < vocab_size:
        # get most frequent pair
        # max_pair = max(pair_counter.items(), key=lambda x: x[1])
        # merged_pair = max_pair[0]
        sorted_pair = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
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
        words_need_update = pair_to_word[merge_pair]
        assert len(words_need_update) == len(set(words_need_update))
        for word in words_need_update:
            word, new_word, new_pairs, old_pairs = get_new_word(word_to_tuple, merge_pair, word)
            word_count = word_counters[word]
            # print(new_pairs)
            for pair in new_pairs:
                pair_counter[pair] = pair_counter.get(pair, 0) + word_count
                if pair not in pair_to_word:
                    pair_to_word[pair] = set()
                pair_to_word[pair].add(word)
            for pair in old_pairs:
                if pair in pair_counter:
                    pair_counter[pair] -= word_count
                if pair not in pair_to_word:
                    pair_to_word[pair] = set()
                pair_to_word[pair].add(word)
            word_to_tuple[word] = new_word
        vocab[len(vocab)] = merge_pair[0]+merge_pair[1]
        pair_counter.pop(merge_pair)
        # print(repr(merged_pair), len(word_counters), len(vocab), len(merges), merged_pair_bytes.decode('utf-8'))
        # break
    return vocab, merges

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--input_path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--special_tokens", type=list, default=["<|endoftext|>"])
    args = parser.parse_args()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)
    print(merges)
    print(vocab)
    
    