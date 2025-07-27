import regex as re
import multiprocessing as mp
from collections import Counter
from functools import reduce, partial

from pretokenization_example import find_chunk_boundaries

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
                word_counts[word.group()] = word_counts.get(word.group(), 0) + 1
        return word_counts

class Word:
    def __init__(self, word, count):
        self.word = word
        self.count = count
        self.parrent = []
    
    def __repr__(self):
        return f"Word(word={self.word}, count={self.count})"
    
    def add_parrent(self, parrent):
        self.parrent.append(parrent)
    
    def get_parrent(self):
        return self.parrent

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    args = parser.parse_args()
    
    with open(args.data_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_processes, b"<|endoftext|>")
    worker_pool = mp.Pool(args.num_processes)
    pre_tokenize_chunk_partial = partial(pre_tokenize_chunk, args.data_path)
    counters = worker_pool.starmap(pre_tokenize_chunk_partial, zip(boundaries[:-1], boundaries[1:]))
    merged = reduce(lambda x, y: x + y, counters)
    print(merged)

    vocab = Counter()
    for i in range(256):
        vocab[bytes([i])] = 0
    pair_counter = Counter()
    sub_vocab_dict = dict()
    for word, count in merged.items():
        word = bytes(word.encode("utf-8"))
        for i in range(len(word)-1):
            pair = word[i:i+2]
            pair_counter[pair] = pair_counter.get(pair, 0) + count
    
    # get most frequent pair
    sorted_pair = sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
    merged_pair = sorted_pair[0][0]

    # merge pair
    
    for byte in merged_pair:
        pair_counter[byte] = pair_counter.get(byte, 0) - sorted_vocab[merged_pair][byte]
    del vocab[merged_pair]

    # get most frequent pair
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    merged_pair = sorted_vocab[0]
    
    