import time
from typing import Optional, Iterable, BinaryIO
import regex as re
import multiprocessing as mp
import pickle
import random
from functools import partial
import itertools
import os
import numpy as np

mp.set_start_method('spawn', force=True)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None):
        self.vocab = vocab
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        self.lookup_vocab = {v: k for k, v in vocab.items()}
        self.max_special_token_length = max(len(token) for token in special_tokens) if special_tokens is not None else 0
        self.max_token_lenght = max(len(token) for token in vocab.values())
        self.max_all_token_length = max(self.max_special_token_length, self.max_token_lenght)
        self.vocab_size = len(vocab)
        self.merges_size = len(merges)
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else []

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable(iter(text)))
    
    def encode_word(self, word: str) -> tuple[int, bytes]:
        bytes_buffer = bytes(word.encode('utf-8'))
        assert len(bytes_buffer) > 0
        if bytes_buffer in self.lookup_vocab:
            return [self.lookup_vocab[bytes_buffer]]
        bytes_buffer = [bytes([b]) for b in bytes_buffer]
        # print(bytes_buffer)
        inf_idx = 100000
        while True:
            possible_merges = {}
            for i in range(len(bytes_buffer)-1):
                merge_idx = self.merges_dict.get((bytes_buffer[i], bytes_buffer[i+1]), inf_idx)
                possible_merges[merge_idx] = possible_merges.get(merge_idx, []) + [i]
            # print(possible_merges)
            idx_in_merges = min(possible_merges.keys())
            if idx_in_merges == inf_idx:
                break
            last_idx = -1
            for idx_in_word in possible_merges[idx_in_merges][::-1]:
                if idx_in_word == last_idx-1:
                    continue
                bytes_buffer = bytes_buffer[:idx_in_word] + [bytes_buffer[idx_in_word]+bytes_buffer[idx_in_word+1]] + bytes_buffer[idx_in_word+2:]
                last_idx = idx_in_word
            # print(bytes_buffer)
            
            # flag = False
            # for merge_idx, merge in enumerate(self.merges):
            #     i = 0
            #     while i < len(bytes_buffer)-1:
            #         if bytes_buffer[i] == merge[0] and bytes_buffer[i+1] == merge[1]:
            #             bytes_buffer[i] = merge[0]+merge[1]
            #             if (bytes_buffer[i], bytes_buffer[i+1]) in self.merges[:merge_idx]:
            #                 del bytes_buffer[i+1]
            #             flag = True
            #         i = i+1
            #     if flag:
            #         break
            # if not flag:
            #     break
        # print(bytes_buffer)
        return [self.lookup_vocab[merged_bytes] for merged_bytes in bytes_buffer]
        # raise ValueError(f"No token found for {bytes_buffer}")

    def split_text_to_words(self, text: str) -> list[str]:
        word_iter = re.finditer(PAT, text)
        for word in word_iter:
            yield word.group()
    
    def process_special_token(self, texts):
        # print(texts)
        flag = False
        for special_token in self.special_tokens:
            if special_token in texts:
                sub_texts = texts.split(special_token)
                for text in sub_texts[:-1]:
                    yield from self.process_special_token(text)
                    yield special_token
                flag = True
                break
        if not flag:
            yield from self.split_text_to_words(texts)
        else:
            yield from self.process_special_token(sub_texts[-1])
    
    def encode_iterable(self, iterable: Iterable[str], map_func = map) -> Iterable[int]:
        text_buffer = ""
        chunk_size = 4096
        
        while True:
            # 读取一个chunk
            chunk_items = list(itertools.islice(iterable, chunk_size))
            if not chunk_items:  # 没有更多数据
                break
            text_buffer += ''.join(chunk_items)
            
            # 如果buffer足够大或者没有更多数据，处理它
            if len(text_buffer) >= 10000 or not chunk_items:  # 1MB阈值
                word_list = list(self.process_special_token(text_buffer))
                for tokens in map_func(self.encode_word, word_list[:-3]):
                    for token in tokens:
                        yield token
                text_buffer = ''.join(word_list[-3:])
        
        # 处理剩余的buffer
        if len(text_buffer) > 0:
            word_list = list(self.process_special_token(text_buffer))
            for tokens in map_func(self.encode_word, word_list):
                for token in tokens:
                    yield token

    def decode(self, tokens: list[int]) -> str:
        return b''.join([self.vocab[token] for token in tokens]).decode("utf-8", errors="replace")

    def process_docs(self, input_tuple: tuple[str, int, int]) -> tuple[list[int], int]:
        input_path, start, end = input_tuple
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)
        chunk = chunk.decode("utf-8", errors="replace")
        return self.encode(chunk), len(chunk)

def read_docs(filepath: str, num_docs: int) -> list[bytes]:
    chunk_size = 1024 * 1024 # 1B chunks for better performance
    split_special_token = b"<|endoftext|>"
    initial_position = random.randint(0, 1000000)
    mini_chunk = b""
    doc_count = 0
    with open(filepath, "rb") as file:
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk += file.read(chunk_size)  # Read larger chunks
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            while found_at != -1:
                yield mini_chunk[:found_at]
                doc_count += 1
                mini_chunk = mini_chunk[found_at+len(split_special_token):]
                found_at = mini_chunk.find(split_special_token)
                if doc_count >= num_docs:
                    break
            if doc_count >= num_docs:
                break

from tqdm import tqdm
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    max_bytes: int = np.inf,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file_size = min(file_size, max_bytes)
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_filepath", type=str, required=True)
    parser.add_argument("--merges_filepath", type=str, required=True)
    parser.add_argument("--input_filepath", type=str, required=True)
    parser.add_argument("--num_docs", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=28)
    parser.add_argument("--max_bytes", type=int, default=np.inf)
    parser.add_argument("--output_filepath", type=str, default=None)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_files(Tokenizer, args.vocab_filepath, args.merges_filepath, ["<|endoftext|>"])

    docs = []
    token_counts = []
    byte_lengths = []
    K = 128
    # byte_length = sum(len(doc) for doc in docs)
    # docs = [doc.decode("utf-8", errors="replace") for doc in docs]
    # docs = list(read_docs(args.input_filepath, args.num_docs))

    with open(args.input_filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_processes*K, b"<|endoftext|>", args.max_bytes)
    
    input_tuples = [(args.input_filepath, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    start_time = time.time()
    bar = tqdm(total=len(input_tuples))
    special_token_id = tokenizer.lookup_vocab[b"<|endoftext|>"]
    
    # Create pool with proper error handling
    pool = mp.Pool(args.num_processes)
    try:
        map_func = partial(pool.imap, chunksize=4)
        with open(args.output_filepath, "wb") as f:
            for tokens, byte_length in map_func(tokenizer.process_docs, input_tuples):
                np.array(tokens+[special_token_id], dtype=np.uint16).tofile(f)
                token_counts.append(len(tokens) + 1)
                byte_lengths.append(byte_length)
                del tokens
                bar.update(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Ensure pool is properly closed
        pool.close()
        pool.join()
    
    bar.close()
    time_cost = time.time() - start_time
    print(f"Time cost: {time_cost} seconds")
    print(f"Speed: {sum(byte_lengths)/time_cost/1024/1024} MB/second, {sum(token_counts)/time_cost/(10**6)} M tokens/second")
