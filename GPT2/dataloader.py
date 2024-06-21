import torch
import tiktoken


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes) -> None:
        self.B = B  # micro batch size
        self.T = T  # sequence length
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("../data/tiny_shakespeare/input.txt") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        # state
        self.current_position = (
            self.B * self.T * self.process_rank
        )  # different rank starts from different part of data

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
