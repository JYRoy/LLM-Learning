import numpy as np
import torch


def subsequent_mask(size):
    """mask subsequent tensor

    size: the size of last two dimensions of the tensor
    """
    attn_shape = (1, size, size)
    # upper triangular matrix
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    # lower triangular matrix
    # 1 means masked, 0 means unmasked
    # row means the current position
    # col means the related postions with current position
    # for example: the index 2(3 position) could see 2 tokens
    return torch.from_numpy(1 - subsequent_mask)  # subsequent_mask == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src  # (batch_size, seq_len)
        self.src_mask = (src != pad).unsqueeze(-2)  # (batch_size, 1, seq_len)
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # (batch_size, seq_len - 1)
            self.tgt_y = tgt[:, 1:]  # (batch_size, seq_len - 1)
            self.tgt_mask = self.make_std_mask(
                self.tgt, pad
            )  # (batch_size, seq_len - 1, seq_len - 1)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)  # (batch_size, 1, seq_len - 1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
