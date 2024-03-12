import numpy as np
import matplotlib.pyplot as plt
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
    return torch.from_numpy(1 - subsequent_mask)


# size = 5
# sm = subsequent_mask(size)
# print("sm: ", sm)
# plt.figure(figsize = (5, 5))
# plt.imshow(subsequent_mask(20)[0])
# plt.savefig('.images/subsequent_mask.jpg',bbox_inches='tight')
"""
output: 
sm:  tensor([[[1, 0, 0, 0, 0],
              [1, 1, 0, 0, 0],
              [1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 1]]], dtype=torch.uint8)
"""
