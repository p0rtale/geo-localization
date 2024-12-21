from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[float] = None,
        encoded_size: Optional[float] = None,
        b: Optional[Tensor] = None,
    ):
        """
        Args:
            sigma (Optional[float]): standard deviation
            input_size (Optional[float]): the number of input dimensions
            encoded_size (Optional[float]): the number of dimensions the `b` matrix maps to
            b (Optional[Tensor], optional): Optionally specify a :attr:`b` matrix already sampled
        Raises:
            ValueError:
                If :attr:`b` is provided and one of :attr:`sigma`, :attr:`input_size`,
                or :attr:`encoded_size` is provided. If :attr:`b` is not provided and one of
                :attr:`sigma`, :attr:`input_size`, or :attr:`encoded_size` is not provided.
        """
        super().__init__()

        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma" "input_size" and "encoded_size" are required.'
                )

            b = torch.randn((encoded_size, input_size)) * sigma
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')

        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        vp = 2 * np.pi * v @ self.b.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
