import torch.nn as nn
from rmsnorm_torch import RMSNorm
import torch.nn.functional as F
import torch

def activation_norm_quant(x: torch.Tensor):
    """RMSNorm & Per-token quantization to 8 bits. It can be implemented as a fused kernel.
    Args:
        x: an activation tensor with shape [n,d]
    Returns:
        y: a quantized activation tensor with shape [n,d]
        scale: a scalar for dequantization with shape [1]
    """
    x = RMSNorm(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale

def activation_quant(x: torch.Tensor):
    """Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
        x: an activation tensor with shape [n,d]
    Returns:
        y: a quantized activation tensor with shape [n,d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w: torch.Tensor):
    """Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
        w: an weight tensor with shape [d,k]
    Returns:
        u: a quantized weight with shape [d,k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear158B(nn.Linear):
    """
    """

    def forward(self, x):
        """
        Args:
            x: an input tensor with shape [n,d]
        Returns:
            y: an output tensor with shape [n,d]
        """
        if self.training:
            w = self.weight
            x_norm = RMSNorm(x)
            # A trick for implementing Straight-Through-Estimator (STE) using detach()
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            y = F.linear(x_quant, w_quant)
            return y
        else:
            w = self.weight  # a 1.58-bit weight tensor with shape [d, k]
            w_scale = (
                self.weight_scale
            )  # a full precision weight scale tensor with shape [1]
            x_quant, x_scale = activation_norm_quant(x)
            #y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale
            y = F.linear(x_quant, w) / w_scale / x_scale
            return y
