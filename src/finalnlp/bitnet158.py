import torch.nn as nn
from finalnlp.rmsnorm_torch import RMSNorm
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
    rms_norm = RMSNorm(x.shape[x.dim() - 1]).to(x.device)
    x = rms_norm(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y.to(x.device), scale

def activation_quant(x: torch.Tensor):
    """Per-token quantization to 8 bits. No grouping is needed for quantization.
    Args:
        x: an activation tensor with shape [n,d]
    Returns:
        y: a quantized activation tensor with shape [n,d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y.to(x.device)

def weight_quant(w: torch.Tensor):
    """Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
        w: an weight tensor with shape [d,k]
    Returns:
        u: a quantized weight with shape [d,k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    return u, scale

class BitLinear158B(nn.Linear):
    """
    """

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: an input tensor with shape [n,d]
        Returns:
            y: an output tensor with shape [n,d]
        """
        w = self.weight.to(x.device)
        rms_norm = RMSNorm(x.shape[x.dim() - 1]).to(x.device)
        x_norm = rms_norm(x)
        # A trick for implementing Straight-Through-Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        if self.training:
            self.wq_cache = weight_quant(w)
        wq, w_scale = self.wq_cache
        w_quant = w + (wq/w_scale - w).detach()
        y = F.linear(x_quant, w_quant)
        return y