import torch.nn as nn
from rmsnorm_torch import RMSNorm
import torch.nn.functional as F

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1,keepdim=True).values.clamp_(min=1e-5)
    y = (x*scale).round().clamp_(-128,127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w*scale).round().clamp_(-1,1) / scale
    return u

class BitLinear(nn.Linear):

    def forward(self, x):
        w = self.weight
        x_norm = RMSNorm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y