from torch.nn import TransformerEncoder
import os
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import lightning as L
import finalnlp
from finalnlp.replacer import replace_linears_in_pytorch_model
from finalnlp import bitnet1
from finalnlp import bitnet158
from lightning.pytorch import loggers as pl_loggers

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SentenceClassifier(L.LightningModule):
    def __init__(self, n_vocab, num_classes, d_model, nhead, num_layers, d_ffl, linear_replacer=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.transformer = nn.Sequential(
            nn.Embedding(n_vocab, d_model),
            PositionalEncoding(d_model),
            TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, activation='gelu'),
                num_layers=num_layers
            ),
            nn.Linear(d_model, d_ffl),
            nn.GELU(),
            nn.Linear(d_ffl, num_classes),
        )
        if linear_replacer:
            replace_linears_in_pytorch_model(self, linear_replacer)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()


    def forward(self, x):
        B, L = x.shape
        out = self.transformer(x)
        assert out.shape == (B, self.num_classes)
        return out 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', (y_hat.argmax(-1) == y).float().mean())
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', (y_hat.argmax(-1) == y).float().mean())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

