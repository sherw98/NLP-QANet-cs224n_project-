"""Assortment of QA layers for use in models.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

def position_encoding(x):
    seq_len = x.size()[2]
    n_embd = x.size()[1]

    pos = nn.Parameter()
    return x+ pos.to(x.get_device())

def get_sin_cos(seq_len, n_embd):
    position = torch.arange(seq_len)
    


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class QA_Conv1d(nn.Module):
    """ Conv layer for QA Net"""
    def __init__(self, in_channels, out_channels, kernel_size = 1,
                 stride = 1, padding = 0, groups = 1, relu = False, bias = False)
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding = padding, groups = groups, bias=bias)
        if relu:
            self.relu = nn.ReLU()
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity = "relu")
        else:
            self.relu = None
            nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self, x):
        if self.relu:
            return F.relu(self.conv(x))
        else:
            return self.conv(x)

class Block(nn.Module):
    """ an QANet Transformer block with Conv nets"""

    def __init__(self, hidden_size, num_conv, resid_pdrop):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Conv1d(in_channels = hidden_size, 
                      out_channels = hidden_size,
                      kernel_size = 7, 
                      groups = hidden_size, 
                      padding = 7//2,
                      bias = False),

            nn.Conv1d(in_channels = hidden_size,
                      out_channels = 128,
                      kernel_size = 1,
                      padding = 0,
                      bias = True),
            nn.ReLU(),
            nn.Dropout(resid_pdrop)
        )

        self.attn_ln = nn.LayerNorm(128)        
        self.attn = CausalSelfAttention(n_embd = 128, 
                                        n_head = 8, 
                                        attn_pdrop = 0.2,
                                        resid_pdrop = resid_pdrop,
                                        block_size =  128)
        
        self.ff_ln = nn.LayerNorm(128)
        self.ff_1 = QA_Conv1d(128, 128, relu= True, bias = True)
        self.ff_2 = QA_Conv1d(128, 128, bias = True)


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class QANetEncoder(nn.Module):
    """Transformer-based encoding layer specific to QANet.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_conv_layers (int): Number of convolution layers for transformer blocks to use
        num_transformer_blocks (int): Number of transformer blocks to use
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_conv_layers,
                 num_transformer_blocks,
                 drop_prob=0.):
        super(QANetEncoder, self).__init__()
        self.drop_prob = drop_prob
        

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x