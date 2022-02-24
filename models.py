"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import QANetLayers
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAF_character(nn.Module):
    """BiDAF model with character-level embeddings for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_character, self).__init__()
        self.emb = layers.FullEmbedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
 
        # embeddings
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, 2*hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, 2*hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 4 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 4 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    """QAnet model with character-level embeddings for SQuAD.

    Based on the paper:
    "QANET: COMBINING LOCAL CONVOLUTION WITH 
    GLOBAL SELF-ATTENTION FOR READING COMPREHENSION"
    by Adams Wei Yu , David Dohan , Minh-Thang Luong
    (https://arxiv.org/pdf/1804.09541.pdf).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(QANet, self).__init__()
        self.emb = layers.FullEmbedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc_blocks = QANetLayers.Block(hidden_size = 2*hidden_size, 
                                            resid_pdrop = drop_prob, 
                                            num_convs= 4) 
        
        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.attn_resizer = QANetLayers.QA_Conv1d(8*hidden_size, hidden_size)
        self.mod_enc_blocks = nn.ModuleList([QANetLayers.Block(hidden_size = hidden_size,
                                                                resid_pdrop = drop_prob,
                                                                num_convs = 2) for _  in range(7)])

        self.out = QANetLayers.QANetOutput(hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
 
        # embeddings
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, 2*hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, 2*hidden_size)

        c_enc = self.enc_blocks(c_emb, c_mask)    # (batch_size, c_len, 4 * hidden_size)
        q_enc = self.enc_blocks(q_emb, q_mask)    # (batch_size, q_len, 4 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)


        # model encoder blocks
        att = self.attn_resizer(att)
        att = F.dropout(att, 0.2, self.training)
        for block in self.mod_enc_blocks:
            att = block(att, c_mask)        # (batch_size, c_len, 2 * hidden_size)
        mod1 = att

        att = F.dropout(att, 0.2, self.training)
        for block in self.mod_enc_blocks:
            att = block(att, c_mask)
        mod2 = att

        att = F.dropout(att, 0.2, self.training)
        for block in self.mod_enc_blocks:
            att = block(att, c_mask)

        mod3 = att
        # output the probabilities
        out = self.out(mod1, mod2, mod3, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

