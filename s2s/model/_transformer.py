import argparse
import torch
import math
import copy
import numpy as np

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = torch.nn.Parameter(torch.ones(self.size))
        self.bias = torch.nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        r"""MultiHeadAttention forward.

        q.shape == (B, S, d_model)
        k.shape == (B, S, d_model)
        v.shape == (B, S, d_model)
        mask.shape == (B, 1, S)
        """
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        # shape: (B, S, H, K)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # shape: (B, H, S, K)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # shape: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # shape: (B, 1, 1, S)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, v)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class TransformerEncLayer(torch.nn.Module):
    def __init__(self, d_model, heads, dropout, ff):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=d_model,
                out_features=ff,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=dropout
            ),
            torch.nn.Linear(
                in_features=ff,
                out_features=d_model,
            ),
            torch.nn.Dropout(
                p=dropout
            ),
        )
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.attn(x, x, x, mask)))
        return self.norm2(x + self.ff(x))

class TransformerEncModel(torch.nn.Module):
    def __init__(
        self,
        enc_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__()
        self.d_model = model_cfg.enc_d_emb
        self.n_layer = model_cfg.enc_n_layer
        self.emb = torch.nn.Embedding(
            num_embeddings=enc_tknzr_cfg.n_vocab,
            embedding_dim=model_cfg.enc_d_emb,
            padding_idx=0,
        )
        self.layers = torch.nn.ModuleList([
            copy.deepcopy(TransformerEncLayer(
                model_cfg.enc_d_emb,
                model_cfg.enc_attn_heads,
                model_cfg.enc_dropout,
                model_cfg.enc_ff
            ))
            for _ in range(model_cfg.enc_n_layer)
        ])
    
    def pos_enc(self, emb):
        r""" Perform positional encoding.

        emb.shape: (B, S, d_model)
        """
        emb = emb * math.sqrt(self.d_model)
        seq_len = emb.size(1)

        # shape: (S, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))

        # shape: (1, S, d_model)
        pe = pe.unsqueeze(0)
        if emb.is_cuda:
            pe = pe.cuda()
        return emb + pe

    def forward(self, src, mask):
        r""" Transformer encode model forward.
        
        src.shape: (B, S)
        mask.shape: (B, 1, S)
        """

        # shape: (B, S, d_model)
        x = self.pos_enc(self.emb(src))
        for i in range(self.n_layer):
            x = self.layers[i](x, mask)
        return x

class TransformerDecLayer(torch.nn.Module):
    def __init__(self, d_model, self_heads, cross_heads, dropout, ff):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=d_model,
                out_features=ff,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=dropout
            ),
            torch.nn.Linear(
                in_features=ff,
                out_features=d_model,
            ),
            torch.nn.Dropout(
                p=dropout
            ),
        )
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(self_heads, d_model, dropout)
        self.cross_attn = MultiHeadAttention(cross_heads, d_model, dropout)

    def forward(self, x, memory, x_mask, memory_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, memory_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, memory, memory, x_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x

class TransformerDecModel(torch.nn.Module):
    def __init__(
        self,
        dec_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__()
        self.d_model = model_cfg.dec_d_emb
        self.n_layer = model_cfg.dec_n_layer
        self.emb = torch.nn.Embedding(
            num_embeddings=dec_tknzr_cfg.n_vocab,
            embedding_dim=model_cfg.dec_d_emb,
            padding_idx=0,
        )
        self.layers = torch.nn.ModuleList([
            copy.deepcopy(TransformerDecLayer(
                model_cfg.dec_d_emb,
                model_cfg.dec_attn_heads,
                model_cfg.cross_attn_heads,
                model_cfg.dec_dropout,
                model_cfg.dec_ff
            ))
            for _ in range(model_cfg.dec_n_layer)
        ])

    def pos_enc(self, emb):
        emb = emb * math.sqrt(self.d_model)
        seq_len = emb.size(1)
        pe = torch.zeros(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))
        pe = pe.unsqueeze(0)
        if emb.is_cuda:
            pe = pe.cuda()
        return emb + pe

    def forward(self, tgt, memory, mask, memory_mask):
        x = self.pos_enc(self.emb(tgt))
        for i in range(self.n_layer):
            x = self.layers[i](x, memory, mask, memory_mask)
        return x

class TransformerModel(torch.nn.Module):
    model_name = 'Transformer'

    def __init__(
            self,
            dec_tknzr_cfg: argparse.Namespace,
            enc_tknzr_cfg: argparse.Namespace,
            model_cfg: argparse.Namespace,
    ):
        super().__init__()
        self.enc = TransformerEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = TransformerDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.out = torch.nn.Linear(
            in_features=model_cfg.dec_d_emb,
            out_features=dec_tknzr_cfg.n_vocab
        )

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        tgt: torch.Tensor,
        tgt_len: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        src.shape == (B, S)
        src_len.shape == (B)
        tgt.shape == (B, S)
        tgt_len.shape == (B)
        """
        # shape: (B, S)
        src_mask = np.zeros((src.size(0), src.size(1))).astype('uint8')
        tgt_mask = np.zeros((tgt.size(0), tgt.size(1))).astype('uint8')
        for i in range(src.size(0)):
            src_mask[i, :src_len[i]] = True

        for i in range(tgt.size(0)):
            tgt_mask[i, :tgt_len[i]] = True

        # shape: (B, 1, S)
        src_mask = torch.tensor(src_mask).unsqueeze(-2)
        tgt_mask = torch.tensor(tgt_mask).unsqueeze(-2)

        nopeak_mask = np.triu(np.ones((1, tgt.size(1), tgt.size(1))), k=1).astype('uint8')
        nopeak_mask = torch.from_numpy(nopeak_mask == 0)
        tgt_mask = tgt_mask & nopeak_mask

        if src.is_cuda:
            src_mask = src_mask.cuda()
            tgt_mask = tgt_mask.cuda()

        memory = self.enc(src, src_mask)
        return self.out(self.dec(tgt, memory, src_mask, tgt_mask))

    @classmethod
    def update_subparser(cls, subparser: argparse.ArgumentParser):
        subparser.add_argument(
            '--dec_d_emb',
            help='Decoder embedding dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_ff',
            help='Decoder feedforward layer dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_attn_heads',
            help='Decoder self attention heads.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--cross_attn_heads',
            help='Cross attention heads.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_dropout',
            help='Decoder dropout rate.',
            required=True,
            type=float,
        )
        subparser.add_argument(
            '--dec_n_layer',
            help=f'Number of decoder {cls.model_name} layer(s).',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_d_emb',
            help='Encoder embedding dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_ff',
            help='Encoder feedforward layer dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_attn_heads',
            help='Encode self attention heads.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--enc_dropout',
            help='Encoder dropout rate.',
            required=True,
            type=float,
        )
        subparser.add_argument(
            '--enc_n_layer',
            help=f'Number of encoder {cls.model_name} layer(s).',
            required=True,
            type=int,
        )
