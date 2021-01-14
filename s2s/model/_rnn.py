import argparse

import torch


class RNNEncModel(torch.nn.Module):
    def __init__(
        self, enc_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=enc_tknzr_cfg.n_vocab,
            embedding_dim=model_cfg.enc_d_emb,
            padding_idx=0,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.enc_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.enc_d_emb,
                out_features=model_cfg.enc_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.enc_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.enc_d_hid,
                out_features=model_cfg.enc_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.enc_dropout,
            ),
        )
        self.hid = torch.nn.RNN(
            input_size=model_cfg.enc_d_hid,
            hidden_size=model_cfg.enc_d_hid,
            num_layers=model_cfg.enc_n_layer,
            batch_first=True,
            dropout=model_cfg.enc_dropout * min(1, model_cfg.enc_n_layer - 1),
            bidirectional=model_cfg.is_bidir,
        )

    def forward(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
    ) -> torch.Tensor:
        r"""Encoder forward.

        src.shape == (B, S)
        src.dtype == torch.int
        src_len.shape == (B)
        src_len.dtype == torch.int
        """
        # shape: (B, S, H * (is_bidir + 1))
        out, _ = self.hid(self.emb_to_hid(self.emb(src)))

        # shape: (B, H * (is_bidir + 1))
        return out[torch.arange(out.size(0)).to(out.device), src_len - 1]


class RNNDecModel(torch.nn.Module):
    def __init__(
        self, dec_tknzr_cfg: argparse.Namespace,
        model_cfg: argparse.Namespace
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(
            num_embeddings=dec_tknzr_cfg.n_vocab,
            embedding_dim=model_cfg.dec_d_emb,
            padding_idx=0,
        )
        self.emb_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_emb,
                out_features=model_cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid,
                out_features=model_cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
        )
        self.enc_to_hid = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.enc_d_hid * (model_cfg.is_bidir + 1),
                out_features=model_cfg.dec_d_hid * model_cfg.dec_n_layer,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid * model_cfg.dec_n_layer,
                out_features=model_cfg.dec_d_hid * model_cfg.dec_n_layer,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
        )
        self.hid = torch.nn.RNN(
            input_size=model_cfg.dec_d_hid,
            hidden_size=model_cfg.dec_d_hid,
            num_layers=model_cfg.dec_n_layer,
            batch_first=True,
            dropout=model_cfg.dec_dropout * min(1, model_cfg.dec_n_layer - 1),
        )
        self.hid_to_emb = torch.nn.Sequential(
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid,
                out_features=model_cfg.dec_d_hid,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                p=model_cfg.dec_dropout,
            ),
            torch.nn.Linear(
                in_features=model_cfg.dec_d_hid,
                out_features=model_cfg.dec_d_emb,
            ),
        )

    def forward(
            self,
            enc_hid: torch.Tensor,
            tgt: torch.Tensor,
    ) -> torch.Tensor:
        r"""Decoder forward.

        enc_hid.shape == (B, H)
        enc_hid.dtype == torch.float
        tgt.shape == (B, S)
        tgt.dtype == torch.int
        """
        # shape: (B, S, H)
        out, _ = self.hid(
            self.emb_to_hid(self.emb(tgt)),
            self.enc_to_hid(enc_hid).reshape(
                self.hid.num_layers,
                -1,
                self.hid.hidden_size
            ),
        )

        # shape: (B, S, V)
        return self.hid_to_emb(out) @ self.emb.weight.transpose(0, 1)


class RNNModel(torch.nn.Module):
    model_name = 'RNN'

    def __init__(
            self,
            dec_tknzr_cfg: argparse.Namespace,
            enc_tknzr_cfg: argparse.Namespace,
            model_cfg: argparse.Namespace,
    ):
        super().__init__()
        self.enc = RNNEncModel(
            enc_tknzr_cfg=enc_tknzr_cfg,
            model_cfg=model_cfg,
        )
        self.dec = RNNDecModel(
            dec_tknzr_cfg=dec_tknzr_cfg,
            model_cfg=model_cfg,
        )

    def forward(
            self,
            src: torch.Tensor,
            src_len: torch.Tensor,
            tgt: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        # shape: (B, S, V)
        return self.dec(self.enc(src=src, src_len=src_len), tgt=tgt)

    @classmethod
    def update_subparser(cls, subparser: argparse.ArgumentParser):
        subparser.add_argument(
            '--dec_d_emb',
            help='Decoder embedding dimension.',
            required=True,
            type=int,
        )
        subparser.add_argument(
            '--dec_d_hid',
            help='Decoder hidden dimension.',
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
            '--enc_d_hid',
            help='Encoder hidden dimension.',
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
        subparser.add_argument(
            '--is_bidir',
            help=f'Whether to use bidirectional {cls.model_name} encoder or not.',
            action='store_true',
        )
