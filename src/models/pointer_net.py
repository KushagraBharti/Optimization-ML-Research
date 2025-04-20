# src/models/pointer_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # embeddings
        self.embed_ep = nn.Linear(1, hidden_dim)   # embed scalar endpoint
        self.embed_L = nn.Linear(1, hidden_dim)    # embed battery scalar

        # encoder
        self.encoder_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # project bi‐directional hidden to single hidden_dim
        self.enc_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # decoder
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.decoder_start = nn.Parameter(torch.zeros(hidden_dim))  # learnable start token

        # attention
        self.W_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, endpoints, mask, L, target_seq=None):
        """
        endpoints: (B, E) float
        mask:      (B, E) bool
        L:         (B, 1) float
        target_seq: (B, T) long (optional, teacher‐forcing)
        """
        B, E = endpoints.size()
        T = target_seq.size(1) if target_seq is not None else None

        # embed endpoints and L
        ep_in = endpoints.unsqueeze(-1)          # (B, E, 1)
        ep_emb = self.embed_ep(ep_in)            # (B, E, H)
        L_emb = self.embed_L(L).unsqueeze(1)     # (B, 1, H)
        enc_in = ep_emb + L_emb                  # broadcast (B, E, H)

        # encoder RNN
        enc_out, (h_n, c_n) = self.encoder_rnn(enc_in)  # enc_out (B, E, 2H)
        enc = self.enc_proj(enc_out)                    # (B, E, H)

        # decoder init
        dec_input = self.decoder_start.unsqueeze(0).expand(B, -1)  # (B, H)
        hx = torch.zeros(B, self.hidden_dim, device=endpoints.device)
        cx = torch.zeros(B, self.hidden_dim, device=endpoints.device)

        # precompute encoder keys
        enc_key = self.W_enc(enc)  # (B, E, H)

        outputs = []
        for t in range(T):
            hx, cx = self.decoder_cell(dec_input, (hx, cx))  # (B, H)
            # attention
            dec_key = self.W_dec(hx).unsqueeze(1)            # (B, 1, H)
            u = self.v(torch.tanh(enc_key + dec_key)).squeeze(-1)  # (B, E, )
            # mask padding endpoints
            u = u.masked_fill(~mask, float("-inf"))
            # allow STOP token by appending one score
            stop_score = torch.zeros(B, 1, device=u.device)
            scores = torch.cat([u, stop_score], dim=1)      # (B, E+1)
            probs = F.log_softmax(scores, dim=1)            # log‐probs for NLLLoss

            outputs.append(probs)

            # next input: teacher forcing or argmax
            if target_seq is not None:
                idx = target_seq[:, t]                     # (B,)
            else:
                idx = probs.argmax(dim=1)
            # embedding of chosen endpoint for next step
            # for STOP, feed zero vector
            dec_input = torch.where(
                idx.unsqueeze(1) < E,
                enc[torch.arange(B), idx],                # (B, H)
                torch.zeros_like(hx),
            )

        # stack outputs: list of (B, E+1) → (B, T, E+1)
        out = torch.stack(outputs, dim=1)
        return out
