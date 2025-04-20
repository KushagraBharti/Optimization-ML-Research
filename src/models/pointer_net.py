# src/models/pointer_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # embeddings
        self.embed_ep = nn.Linear(1, hidden_dim)
        self.embed_L  = nn.Linear(1, hidden_dim)

        # encoder
        self.encoder_rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.enc_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # decoder
        self.decoder_cell  = nn.LSTMCell(hidden_dim, hidden_dim)
        self.decoder_start = nn.Parameter(torch.zeros(hidden_dim))

        # attention
        self.W_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v     = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, endpoints, mask, L, target_seq=None):
        """
        endpoints: (B, E)
        mask:      (B, E) boolean for valid endpoints
        L:         (B, 1)
        target_seq:(B, T) long, values in [0..E] where E=STOP
        """
        B, E = endpoints.size()
        T = target_seq.size(1) if target_seq is not None else None

        # embed inputs
        ep_in  = endpoints.unsqueeze(-1)    # (B, E, 1)
        ep_emb = self.embed_ep(ep_in)       # (B, E, H)
        L_emb  = self.embed_L(L).unsqueeze(1)  # (B, 1, H)
        enc_in = ep_emb + L_emb               # (B, E, H)

        # encoder
        enc_out, _ = self.encoder_rnn(enc_in)     # (B, E, 2H)
        enc = self.enc_proj(enc_out)              # (B, E, H)

        # decoder init
        hx = torch.zeros(B, self.hidden_dim, device=enc.device)
        cx = torch.zeros(B, self.hidden_dim, device=enc.device)
        dec_input = self.decoder_start.unsqueeze(0).expand(B, -1)  # (B, H)

        # precompute keys
        enc_key = self.W_enc(enc)  # (B, E, H)

        outputs = []
        batch_idx = torch.arange(B, device=enc.device)

        for t in range(T):
            # one LSTMCell step
            hx, cx = self.decoder_cell(dec_input, (hx, cx))  # (B, H)

            # attention
            dec_key = self.W_dec(hx).unsqueeze(1)              # (B, 1, H)
            scores  = self.v(torch.tanh(enc_key + dec_key)).squeeze(-1)  # (B, E)
            scores  = scores.masked_fill(~mask, float("-inf"))

            # STOP token gets zero score
            stop_scores = torch.zeros(B, 1, device=enc.device) # (B,1)
            logp = F.log_softmax(torch.cat([scores, stop_scores], dim=1), dim=1)  # (B, E+1)
            outputs.append(logp)

            # pick next index
            if target_seq is not None:
                idx = target_seq[:, t]           # (B,)
            else:
                idx = logp.argmax(dim=1)         # (B,)

            # build next decoder input safely
            # for idx < E, grab enc[batch_idx, idx]; else ZERO
            next_input = torch.zeros_like(hx)   # (B, H)
            valid = idx < E                     # (B,) bool
            if valid.any():
                ni = idx[valid]
                bi = batch_idx[valid]
                next_input[valid] = enc[bi, ni]  # gather valid
            dec_input = next_input

        return torch.stack(outputs, dim=1)  # (B, T, E+1)
