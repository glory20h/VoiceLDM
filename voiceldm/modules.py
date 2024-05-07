# Codes adapted from https://github.com/heatz123/naturalspeech/blob/main/models/models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()
    ):
        super(ConvBlock, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
            ),
            nn.BatchNorm1d(out_channels),
            activation,
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, mask=None):
        x = x.contiguous().transpose(1, 2)
        x = F.dropout(self.conv_layer(x), self.dropout, self.training)
        x = self.layer_norm(x.contiguous().transpose(1, 2))
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            
        return x
    
    
class SwishBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels, bias=True),
        )
        
    def forward(self, S, E, V):
        out = torch.cat(
            [
                S.unsqueeze(-1),
                E.unsqueeze(-1),
                V.unsqueeze(1).expand(-1, E.size(1), -1, -1),
            ],
            dim=-1,
        )
        out = self.layer(out)
        
        return out


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class LearnableUpsampler(nn.Module):
    def __init__(
        self,
        d_predictor=192,
        out_channels=192,
        kernel_size=3,
        dropout=0.0,
        conv_output_size=8,
        dim_w=4,
        dim_c=2,
        max_seq_len=1000,
    ):
        super(LearnableUpsampler, self).__init__()
        self.max_seq_len = max_seq_len

        # Attention (W)
        self.conv_w = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_w = SwishBlock(conv_output_size + 2, dim_w, dim_w)
        self.linear_w = nn.Linear(dim_w * d_predictor, d_predictor, bias=True)
        self.softmax_w = nn.Softmax(dim=2)

        # Auxiliary Attention Context (C)
        self.conv_c = ConvBlock(
            d_predictor,
            conv_output_size,
            kernel_size,
            dropout=dropout,
            activation=nn.SiLU(),
        )
        self.swish_c = SwishBlock(conv_output_size + 2, dim_c, dim_c)

        # Upsampled Representation (O)
        self.linear_einsum = nn.Linear(dim_c * dim_w, d_predictor)  # A
        self.layer_norm = nn.LayerNorm(d_predictor)

        self.proj_o = nn.Linear(d_predictor, out_channels)

    def forward(self, duration, V, src_len, src_mask, max_src_len):
        batch_size = duration.shape[0]

        # Duration Interpretation
        mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(V.device)
        mel_len = torch.clamp(mel_len, max=self.max_seq_len)
        max_mel_len = mel_len.max().item()
        mel_mask = self.get_mask_from_lengths(mel_len, max_mel_len)

        # Prepare Attention Mask
        src_mask_ = src_mask.unsqueeze(1).expand(
            -1, mel_mask.shape[1], -1
        )  # [B, tgt_len, src_len]
        mel_mask_ = mel_mask.unsqueeze(-1).expand(
            -1, -1, src_mask.shape[1]
        )  # [B, tgt_len, src_len]
        attn_mask = torch.zeros(
            (src_mask.shape[0], mel_mask.shape[1], src_mask.shape[1])
        ).to(V.device)
        attn_mask = attn_mask.masked_fill(src_mask_, 1.0)
        attn_mask = attn_mask.masked_fill(mel_mask_, 1.0)
        attn_mask = attn_mask.bool()

        # Token Boundary Grid
        e_k = torch.cumsum(duration, dim=1)
        s_k = e_k - duration
        e_k = e_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        t_arange = (
            torch.arange(1, max_mel_len + 1, device=V.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, max_src_len)
        )

        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(
            attn_mask, 0
        )

        # Attention (W)
        W = self.swish_w(S, E, self.conv_w(V))  # [B, T, K, dim_w]
        W = W.masked_fill(src_mask_.unsqueeze(-1), -float("Inf"))
        W = self.softmax_w(W)  # [B, T, K]
        W = W.masked_fill(mel_mask_.unsqueeze(-1), 0.0)
        W = W.permute(0, 3, 1, 2)

        # Auxiliary Attention Context (C)
        C = self.swish_c(S, E, self.conv_c(V))  # [B, T, K, dim_c]

        # Upsampled Representation (O)
        upsampled_rep = self.linear_w(
            torch.einsum("bqtk,bkh->bqth", W, V).permute(0, 2, 1, 3).flatten(2)
        ) + self.linear_einsum(
            torch.einsum("bqtk,btkp->bqtp", W, C).permute(0, 2, 1, 3).flatten(2)
        )  # [B, T, M]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask.unsqueeze(-1), 0)
        upsampled_rep = self.proj_o(upsampled_rep)

        return upsampled_rep, mel_mask, mel_len, W

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = (
            torch.arange(0, max_len)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(lengths.device)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


class Durator(nn.Module):
    def __init__(
        self,
        in_channels=768,
        out_channels=768,
        kernel_size=3,
        dropout=0.5,
        conv_output_size=8,
        dim_w=4,
        dim_c=2,
        max_seq_len=1000,
    ):
        super().__init__()
        self.dp = DurationPredictor(
            in_channels=in_channels,
            filter_channels=in_channels,
            kernel_size=kernel_size,
            p_dropout=dropout,
        )
        
        self.learnable_upsampler = LearnableUpsampler(
            d_predictor=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            conv_output_size=conv_output_size,
            dim_w=dim_w,
            dim_c=dim_c,
            max_seq_len=max_seq_len,
        )
        
    def forward(self, x, x_mask):
        logw = self.dp(x.transpose(1,2), x_mask)
        logw = torch.clamp(logw, max=3.5)
        w = torch.clamp(torch.exp(logw), min=1.0) * x_mask
        duration = w.squeeze(1)
        
        upsampled_rep, p_mask, _, W = self.learnable_upsampler(
            duration,
            x,
            x_mask.sum(-1),
            ~(x_mask.squeeze(1).bool()),
            x_mask.shape[-1],
        )
        
        return upsampled_rep, p_mask


class TextEncoder(nn.Module):
    def __init__(self, speecht5):
        super().__init__()
        self.encoder = speecht5.speecht5.encoder
        
        # Replace positional encoding with longer max_len
        max_len = 1000
        dim = self.encoder.config.hidden_size # 768
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.encoder.prenet.encode_positions.register_buffer("pe", pe)
        
    def forward(self, tokens):
        x = self.encoder(
            input_values=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            return_dict=True,
        ).last_hidden_state
        return x


class UNetWrapper(nn.Module):
    def __init__(self, unet, text_encoder=None):
        super().__init__()
        self.unet = unet
        self.durator = Durator(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            dropout=0.5,
        )
        self.text_encoder = text_encoder
        
    def forward(self, sample, timestep, text_embed, text_embed_mask, cond, training=False):
        if training:
            if self.text_encoder is not None:
                text_embed = self.text_encoder(text_embed)

            text_embed, text_embed_mask = self.durator(text_embed, text_embed_mask.unsqueeze(1))

        sample = self.unet(
            sample, 
            timestep, 
            encoder_hidden_states=text_embed,
            class_labels=cond,
        ).sample
    
        return sample