import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from src.config import Config


class PhoneticEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.gelu2 = nn.GELU()
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.gelu1(self.conv1(x))
        x = self.gelu2(self.conv2(x))
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class LinguisticEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_scale = emb_dim ** 0.5
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) * self.emb_scale
        x = x + self._get_positional_encoding(x)
        x, _ = self.bilstm(x)
        x = self.proj(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    def _get_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, dim = x.shape
        pe = torch.zeros(seq_len, dim, device=x.device, dtype=x.dtype)
        position = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=x.dtype, device=x.device)
            * (-torch.log(torch.tensor(10000.0, device=x.device)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        attn_output: torch.Tensor,
        phonetic_q: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([attn_output, phonetic_q, residual], dim=-1)
        gate = self.gate(concat)
        fused = gate * attn_output + (1 - gate) * phonetic_q
        fused = fused + residual
        return fused


class PLModel(Wav2Vec2PreTrainedModel):
    def __init__(
        self,
        config,
        vocab_size: int = None,
        hidden_dim: int = 768,
        emb_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(config)
        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = getattr(config, 'vocab_size', 100)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.phonetic_encoder = PhoneticEncoder(hidden_dim, dropout)
        self.linguistic_encoder = LinguisticEncoder(self.vocab_size, emb_dim, hidden_dim, dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.gated_fusion = GatedFusion(hidden_dim, dropout)
        self.lm_head = nn.Linear(hidden_dim, self.vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean")

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def load_pretrained_wav2vec2(self, pretrained_model_name: str):
        wav2vec2_pretrained = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.wav2vec2.load_state_dict(wav2vec2_pretrained.state_dict())

    def forward(
        self,
        input_values: torch.Tensor,
        canonical_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        wav2vec2_output = self.wav2vec2(input_values, attention_mask=attention_mask)[0]

        phonetic_q = self.phonetic_encoder(wav2vec2_output)

        ling_features = self.linguistic_encoder(canonical_ids)

        attn_output, _ = self.cross_attn(phonetic_q, ling_features, ling_features)

        if wav2vec2_output.shape[1] != phonetic_q.shape[1]:
            residual = nn.functional.interpolate(
                wav2vec2_output.transpose(1, 2),
                size=phonetic_q.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            residual = wav2vec2_output

        fused = self.gated_fusion(attn_output, phonetic_q, residual)
        logits = self.lm_head(fused)

        loss = None
        if labels is not None:
            logits_ctc = logits.log_softmax(dim=-1).transpose(0, 1)
            B = logits.shape[0]
            input_lengths = torch.full(
                (B,), logits.shape[1], dtype=torch.long, device=logits.device
            )
            target_lengths = (labels != -100).sum(dim=1)
            loss = self.ctc_loss_fn(logits_ctc, labels, input_lengths, target_lengths)

        return PLModelOutput(loss=loss, logits=logits)


class PLModelOutput:
    def __init__(self, loss: torch.Tensor = None, logits: torch.Tensor = None):
        self.loss = loss
        self.logits = logits


def create_model(cfg: Config, vocab_size: int) -> PLModel:
    config = Wav2Vec2Model.config_class.from_pretrained(cfg.model_name)
    model = PLModel(
        config,
        vocab_size=vocab_size,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.linguistic_emb_dim,
        num_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
    )
    model.load_pretrained_wav2vec2(cfg.model_name)
    if cfg.freeze_feature_extractor:
        model.freeze_feature_extractor()
    return model


def count_parameters(model, requires_grad_only: bool = True) -> int:
    if requires_grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())