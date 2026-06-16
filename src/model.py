import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from src.config import Config


class CrossAttentionBlock(nn.Module):
    """One block: cross-attention -> gated fusion with residual + layer norm."""

    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.gated_fusion = GatedFusion(hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        phonetic_q: torch.Tensor,
        ling_kv: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        attn_output, _ = self.cross_attn(phonetic_q, ling_kv, ling_kv)
        fused = self.gated_fusion(attn_output, phonetic_q, residual)
        fused = self.norm(fused)
        return fused


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
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
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
        num_cross_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(config)
        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = getattr(config, 'vocab_size', 100)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.phonetic_encoder = PhoneticEncoder(hidden_dim, dropout)
        self.linguistic_encoder = LinguisticEncoder(
            self.vocab_size, emb_dim, hidden_dim, dropout
        )

        # Stacked cross-attention + gated fusion blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # Self-attention head (transformer encoder) after fusion
        self.self_attn_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )

        self.lm_head = nn.Linear(hidden_dim, self.vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="none")

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_encoder(self):
        """Freeze the wav2vec2 transformer encoder (the bulk ~95M params)."""
        for param in self.wav2vec2.encoder.parameters():
            param.requires_grad = False

    def unfreeze_wav2vec2(self):
        """Unfreeze ALL wav2vec2 weights for full fine-tuning."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True

    def load_pretrained_wav2vec2(self, pretrained_model_name: str):
        wav2vec2_pretrained = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.wav2vec2.load_state_dict(wav2vec2_pretrained.state_dict())

    def _get_sample_weights(
        self,
        canonical_ids: torch.Tensor,
        labels: torch.Tensor,
        error_weight: float = 2.0,
    ) -> torch.Tensor:
        """Compute per-sample weight: error_weight if canonical != labels, else 1.0.
        
        Note: canonical_ids and labels may have different lengths (padded independently
        in collator), so we compare only the overlapping portion.
        """
        B = canonical_ids.shape[0]
        weights = torch.ones(B, device=canonical_ids.device, dtype=torch.float)
        for i in range(B):
            can = canonical_ids[i]
            lab = labels[i]
            min_len = min(can.shape[0], lab.shape[0])
            can = can[:min_len]
            lab = lab[:min_len]
            mask = (lab != -100) & (can != 0)
            if mask.sum() > 0:
                if not torch.equal(can[mask], lab[mask]):
                    weights[i] = error_weight
        return weights

    def forward(
        self,
        input_values: torch.Tensor,
        canonical_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        sample_weights: torch.Tensor = None,
    ):
        wav2vec2_output = self.wav2vec2(input_values, attention_mask=attention_mask)[0]

        phonetic_q = self.phonetic_encoder(wav2vec2_output)

        ling_features = self.linguistic_encoder(canonical_ids)

        # Residual from raw wav2vec2 (downsampled to match phonetic_q length)
        if wav2vec2_output.shape[1] != phonetic_q.shape[1]:
            residual = nn.functional.interpolate(
                wav2vec2_output.transpose(1, 2),
                size=phonetic_q.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            residual = wav2vec2_output

        # Stacked cross-attention + fusion blocks
        x = phonetic_q
        for block in self.cross_attn_blocks:
            x = block(x, ling_features, residual)

        # Self-attention head
        x = self.self_attn_head(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            logits_ctc = logits.log_softmax(dim=-1).transpose(0, 1)
            B = logits.shape[0]
            input_lengths = torch.full(
                (B,), logits.shape[1], dtype=torch.long, device=logits.device
            )
            target_lengths = (labels != -100).sum(dim=1)
            per_sample_loss = self.ctc_loss_fn(
                logits_ctc, labels, input_lengths, target_lengths
            )

            # Weighted CTC
            if sample_weights is None:
                sample_weights = self._get_sample_weights(
                    canonical_ids, labels, error_weight=2.0
                )
            loss = (per_sample_loss * sample_weights).mean()

        return PLModelOutput(loss=loss, logits=logits)


class PLModelOutput:
    def __init__(self, loss: torch.Tensor = None, logits: torch.Tensor = None):
        self.loss = loss
        self.logits = logits


def create_model(cfg: Config, vocab_size: int) -> PLModel:
    w2v_config = Wav2Vec2Model.config_class.from_pretrained(cfg.model_name)

    # Enable SpecAugment via HuggingFace Wav2Vec2 config
    if cfg.spec_augment:
        w2v_config.apply_spec_augment = True
        w2v_config.mask_time_prob = cfg.mask_time_prob
        w2v_config.mask_time_length = cfg.mask_time_length
        w2v_config.mask_feature_prob = cfg.mask_feature_prob
        w2v_config.mask_feature_length = cfg.mask_feature_length

    model = PLModel(
        w2v_config,
        vocab_size=vocab_size,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.linguistic_emb_dim,
        num_heads=cfg.num_attention_heads,
        num_cross_attn_layers=cfg.num_cross_attn_layers,
        dropout=cfg.dropout,
    )
    model.load_pretrained_wav2vec2(cfg.model_name)
    if cfg.freeze_feature_extractor:
        model.freeze_feature_extractor()
    if cfg.freeze_wav2vec2_encoder:
        model.freeze_encoder()
    return model


def count_parameters(model, requires_grad_only: bool = True) -> int:
    if requires_grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def beam_search_decode(
    logits: torch.Tensor,
    decoder=None,
    beam_width: int = 10,
    blank_id: int = 0,
):
    """Decode CTC logits using beam search (phoneme-aware) or greedy fallback.

    Args:
        logits: (batch, time, vocab_size) tensor.
        decoder: ignored, kept for API compat. Uses built-in beam search.
        beam_width: beam width (1 = greedy).
        blank_id: CTC blank token ID.

    Returns:
        List of List[int]: decoded token ID sequences (after collapsing).
    """
    if beam_width <= 1:
        return _greedy_ctc(logits, blank_id)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_np = log_probs.detach().cpu().numpy()
    results = []
    for b in range(log_probs_np.shape[0]):
        seq = _ctc_beam_search(log_probs_np[b], beam_width, blank_id)
        results.append(seq)
    return results


def _greedy_ctc(logits: torch.Tensor, blank_id: int = 0) -> list:
    """Greedy argmax + collapse repeats + remove blank."""
    pred_ids = torch.argmax(logits, dim=-1)
    results = []
    for b in range(pred_ids.shape[0]):
        ids = pred_ids[b].tolist()
        collapsed = []
        prev = None
        for tid in ids:
            if tid != prev and tid != blank_id:
                collapsed.append(tid)
            prev = tid
        results.append(collapsed)
    return results


def _ctc_beam_search(
    log_probs: np.ndarray,
    beam_width: int = 10,
    blank_id: int = 0,
) -> list:
    """Simple CTC beam search for phoneme-level vocab.

    log_probs: (time, vocab_size) numpy array of log probabilities.

    Returns list of token IDs (after CTC collapsing).
    """
    T, V = log_probs.shape

    # Each beam: (tuple_of_ids, score)
    # Score is in log space (higher = better)
    beams = {tuple(): 0.0}  # prefix -> score

    for t in range(T):
        new_beams = {}
        for prefix, score in beams.items():
            for v in range(V):
                p = log_probs[t, v]
                new_score = score + p

                if v == blank_id:
                    new_prefix = prefix
                elif prefix and prefix[-1] == v:
                    # Same token repeated: CTC merges identical consecutive tokens
                    # We keep both versions: with and without merging
                    # Version 1: merged (blank forced between same tokens)
                    merged_prefix = prefix
                    merged_score = new_score
                    if merged_prefix not in new_beams or new_beams[merged_prefix] < merged_score:
                        new_beams[merged_prefix] = merged_score
                    # Version 2: not merged (actual repetition of same phoneme)
                    unmerged_prefix = prefix + (v,)
                    unmerged_score = new_score
                    if unmerged_prefix not in new_beams or new_beams[unmerged_prefix] < unmerged_score:
                        new_beams[unmerged_prefix] = unmerged_score
                else:
                    new_prefix = prefix + (v,)
                    if new_prefix not in new_beams or new_beams[new_prefix] < new_score:
                        new_beams[new_prefix] = new_score

        # Prune to top beam_width
        beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])

    # Return best sequence
    if beams:
        best_prefix = max(beams, key=beams.get)
        return list(best_prefix)
    return []


def create_ctc_decoder(vocab: dict, blank_token: str = "[CTC_BLANK]"):
    """Placeholder kept for API compatibility. Beam search is built-in now."""
    return None


def decode_to_text(
    decoded: list,
    id2token: dict,
    blank_token: str = "[CTC_BLANK]",
) -> list:
    """Convert token ID sequences to space-separated phoneme text."""
    results = []
    for item in decoded:
        if isinstance(item, list):
            tokens = [id2token.get(tid, "") for tid in item]
            tokens = [t for t in tokens if t and t != blank_token]
            results.append(" ".join(tokens))
        else:
            results.append(str(item))
    return results