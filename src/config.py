from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"

    sample_rate: int = 16000
    max_audio_len: int = 20

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    batch_size: int = 8
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    freeze_feature_extractor: bool = True
    freeze_wav2vec2_encoder: bool = True  # freeze transformer encoder (most params)
    unfreeze_epoch: int = 0  # unfreeze wav2vec2 after this epoch (0=never)

    max_canonical_len: int = 128
    hidden_dim: int = 768
    linguistic_emb_dim: int = 256
    num_attention_heads: int = 8
    num_cross_attn_layers: int = 2  # stacked cross-attention + fusion blocks
    dropout: float = 0.1

    # SpecAugment (applied via Wav2Vec2 config)
    spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.05
    mask_feature_length: int = 10

    # Speed perturbation
    speed_perturb: bool = True
    speed_perturb_range: tuple = (0.9, 1.1)

    # Oversampling error samples
    oversample_errors: bool = True
    error_oversample_weight: float = 3.0

    # Weighted CTC
    weighted_ctc: bool = True
    error_loss_weight: float = 2.0

    # Beam search decoding
    beam_width: int = 10
    beam_search_lm_weight: float = 0.0  # 0.0 = no LM, >0 for shallow fusion

    data_root: str = "MDD-Challenge-2025-training-set"
    metadata_path: str = field(init=False)
    audio_dir: str = field(init=False)
    output_dir: str = "outputs"
    checkpoint_dir: str = field(init=False)
    vocab_path: str = field(init=False)
    processor_dir: str = field(init=False)

    def __post_init__(self):
        self.metadata_path = f"{self.data_root}/metadata/train_phones.csv"
        self.audio_dir = f"{self.data_root}/audio_data/train"
        self.checkpoint_dir = f"{self.output_dir}/checkpoints"
        self.vocab_path = f"{self.output_dir}/vocab.json"
        self.processor_dir = f"{self.output_dir}/processor"