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
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    freeze_feature_extractor: bool = True

    max_canonical_len: int = 128
    hidden_dim: int = 768
    linguistic_emb_dim: int = 256
    num_attention_heads: int = 8
    dropout: float = 0.1

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