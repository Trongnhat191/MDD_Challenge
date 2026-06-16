import os
import sys

import pandas as pd
import torch

# Ensure project root is in sys.path so 'from src.xxx' imports work
# regardless of whether running from project root or from src/ directory
_src_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_src_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.config import Config
from src.data import extract_phonemes_from_df, build_vocab_dict, save_vocab
from src.train import train
from src.eval import evaluate


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = pd.read_csv(cfg.metadata_path)
    phonemes = extract_phonemes_from_df(df)
    vocab = build_vocab_dict(phonemes)
    save_vocab(vocab, cfg.vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens ({len(phonemes)} phonemes + CTC blank)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train(cfg, device)

    evaluate(cfg, device)


if __name__ == "__main__":
    main()