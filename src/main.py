import os

import pandas as pd
import torch

from config import Config
from data import extract_phonemes_from_df, build_vocab_dict, save_vocab
from train import train
from eval import evaluate


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