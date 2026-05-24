import json
import os
from typing import List, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from config import Config


def extract_phonemes_from_df(df: pd.DataFrame) -> List[str]:
    phonemes = set()
    for col in ["canonical", "transcript"]:
        for seq in df[col]:
            for ph in seq.strip().split():
                phonemes.add(ph)
    return sorted(phonemes)


def build_vocab_dict(phonemes: List[str]) -> dict:
    vocab = {"[CTC_BLANK]": 0}
    for i, ph in enumerate(phonemes, start=1):
        vocab[ph] = i
    return vocab


def save_vocab(vocab: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_data(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    torch.manual_seed(cfg.seed)
    indices = torch.randperm(n).tolist()

    train_end = int(cfg.train_ratio * n)
    val_end = train_end + int(cfg.val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def create_processor(vocab_path: str, cfg: Config) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[CTC_BLANK]",
        word_delimiter_token=None,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=cfg.sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    return processor


class MDDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        vocab: dict,
        cfg: Config,
    ):
        self.df = df
        self.audio_dir = audio_dir
        self.vocab = vocab
        self.sample_rate = cfg.sample_rate
        self.max_audio_len = cfg.max_audio_len * cfg.sample_rate
        self.blank_id = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.audio_dir, os.path.basename(row["path"]))
        waveform, orig_sr = torchaudio.load(audio_path)

        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        if waveform.size(0) > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]

        input_values = waveform.numpy()
        phoneme_str = row["transcript"]
        canonical_str = row["canonical"]

        labels = [self.vocab.get(ph, self.blank_id) for ph in phoneme_str.split()]
        canonical_ids = [self.vocab.get(ph, self.blank_id) for ph in canonical_str.split()]

        return {
            "input_values": input_values,
            "canonical_ids": canonical_ids,
            "labels": labels,
        }


class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, batch):
        input_values = [item["input_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        canonical_ids = [item["canonical_ids"] for item in batch]

        batch_processed = self.processor(
            input_values,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        max_label_len = max(len(l) for l in labels)
        padded_labels = torch.full((len(labels), max_label_len), -100, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = torch.tensor(label, dtype=torch.long)

        max_can_len = max(len(item) for item in canonical_ids)
        padded_canonical = torch.full((len(canonical_ids), max_can_len), 0, dtype=torch.long)
        for i, cids in enumerate(canonical_ids):
            padded_canonical[i, :len(cids)] = torch.tensor(cids, dtype=torch.long)

        return {
            "input_values": batch_processed["input_values"],
            "attention_mask": batch_processed["attention_mask"],
            "canonical_ids": padded_canonical,
            "labels": padded_labels,
        }