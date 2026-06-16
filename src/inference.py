import os
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

# Ensure project root is in sys.path (Colab compatibility)
_src_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_src_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.config import Config
from src.data import MDDDataset, DataCollatorCTCWithPadding, load_vocab
from src.model import create_model, create_ctc_decoder, decode_to_text, beam_search_decode


@torch.no_grad()
def predict(model, dataloader, device, decoder=None, beam_width=10):
    """Run inference with optional beam search decoding."""
    model.eval()
    all_decoded = []

    for batch in dataloader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        canonical_ids = batch["canonical_ids"].to(device)

        output = model(input_values, canonical_ids, attention_mask=attention_mask)
        logits = output.logits

        decoded = beam_search_decode(logits, decoder, beam_width=beam_width)
        all_decoded.extend(decoded)

    return all_decoded


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for public test set")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to public test CSV (e.g., MDD-Challenge-2025-public-test/metadata/public_test_phones.csv)")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Path to audio directory (e.g., MDD-Challenge-2025-public-test/audio_data)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default="outputs/vocab.json",
                        help="Path to vocabulary file")
    parser.add_argument("--processor", type=str, default="outputs/processor",
                        help="Path to processor directory")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=10,
                        help="Beam search width (1=greedy)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(df)}")

    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab)
    id2token = {v: k for k, v in vocab.items()}
    print(f"Vocab size: {vocab_size}")

    cfg = Config()
    model = create_model(cfg, vocab_size)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.to(device)
    print("Model loaded successfully")

    # Create CTC decoder for beam search
    decoder = create_ctc_decoder(vocab)
    beam_type = f"beam search (width={args.beam_width})" if args.beam_width > 1 else "greedy"
    print(f"Decoding: {beam_type}")

    processor = Wav2Vec2Processor.from_pretrained(args.processor)

    test_dataset = MDDDataset(df, args.audio_dir, vocab, cfg, is_train=False)
    collator = DataCollatorCTCWithPadding(processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    predictions = predict(model, test_loader, device, decoder, args.beam_width)
    results = decode_to_text(predictions, id2token)

    results_df = pd.DataFrame({
        "id": df["id"],
        "path": df["path"],
        "predict": results
    })
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Sample predictions:")
    print(results_df.head(5))


if __name__ == "__main__":
    main()