import os
import subprocess

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from src.config import Config
from src.data import split_data, MDDDataset, DataCollatorCTCWithPadding, load_vocab
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


def evaluate(cfg: Config, device: torch.device):
    df = pd.read_csv(cfg.metadata_path)
    _, _, test_df = split_data(df, cfg)
    print(f"Test samples: {len(test_df)}")

    vocab = load_vocab(cfg.vocab_path)
    vocab_size = len(vocab)
    id2token = {v: k for k, v in vocab.items()}

    model = create_model(cfg, vocab_size)
    model.load_state_dict(
        torch.load(
            os.path.join(cfg.checkpoint_dir, "model.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    model.to(device)

    # Create CTC decoder for beam search
    decoder = create_ctc_decoder(vocab)
    beam_type = f"beam search (width={cfg.beam_width})" if cfg.beam_width > 1 else "greedy"
    print(f"Decoding: {beam_type}")

    processor = Wav2Vec2Processor.from_pretrained(cfg.processor_dir)

    test_dataset = MDDDataset(test_df, cfg.audio_dir, vocab, cfg, is_train=False)
    collator = DataCollatorCTCWithPadding(processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    predictions = predict(model, test_loader, device, decoder, cfg.beam_width)
    results = decode_to_text(predictions, id2token)

    results_df = pd.DataFrame({"predict": results})
    results_path = os.path.join(cfg.output_dir, "predictions.csv")
    results_df.to_csv(results_path, index=False)

    ground_truth = pd.DataFrame(
        {
            "canonical": test_df["canonical"],
            "transcript": test_df["transcript"],
        }
    )
    gt_path = os.path.join(cfg.output_dir, "ground_truth.csv")
    ground_truth.to_csv(gt_path, index=False)

    print(f"Predictions saved to {results_path}")
    print(f"Ground truth saved to {gt_path}")

    eval_script = "MDD-Metrics/evaluate.py"
    result = subprocess.run(
        ["python", eval_script, gt_path, results_path],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)