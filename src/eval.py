import os
import subprocess

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from config import Config
from data import split_data, MDDDataset, DataCollatorCTCWithPadding, load_vocab
from model import create_model


@torch.no_grad()
def predict(model, dataloader, device, blank_id=0):
    model.eval()
    all_preds = []

    for batch in dataloader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        canonical_ids = batch["canonical_ids"].to(device)

        output = model(input_values, canonical_ids, attention_mask=attention_mask)
        logits = output.logits
        pred_ids = torch.argmax(logits, dim=-1)

        for b in range(pred_ids.shape[0]):
            ids = pred_ids[b].tolist()
            collapsed = []
            prev = None
            for tid in ids:
                if tid != prev and tid != blank_id:
                    collapsed.append(tid)
                prev = tid
            all_preds.append(collapsed)

    return all_preds


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

    processor = Wav2Vec2Processor.from_pretrained(cfg.processor_dir)

    test_dataset = MDDDataset(test_df, cfg.audio_dir, vocab, cfg)
    collator = DataCollatorCTCWithPadding(processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    predictions = predict(model, test_loader, device)

    results = []
    for pred_ids in predictions:
        tokens = [id2token.get(tid, "") for tid in pred_ids]
        tokens = [t for t in tokens if t and t != "[CTC_BLANK]"]
        results.append(" ".join(tokens))

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