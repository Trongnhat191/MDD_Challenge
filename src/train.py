import os

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import split_data, MDDDataset, DataCollatorCTCWithPadding, create_processor, load_vocab
from model import create_model


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def compute_per(logits: torch.Tensor, label_ids: torch.Tensor, blank_id: int = 0) -> float:
    pred_ids = torch.argmax(logits, dim=-1)

    total_ed = 0
    total_ref_len = 0
    for b in range(pred_ids.shape[0]):
        raw_ids = pred_ids[b].tolist()
        collapsed = []
        prev = None
        for tid in raw_ids:
            if tid != prev and tid != blank_id:
                collapsed.append(tid)
            prev = tid
        ref_seq = [tid.item() for tid in label_ids[b] if tid.item() != -100]
        total_ed += _edit_distance(collapsed, ref_seq)
        total_ref_len += len(ref_seq)
    return total_ed / total_ref_len if total_ref_len > 0 else 0.0


def _edit_distance(pred, ref):
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def train_one_epoch(model, dataloader, optimizer, scheduler, device, cfg, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        canonical_ids = batch["canonical_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type):
            outputs = model(
                input_values,
                canonical_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.gradient_accumulation

        loss.backward()

        if (step + 1) % cfg.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.gradient_accumulation
        progress_bar.set_postfix({"loss": f"{loss.item() * cfg.gradient_accumulation:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_logits = []
    all_label_ids = []

    for batch in tqdm(dataloader, desc="Validating"):
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        canonical_ids = batch["canonical_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_values,
            canonical_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()

        all_logits.append(outputs.logits.cpu())
        all_label_ids.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    logits = torch.cat(all_logits, dim=0)
    label_ids = torch.cat(all_label_ids, dim=0)
    per = compute_per(logits, label_ids)

    return avg_loss, per


def train(cfg: Config, device: torch.device):
    df = pd.read_csv(cfg.metadata_path)
    train_df, val_df, _ = split_data(df, cfg)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    processor = create_processor(cfg.vocab_path, cfg)
    vocab = load_vocab(cfg.vocab_path)
    vocab_size = len(vocab)

    model = create_model(cfg, vocab_size)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    train_dataset = MDDDataset(train_df, cfg.audio_dir, vocab, cfg)
    val_dataset = MDDDataset(val_df, cfg.audio_dir, vocab, cfg)

    collator = DataCollatorCTCWithPadding(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg, epoch)
        val_loss, val_per = validate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_PER={val_per:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "model.pt"))
            processor.save_pretrained(cfg.processor_dir)
            print(f"Checkpoint saved to {cfg.checkpoint_dir}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")