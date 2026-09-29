"""Train a model and keep the checkpoint with the best validation span F1.
Run from bioner/:
    python src/train.py --model baseline
    python src/train.py --model span
    python src/train.py --model span --debug       # 50 sentences, 1 epoch: checks the code runs
"""
import argparse
import json
import random

import torch
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from data import collate, featurize, load_csv, train_val_split


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")          # Kaggle T4
    if torch.backends.mps.is_available():
        return torch.device("mps")           # Apple silicon
    return torch.device("cpu")


def make_scheduler(optimizer, n_steps):
    """Learning rate climbs linearly for the first 10% of steps, then decays to 0."""
    return get_linear_schedule_with_warmup(optimizer, int(config.WARMUP_FRACTION * n_steps), n_steps)


def train_baseline(train_items, val_items, val_tags, tokenizer, device, epochs, out_dir):
    import baseline
    model = baseline.build_model().to(device)
    loader = DataLoader(train_items, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.BASE_LR)
    scheduler = make_scheduler(optimizer, len(loader) * epochs)
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()                                      # dropout on
        total = 0.0
        for batch in loader:
            loss = model(input_ids=batch["input_ids"].to(device),
                         attention_mask=batch["attention_mask"].to(device),
                         labels=batch["labels"].to(device)).loss   # cross-entropy, skips -100
            loss.backward()                                # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()

        f1 = f1_score(val_tags, baseline.predict(model, val_items, device))  # entity-level
        print(f"epoch {epoch}: train loss {total / len(loader):.4f}  val span F1 {f1:.4f}")
        if f1 > best:                                      # keep only the best epoch
            best = f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            (out_dir / "metrics.json").write_text(json.dumps({"val_f1": f1, "epoch": epoch}))
            print(f"  saved to {out_dir}")
    print(f"best val span F1: {best:.4f}")


def train_span(train_items, val_items, val_tags, tokenizer, device, epochs, out_dir):
    from decode import decode_to_bio
    from sampling import sample_spans, span_targets
    from span_model import SpanModel, predict_probs

    model = SpanModel().to(device)
    # collate the tensors, and pass the gold span lists alongside them
    loader = DataLoader(train_items, batch_size=config.BATCH_SIZE, shuffle=True,
                        collate_fn=lambda items: (collate(items), [it["gold_spans"] for it in items]))
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.ENCODER_LR},  # pretrained: small steps
        {"params": head, "lr": config.HEAD_LR},                          # new layers: bigger steps
    ])
    scheduler = make_scheduler(optimizer, len(loader) * epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()      # sigmoid + binary cross-entropy in one stable op
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch, gold in loader:
            logits, mask = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                 batch["word_index"].to(device), batch["word_mask"].to(device))  # (B, W, L)
            labels, hard = span_targets(gold, logits.shape[1], device)                            # (B, W, L)
            keep = sample_spans(labels, hard, mask)            # positives + hard + sampled easy
            loss = loss_fn(logits[keep], labels[keep])         # 1-D over the kept spans
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()

        # decode validation to BIO, then the same seqeval span F1 as the baseline
        preds = [decode_to_bio(p, m, it["n_words"])
                 for (p, m), it in zip(predict_probs(model, val_items, device), val_items)]
        f1 = f1_score(val_tags, preds)
        print(f"epoch {epoch}: train loss {total / len(loader):.4f}  val span F1 {f1:.4f}")
        if f1 > best:
            best = f1
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "span_model.pt")   # encoder + head weights
            tokenizer.save_pretrained(out_dir)
            (out_dir / "metrics.json").write_text(json.dumps({"val_f1": f1, "epoch": epoch}))
            print(f"  saved to {out_dir}")
    print(f"best val span F1: {best:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "span"], required=True)
    parser.add_argument("--epochs", type=int, help="override the config value")
    parser.add_argument("--debug", action="store_true",
                        help="50 sentences, 1 epoch, saves to checkpoints/debug (laptop check only)")
    args = parser.parse_args()

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)                        # same init and shuffling every run
    device = get_device()
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_df, val_df = train_val_split(load_csv(config.TRAIN_CSV))
    epochs = args.epochs or (config.BASE_EPOCHS if args.model == "baseline" else config.SPAN_EPOCHS)
    out_dir = config.CKPT_DIR / args.model
    if args.debug:
        train_df, val_df, epochs = train_df[:50], val_df[:50], 1
        out_dir = config.CKPT_DIR / "debug"             # never overwrite a real checkpoint
    train_items = [featurize(tokenizer, t, g) for t, g in zip(train_df["tokens"], train_df["ner_tags"])]
    val_items = [featurize(tokenizer, t, g) for t, g in zip(val_df["tokens"], val_df["ner_tags"])]
    val_tags = val_df["ner_tags"].tolist()

    train_fn = train_baseline if args.model == "baseline" else train_span
    train_fn(train_items, val_items, val_tags, tokenizer, device, epochs, out_dir)


if __name__ == "__main__":
    main()
