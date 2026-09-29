"""BIO token-classification baseline: BioBERT plus one linear layer that gives
every subword a score for O / B-Disease / I-Disease."""
import torch
from transformers import AutoModelForTokenClassification

import config
from data import collate


def build_model(path=config.MODEL_NAME):
    """Pretrained BioBERT with a fresh 3-way classification layer on top.
    Pass a checkpoint folder as `path` to reload a trained model."""
    return AutoModelForTokenClassification.from_pretrained(
        path,
        num_labels=len(config.LABELS),
        id2label=dict(enumerate(config.LABELS)),
        label2id=config.LABEL2ID,
    )


@torch.no_grad()                      # inference only: no gradients, less memory
def word_probs(model, items, device):
    """Featurized sentences -> per sentence a (words kept, 3) tensor of O/B/I probabilities,
    read at each word's first subword only."""
    model.eval()                      # turns dropout off
    out = []
    for i in range(0, len(items), config.BATCH_SIZE):
        chunk = items[i:i + config.BATCH_SIZE]
        batch = collate(chunk)
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits  # (B, T, 3)
        probs = logits.softmax(-1).cpu()                                          # (B, T, 3)
        out += [row[item["word_index"]] for row, item in zip(probs, chunk)]
    return out


def predict(model, items, device):
    """Featurized sentences -> one BIO tag list per sentence (one tag per original word)."""
    out = []
    for p, item in zip(word_probs(model, items, device), items):
        tags = [config.LABELS[i] for i in p.argmax(-1).tolist()]        # best label per word
        out.append(tags + ["O"] * (item["n_words"] - len(tags)))       # truncated words -> O
    return out
