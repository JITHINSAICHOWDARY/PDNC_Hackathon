"""CSV parsing, subword alignment and span helpers shared by both models."""
import json

import pandas as pd
import torch

import config


def load_csv(path):
    """Read a dataset CSV and turn the JSON-string columns into Python lists."""
    df = pd.read_csv(path)
    df["tokens"] = df["tokens"].apply(json.loads)
    if "ner_tags" in df:                                  # test.csv has no tags
        df["ner_tags"] = df["ner_tags"].apply(json.loads)
    return df


def train_val_split(df):
    """Fixed-seed 85/15 split so every run validates on the same sentences."""
    val = df.sample(frac=config.VAL_FRACTION, random_state=config.SEED)
    return df.drop(val.index).reset_index(drop=True), val.reset_index(drop=True)


def first_subword_index(word_ids):
    """Position of each word's first subword. Words cut off by truncation are missing,
    so len(result) can be smaller than the number of words."""
    index, prev = [], None
    for pos, w in enumerate(word_ids):
        if w is not None and w != prev:   # None = [CLS]/[SEP]; a new word id = first piece
            index.append(pos)
        prev = w
    return index


def baseline_labels(word_ids, tags):
    """One label per subword: the word's label on its first piece, -100 elsewhere.
    -100 is the index PyTorch's cross-entropy ignores."""
    labels, prev = [], None
    for w in word_ids:
        if w is None or w == prev:
            labels.append(-100)          # special token or a continuation piece
        else:
            labels.append(config.LABEL2ID[tags[w]])
        prev = w
    return labels


def bio_to_spans(tags):
    """BIO list -> [(start, end)] with inclusive word indices.
    A stray I after O starts a new entity (the same rule seqeval uses)."""
    spans, start = [], None
    for i, t in enumerate(tags):
        if t == "B-Disease" or (t == "I-Disease" and start is None):
            if start is not None:
                spans.append((start, i - 1))   # close the previous entity
            start = i
        elif t == "O" and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(tags) - 1))   # entity runs to the end of the sentence
    return spans


def spans_to_bio(spans, n_words):
    """[(start, end)] -> BIO list. Assumes the spans do not overlap."""
    tags = ["O"] * n_words
    for s, e in spans:
        tags[s] = "B-Disease"
        for i in range(s + 1, e + 1):
            tags[i] = "I-Disease"
    return tags


def enumerate_spans(n_words, max_len=config.MAX_SPAN_LEN):
    """Every candidate (start, end) with length <= max_len, in a fixed order."""
    return [(s, e) for s in range(n_words) for e in range(s, min(s + max_len, n_words))]


def featurize(tokenizer, tokens, tags=None):
    """One sentence -> the dict both models train from."""
    encode = lambda toks: tokenizer(toks, is_split_into_words=True, truncation=True,
                                    max_length=config.MAX_LENGTH)
    enc = encode(tokens)
    # A token made only of characters BERT drops (zero-width space, soft hyphen, ...) gets no
    # subword, and every later word would shift by one. Swap such tokens for [UNK] and re-encode.
    # (The dataset is pure ASCII, so this only matters for text pasted into the demo.)
    seen = {w for w in enc.word_ids() if w is not None}
    kept = max(seen) + 1 if seen else len(tokens)      # words from here on were cut by truncation
    if len(seen) < kept:
        tokens = [t if i in seen or i >= kept else tokenizer.unk_token for i, t in enumerate(tokens)]
        enc = encode(tokens)
    word_ids = enc.word_ids()
    item = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "word_index": first_subword_index(word_ids),   # one entry per word kept
        "n_words": len(tokens),                        # original count, for padding with O
    }
    if tags is not None:
        item["labels"] = baseline_labels(word_ids, tags)
        item["gold_spans"] = bio_to_spans(tags)
    return item


def collate(items):
    """Pad a list of featurized sentences into one batch of tensors."""
    T = max(len(it["input_ids"]) for it in items)            # longest sentence in the batch
    def pad(key, value):
        return torch.tensor([it[key] + [value] * (T - len(it[key])) for it in items])
    batch = {
        "input_ids": pad("input_ids", 0),           # pad id; masked out, so its value is irrelevant
        "attention_mask": pad("attention_mask", 0), # 0 = the encoder ignores this position
    }
    if "labels" in items[0]:
        batch["labels"] = pad("labels", -100)       # padding is ignored by the loss too

    # Word-level tensors for the span model: W = most words kept in this batch
    W = max(len(it["word_index"]) for it in items)
    batch["word_index"] = torch.tensor([it["word_index"] + [0] * (W - len(it["word_index"]))
                                        for it in items])            # (B, W) subword positions
    batch["word_mask"] = torch.tensor([[1] * len(it["word_index"]) + [0] * (W - len(it["word_index"]))
                                       for it in items]).bool()      # (B, W) True = real word
    return batch
