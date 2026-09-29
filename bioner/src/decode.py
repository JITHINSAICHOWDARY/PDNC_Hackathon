"""Turn span probabilities into non-overlapping entities, then into BIO tags,
and pick the probability threshold on the validation set."""
import json

import config
from data import spans_to_bio


def decode(probs, mask, threshold=config.THRESHOLD):
    """probs, mask: (W, L) for ONE sentence. Returns [(start, end, score)] with no overlaps.
    Greedy: take the most confident span, block its words, repeat."""
    cells = ((probs >= threshold) & mask).nonzero().tolist()          # [[s, k], ...]
    candidates = sorted(((probs[s, k].item(), s, s + k) for s, k in cells), reverse=True)
    taken, chosen = set(), []
    for score, s, e in candidates:
        words = set(range(s, e + 1))
        if not words & taken:                                         # no shared word with a kept span
            taken |= words
            chosen.append((s, e, score))
    return sorted(chosen)                                             # left to right


def decode_to_bio(probs, mask, n_words, threshold=config.THRESHOLD):
    """Always a valid BIO list of length n_words (words lost to truncation stay O)."""
    return spans_to_bio([(s, e) for s, e, _ in decode(probs, mask, threshold)], n_words)


THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(9)]    # 0.30, 0.35, ..., 0.70


def tune_threshold(probs_masks, items, gold_tags):
    """Try every threshold on validation; return (best threshold, {threshold: span F1})."""
    from seqeval.metrics import f1_score
    scores = {}
    for t in THRESHOLDS:
        preds = [decode_to_bio(p, m, it["n_words"], t) for (p, m), it in zip(probs_masks, items)]
        scores[t] = f1_score(gold_tags, preds)
    return max(scores, key=scores.get), scores


def saved_threshold(ckpt_dir):
    """The tuned threshold stored next to the span checkpoint (config default if not tuned yet)."""
    metrics = json.loads((ckpt_dir / "metrics.json").read_text())
    return metrics.get("threshold", config.THRESHOLD)


if __name__ == "__main__":
    import torch
    probs = torch.zeros(5, config.MAX_SPAN_LEN)
    probs[1, 1] = 0.9      # span (1, 2)  best
    probs[2, 0] = 0.8      # span (2, 2)  overlaps (1, 2) -> dropped
    probs[4, 0] = 0.6      # span (4, 4)  kept
    probs[0, 0] = 0.4      # span (0, 0)  below threshold
    mask = torch.ones_like(probs, dtype=torch.bool)
    assert decode(probs, mask, 0.5) == [(1, 2, probs[1, 1].item()), (4, 4, probs[4, 0].item())]
    assert decode_to_bio(probs, mask, 6, 0.5) == ["O", "B-Disease", "I-Disease", "O", "B-Disease", "O"]
    print("decode check passed")
