"""Evaluate baseline vs span model on the validation split (the same 750 sentences for both).
Also tunes the span threshold and saves it to checkpoints/span/metrics.json.
Run from bioner/ after training both models:  python src/evaluate.py
"""
import json
from collections import Counter

from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score as sk_f1   # same token metrics as validator.py
from transformers import AutoTokenizer

import baseline
import config
from data import bio_to_spans, featurize, load_csv, train_val_split
from decode import decode_to_bio, tune_threshold
from span_model import load_span, predict_probs
from train import get_device

BUCKETS = ["1", "2", "3", "4+"]


def bucket(span):
    n = span[1] - span[0] + 1
    return str(n) if n < 4 else "4+"


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def f1_by_length(gold_tags, pred_tags):
    """Exact-span F1 per entity length. Gold spans count toward recall and predicted spans
    toward precision, each in its own length bucket."""
    tp, n_pred, n_gold = Counter(), Counter(), Counter()
    for g, p in zip(gold_tags, pred_tags):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        n_gold.update(bucket(s) for s in gs)
        n_pred.update(bucket(s) for s in ps)
        tp.update(bucket(s) for s in gs & ps)                    # exact matches only
    out = {}
    for b in BUCKETS:
        p = tp[b] / n_pred[b] if n_pred[b] else 0.0
        r = tp[b] / n_gold[b] if n_gold[b] else 0.0
        out[b] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def metrics(gold_tags, pred_tags):
    flat_g = [t for s in gold_tags for t in s]
    flat_p = [t for s in pred_tags for t in s]
    return {
        "span F1": f1_score(gold_tags, pred_tags),
        "span precision": precision_score(gold_tags, pred_tags),
        "span recall": recall_score(gold_tags, pred_tags),
        "token accuracy": accuracy_score(flat_g, flat_p),
        "token macro F1": sk_f1(flat_g, flat_p, average="macro", zero_division=0),
        **{f"F1 len {b}": v for b, v in f1_by_length(gold_tags, pred_tags).items()},
    }


def example_errors(tokens_list, gold_tags, pred_tags, quota=(("partial", 4), ("missed", 3), ("false positive", 3))):
    """10 examples: partial = predicted span overlaps gold but edges differ,
    missed = no prediction touches the gold span, false positive = prediction touches no gold span."""
    found = {kind: [] for kind, _ in quota}
    text = lambda toks, s: " ".join(toks[s[0]:s[1] + 1])
    for toks, g, p in zip(tokens_list, gold_tags, pred_tags):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        for s in gs - ps:
            near = [q for q in ps if overlaps(s, q)]
            kind = "partial" if near else "missed"
            found[kind].append(f"gold '{text(toks, s)}'  pred {[text(toks, q) for q in near]}")
        for s in ps - gs:
            if not any(overlaps(s, q) for q in gs):
                found["false positive"].append(f"pred '{text(toks, s)}'  in: {' '.join(toks)[:90]}")
    return [f"[{kind}] {line}" for kind, n in quota for line in found[kind][:n]]


def main():
    device = get_device()
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    _, val_df = train_val_split(load_csv(config.TRAIN_CSV))
    items = [featurize(tok, t, g) for t, g in zip(val_df["tokens"], val_df["ner_tags"])]
    gold = val_df["ner_tags"].tolist()

    base_model = baseline.build_model(config.CKPT_DIR / "baseline").to(device)
    preds = {"baseline": baseline.predict(base_model, items, device)}
    del base_model                                              # free GPU memory

    span_dir = config.CKPT_DIR / "span"
    probs = predict_probs(load_span(span_dir, device), items, device)   # BioBERT runs once
    best_t, scores = tune_threshold(probs, items, gold)                 # decoding reruns cheaply
    print("threshold -> val span F1:", {t: round(f, 4) for t, f in scores.items()})
    print(f"best threshold: {best_t}")
    if best_t in (min(scores), max(scores)):   # the true optimum may lie outside 0.30-0.70
        print("WARNING: best threshold is at the edge of the search range; consider widening THRESHOLDS")
    saved = json.loads((span_dir / "metrics.json").read_text())
    saved.update(threshold=best_t, val_f1=scores[best_t])
    (span_dir / "metrics.json").write_text(json.dumps(saved))    # predict.py and the backend read it
    preds["span"] = [decode_to_bio(p, m, it["n_words"], best_t) for (p, m), it in zip(probs, items)]

    results = {name: metrics(gold, p) for name, p in preds.items()}
    rows = ["| metric | baseline (BIO) | span model |", "|---|---|---|"]
    rows += [f"| {k} | {results['baseline'][k]:.4f} | {results['span'][k]:.4f} |" for k in results["span"]]
    table = "\n".join(rows)
    print("\n" + table)

    report = [table]
    for name, p in preds.items():
        errs = example_errors(val_df["tokens"].tolist(), gold, p)
        print(f"\nexample errors, {name}:")
        print("\n".join("  " + e for e in errs))
        report += [f"\n### Example errors: {name}", *[f"- {e}" for e in errs]]
    (config.ROOT / "results.md").write_text("\n".join(report) + "\n")
    print(f"\nwrote {config.ROOT / 'results.md'}")


if __name__ == "__main__":
    main()
