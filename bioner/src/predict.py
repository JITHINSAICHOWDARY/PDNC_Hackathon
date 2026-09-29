"""Test-set inference -> submission CSV (columns id,ner_tags; one tag per token; test.csv order).
Run from bioner/ on Kaggle:
    python src/predict.py                     # span model  -> submission/submission.csv
    python src/predict.py --model baseline    # baseline    -> submission/submission_baseline.csv
"""
import argparse

import pandas as pd
from transformers import AutoTokenizer

import baseline
import config
from data import featurize, load_csv
from decode import decode_to_bio, saved_threshold
from span_model import load_span, predict_probs
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["span", "baseline"], default="span")
    args = parser.parse_args()

    device = get_device()
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    test = load_csv(config.TEST_CSV)                              # keeps test.csv row order
    items = [featurize(tok, t) for t in test["tokens"]]

    # words past 256 subwords have no vector; decode_to_bio / baseline.predict tag them O
    truncated = sum(len(it["word_index"]) < it["n_words"] for it in items)
    print(f"sentences truncated at {config.MAX_LENGTH} subwords (extra words tagged O): {truncated}")

    ckpt = config.CKPT_DIR / args.model
    if args.model == "span":
        threshold = saved_threshold(ckpt)                         # tuned on validation in Phase 5
        print(f"threshold: {threshold}")
        probs = predict_probs(load_span(ckpt, device), items, device)
        preds = [decode_to_bio(p, m, it["n_words"], threshold) for (p, m), it in zip(probs, items)]
        out = config.SUBMISSION_CSV
    else:
        preds = baseline.predict(baseline.build_model(ckpt).to(device), items, device)
        out = config.SUBMISSION_CSV.with_name("submission_baseline.csv")

    # The validator silently pads or cuts bad rows, so we fail loudly here instead
    for sid, tokens, tags in zip(test["id"], test["tokens"], preds):
        assert len(tags) == len(tokens), f"{sid}: {len(tags)} tags for {len(tokens)} tokens"
        assert set(tags) <= set(config.LABELS), f"{sid}: unknown tag"

    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": test["id"], "ner_tags": [" ".join(t) for t in preds]}).to_csv(out, index=False)
    n_ent = sum(t.count("B-Disease") for t in preds)
    print(f"wrote {out}: {len(preds)} rows, {n_ent} disease entities")


if __name__ == "__main__":
    main()
