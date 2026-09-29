"""FastAPI demo server. Runs on Kaggle (GPU if there is one, else CPU); the notebook
exposes it through a tunnel so the frontend on the laptop can reach it.
Run from bioner/:  python -m uvicorn backend.app:app --port 8000
"""
import json
import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # import our src/ modules

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

import baseline
import config
from data import bio_to_spans, featurize, load_csv, spans_to_bio
from decode import decode, saved_threshold
from span_model import load_span, predict_probs
from train import get_device

TOKEN_RE = re.compile(r"\w+|[^\w\s]")   # words, and each punctuation mark alone; matches all 6,993 dataset sentences
TOP_K = 15

# ---- load everything once, at startup ----
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
span_model = load_span(config.CKPT_DIR / "span", device)
base_model = baseline.build_model(config.CKPT_DIR / "baseline").to(device).eval()
threshold = saved_threshold(config.CKPT_DIR / "span")
val_f1 = {m: json.loads((config.CKPT_DIR / m / "metrics.json").read_text())["val_f1"]
          for m in ("span", "baseline")}


def load_examples(n=10):
    """First n test sentences (8-30 words) where our span model found a disease."""
    if not config.SUBMISSION_CSV.exists():
        return []
    test, sub = load_csv(config.TEST_CSV), pd.read_csv(config.SUBMISSION_CSV)
    ok = sub["ner_tags"].str.contains("B-Disease") & test["tokens"].str.len().between(8, 30)
    return test.loc[ok, "text"].head(n).tolist()


examples = load_examples()

app = FastAPI(title="Team Lannisters BioNER")
app.add_middleware(CORSMiddleware, allow_origins=["*"],     # the frontend is a local file / localhost page
                   allow_methods=["GET", "POST"], allow_headers=["*"])


# ---- inference helpers ----
def tokenize(text):
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        raise HTTPException(422, "text has no words")
    return tokens


def entity(tokens, s, e, score):
    return {"start": s, "end": e, "text": " ".join(tokens[s:e + 1]), "score": round(score, 4)}


def run_span(tokens):
    probs, mask = predict_probs(span_model, [featurize(tokenizer, tokens)], device)[0]   # (W, L)
    chosen = decode(probs, mask, threshold)                  # [(start, end, score)], no overlaps
    ranked = sorted(mask.nonzero().tolist(), key=lambda c: -probs[c[0], c[1]].item())[:TOP_K]
    return {
        "tags": spans_to_bio([(s, e) for s, e, _ in chosen], len(tokens)),
        "entities": [entity(tokens, s, e, p) for s, e, p in chosen],
        "candidates": [entity(tokens, s, s + k, probs[s, k].item()) for s, k in ranked],
        # grid[s][k] = P(disease) of the span (s, s+k); None where the span runs off the sentence
        "grid": [[round(p, 4) if m else None for p, m in zip(pr, mr)]
                 for pr, mr in zip(probs.tolist(), mask.tolist())],
        "threshold": threshold,
    }


def run_baseline(tokens):
    p = baseline.word_probs(base_model, [featurize(tokenizer, tokens)], device)[0]    # (W, 3)
    tags = [config.LABELS[i] for i in p.argmax(-1).tolist()]
    tags += ["O"] * (len(tokens) - len(tags))                                          # truncated -> O
    conf = p.max(-1).values                                   # probability of each chosen tag
    return {
        "tags": tags,
        # entity score = mean confidence of its words' tags
        "entities": [entity(tokens, s, e, conf[s:e + 1].mean().item()) for s, e in bio_to_spans(tags)],
    }


# ---- endpoints ----
class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model: Literal["span", "baseline"] = "span"


class CompareIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model_name": config.MODEL_NAME,
            "val_f1": val_f1, "threshold": threshold}


@app.post("/predict")
def predict(req: PredictIn):
    tokens = tokenize(req.text)
    out = run_span(tokens) if req.model == "span" else run_baseline(tokens)
    return {"tokens": tokens, "model": req.model, **out}


@app.post("/compare")
def compare(req: CompareIn):
    tokens = tokenize(req.text)
    span, base = run_span(tokens), run_baseline(tokens)
    disagree = [i for i, (a, b) in enumerate(zip(span["tags"], base["tags"])) if a != b]
    return {"tokens": tokens, "span": span, "baseline": base, "disagree": disagree}


@app.get("/examples")
def get_examples():
    return {"examples": examples}
