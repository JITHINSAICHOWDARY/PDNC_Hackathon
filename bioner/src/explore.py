"""Print dataset statistics so we can confirm the numbers in our pitch.
Uses only the standard library, so it runs before any pip install.
Run from bioner/:  python src/explore.py
"""
import csv
import json
from collections import Counter

import config


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def entity_lengths(tags):
    """Length in words of each B-Disease... run of I-Disease."""
    lengths = []
    for t in tags:
        if t == "B-Disease":
            lengths.append(1)                # a new entity starts
        elif t == "I-Disease" and lengths:
            lengths[-1] += 1                 # extend the current entity
    return lengths


train = load(config.TRAIN_CSV)
test = load(config.TEST_CSV)

tag_counts = Counter()
len_counts = Counter()
with_disease = 0
i_after_o = 0
longest = 0
for row in train:
    tokens, tags = json.loads(row["tokens"]), json.loads(row["ner_tags"])
    assert len(tokens) == len(tags), row["id"]   # one tag per token
    tag_counts.update(tags)
    with_disease += "B-Disease" in tags
    # "O" before the start, so an I at position 0 also counts as I-after-O
    i_after_o += sum(p == "O" and c == "I-Disease" for p, c in zip(["O"] + tags, tags))
    len_counts.update(entity_lengths(tags))
    longest = max(longest, len(tokens))

n_ent = sum(len_counts.values())
covered = sum(n for L, n in len_counts.items() if L <= config.MAX_SPAN_LEN)

print(f"train sentences: {len(train)}   test sentences: {len(test)}")
print("tag counts:", dict(tag_counts))
print(f"sentences with >=1 disease: {with_disease}")
print(f"I-Disease directly after O: {i_after_o}")
print(f"longest train sentence: {longest} words;  longest test: "
      f"{max(len(json.loads(r['tokens'])) for r in test)} words")
print(f"\nentities: {n_ent}")
for L in sorted(len_counts):
    print(f"  {L:>2} words: {len_counts[L]}")
print(f"MAX_SPAN_LEN={config.MAX_SPAN_LEN} covers {covered}/{n_ent} = {covered / n_ent:.1%}")
