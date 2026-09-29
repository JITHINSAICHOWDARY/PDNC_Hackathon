"""Sanity checks for data.py. Run from bioner/:  python src/test_data.py"""
from transformers import AutoTokenizer

import config
from data import (bio_to_spans, enumerate_spans, featurize, load_csv,
                  spans_to_bio, train_val_split)

train = load_csv(config.TRAIN_CSV)
test = load_csv(config.TEST_CSV)
tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# 1. BIO -> spans -> BIO gives back the original tags on every training sentence
for tags in train["ner_tags"]:
    assert spans_to_bio(bio_to_spans(tags), len(tags)) == tags
print(f"round trip OK on {len(train)} sentences")

# 2. Hand-checked spans, including an entity that runs to the end of the sentence
assert bio_to_spans(["O", "B-Disease", "I-Disease", "O", "B-Disease"]) == [(1, 2), (4, 4)]
assert bio_to_spans(["B-Disease", "B-Disease"]) == [(0, 0), (1, 1)]

# 3. Span count: n words, max length L -> sum over starts of min(L, words left)
assert len(enumerate_spans(3, 10)) == 6            # (0,0)(0,1)(0,2)(1,1)(1,2)(2,2)
assert len(enumerate_spans(20, 10)) == sum(min(10, 20 - s) for s in range(20))

# 4. Alignment: each word's first subword starts that word, and carries its label
truncated = 0
for tokens, tags in zip(train["tokens"], train["ner_tags"]):
    item = featurize(tok, tokens, tags)
    assert item["labels"][0] == -100                         # [CLS]
    kept = len(item["word_index"])
    truncated += kept < len(tokens)
    for w, pos in enumerate(item["word_index"]):
        assert item["labels"][pos] == config.LABEL2ID[tags[w]]
    assert sum(l != -100 for l in item["labels"]) == kept    # exactly one label per word
print(f"alignment OK; train sentences truncated at {config.MAX_LENGTH}: {truncated}")

longest = max(len(featurize(tok, t)["input_ids"]) for t in test["tokens"])
print(f"longest test sentence in subwords (incl. [CLS]/[SEP]): {longest}")
assert all(len(featurize(tok, t)["word_index"]) == len(t) for t in test["tokens"])   # every test word kept

# A zero-width space gets no subword from BERT; featurize must still keep one vector per word
item = featurize(tok, ["breast", "\u200b", "cancer"], ["B-Disease", "O", "B-Disease"])
assert len(item["word_index"]) == 3 and sum(l != -100 for l in item["labels"]) == 3

# 5. Fixed split
tr, va = train_val_split(train)
assert (len(tr), len(va)) == (4250, 750)
assert va["id"].tolist() == train_val_split(train)[1]["id"].tolist()   # same seed, same split

# Show one example that contains a disease
i = next(i for i, t in enumerate(train["ner_tags"]) if "B-Disease" in t)
tokens, tags = train["tokens"][i], train["ner_tags"][i]
item = featurize(tok, tokens, tags)
print("\nexample:", list(zip(tokens, tags)))
print("subwords:", tok.convert_ids_to_tokens(item["input_ids"]))
print("word_index:", item["word_index"], " gold spans:", item["gold_spans"])
print("\nall tests passed")
