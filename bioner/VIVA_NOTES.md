# Viva Notes (Team Lannisters)

Plain-English notes for defending every file. Each section has what the file does, then the questions judges are likely to ask, with short answers.

---

## Numbers to remember
- **Data:** 5,000 training sentences split into 4,250 train and 750 validation (seed 42), plus 1,993 test sentences.
  - Tag counts: O 116,464, B 4,753, I 5,738.
  - 2,707 sentences contain at least one disease.
- **Entity lengths:** 43% are 1 word and 27% are 2 words. The longest is 20 words. `MAX_SPAN_LEN = 10` covers 4,739 of 4,753 entities (99.7%).
- **Sentence lengths:** 25.4 words on average. The longest is 123 words, or 146 subwords, so the 256-subword limit never truncates.
- **Candidate spans:** about 209 per sentence, 1.05 million in the training set. Only 0.45% of them are diseases.
- **Hard negatives:** 43 per gold entity on average. Easy negatives: 837k in total, sampled down to 10 per positive per batch.
- **Model size:** the span head adds 0.81M parameters on top of BioBERT's roughly 108M.
- **Baseline validation span F1:** 0.894. The span model's number comes from `results.md` once the Kaggle run finishes.

---

## The five questions we will definitely get

**Why span scoring and not BIO?**
- **BIO judges words one at a time.** If a 4-word disease gets one wrong tag, the whole entity counts as wrong, because the metric is exact-span F1.
- **BIO can produce illegal output.** For example `O I-Disease`.
- **Our model scores the whole phrase.** "Is *hereditary breast and ovarian cancer* a disease?" is exactly the unit the metric counts.
- **Our output is always valid BIO.** Spans are converted to tags, so the tags can't be illegal.
- **The span vector includes length information.** It has the start word, the end word, the mean of the words inside, and a width embedding. BIO tags don't have that.

**What is a hard negative?**
- **Definition:** a wrong span that overlaps a real disease, for example "breast and ovarian" when the answer is "hereditary breast and ovarian cancer".
- **Why they matter:** these are exactly the boundary mistakes that cost F1, so we always train on all of them.
- **Easy negatives:** spans nowhere near a disease, such as "the patients were". They're easy to reject, so we randomly keep only 10 per positive.

**Why use only the first subword?**
- **One vector per word:** BioBERT splits words, e.g. "lipoma" → "lip ##oma", but tags belong to whole words.
- **Standard practice:** using the first piece is the usual approach and the one the hackathon README recommends.
- **Context is already there:** self-attention means the first piece already "sees" the rest of the word.
- **Baseline:** the other pieces get label `-100`, which the loss ignores, so each word is counted exactly once.
- **Span model:** we gather the first-piece vectors into a word-level tensor of shape (B, W, 768).

**What does biaffine mean?**
- **Bi-linear plus affine:** the score is `h_startᵀ · U · h_end` (bilinear) plus linear terms and a bias (affine).
- **Why it helps:** it scores how well a start word and an end word go together, e.g. "hereditary … cancer".
- **Our version:** the FFN head supplies the affine part, and `U` is 128×128 after projecting both words from 768 to 128 dimensions.
- **Why we project first:** a full 768×768 `U` has 590k weights multiplying large vectors, which is unstable at learning rate 1e-3. The biaffine parser of Dozat & Manning also projects first.
- **Why `U` starts at zero:** the untrained biaffine adds no noise at the start.

**Why tune the threshold on validation data?**
- **The default isn't optimal:** the model's probabilities aren't perfectly calibrated, so 0.5 isn't automatically the best cut-off. A lower threshold favours recall and a higher one favours precision.
- **What we try:** 0.30 to 0.70 in steps of 0.05, keeping the best span F1.
- **Never on test:** that would be cheating, and test has no labels anyway.
- **It's cheap:** BioBERT runs once, and only the fast greedy decoder repeats 9 times.
- **Honest caveat:** we report validation F1 at the tuned threshold, which is slightly optimistic. It's one number chosen from 9, so the effect is small.

---

## src/config.py
- Every hyperparameter and path in one place. Every other file imports it.
- The encoder name, `MAX_LENGTH = 256`, the split (15%, seed 42), the learning rates, epochs, `MAX_SPAN_LEN`, and the sizes of the span head.
- `THRESHOLD = 0.5` is only a default. The tuned value lives in `checkpoints/span/metrics.json`.

**Q: Why store the tuned threshold with the checkpoint and not in config?**
It belongs to that specific trained model, and it's tuned on Kaggle. Keeping it next to the weights means it travels with them.

## src/explore.py
- Recomputes the dataset statistics from the pitch using only the standard library.
- Asserts that every training sentence has one tag per token.
- Counts `I` directly after `O` (zero) and the entity-length histogram, which is how we chose `MAX_SPAN_LEN = 10`.

**Q: How did you choose 10?**
It covers 99.7% of gold entities. Going to 20 would double the number of candidate spans to save only 14 entities.

## src/data.py
- `load_csv` parses the JSON `tokens` and `ner_tags` columns. `train_val_split` makes the fixed 85/15 split.
- `first_subword_index` gives the position of each word's first subword. `baseline_labels` gives the word's label to its first piece and `-100` to the other pieces.
- `bio_to_spans` and `spans_to_bio` convert between tags and `(start, end)` word ranges. An `I` after `O` starts a new entity, the same rule seqeval uses.
- `enumerate_spans` lists every span up to 10 words long.
- `featurize` does all of the above for one sentence. `collate` pads a batch: attention mask 0 and label `-100` on padding, plus `word_index` and `word_mask` for the span model.

**Q: What if a word produces no subwords at all?**
Some characters, such as a zero-width space or a soft hyphen, are dropped entirely by BERT's tokenizer, so a token made only of them gets no subword. Every later word would then shift by one. `featurize` detects this, swaps such tokens for `[UNK]`, and re-encodes. The dataset is pure ASCII, so this only protects text pasted into the demo.

**Q: What is -100?**
The index PyTorch's cross-entropy ignores. We use it for special tokens, continuation subwords and padding.

**Q: What happens to words cut by truncation?**
They get no vector, and we tag them `O`. `featurize` keeps both `n_words` and the kept `word_index`, so we know which words are missing. On this data it never happens.

## src/test_data.py
- **Round trip:** BIO → spans → BIO gives back the exact tags on all 5,000 training sentences.
- **Alignment:** every word's first subword carries that word's label, with exactly one label per word.
- **Other checks:** hand-checked span examples, the span-count formula, and that the same seed always gives the same split.

**Q: How do you know your span conversion is correct?**
The round trip is exact on every training sentence. If the conversion lost or shifted anything, that test would fail.

## src/baseline.py
- `AutoModelForTokenClassification`: BioBERT plus one linear layer mapping 768 → 3 (O / B / I) for every subword.
- `word_probs` computes softmax probabilities, read only at each word's first subword.
- `predict` takes the highest-scoring label per word and tags truncated words `O`.

**Q: Why no CRF?**
The baseline is deliberately the plain "before" model. Its weaknesses, such as independent decisions and possible `O I`, are what the span model fixes by construction.

## src/span_model.py
- **Encode:** `encoder(...)` gives shape (B, T, 768). **Gather** first subwords to get word vectors of shape (B, W, 768).
- **Span grid:** cell `[s, k]` is the phrase from word `s` to word `s+k`, shape (B, W, 10). The mask marks spans that stay inside the sentence.
- **Span vector:** `[h_start; h_end; mean; width_emb(32)]`, 2,336 dimensions. The mean uses prefix sums, `cs[e+1] - cs[s]`, so there's no loop.
- **Score:** `FFN(span_vec)` (2336 → 256 → 1, with ReLU and dropout 0.2) plus the biaffine `sᵀUe`. The result is one logit per span, and `sigmoid` of it is P(disease).
- `predict_probs` returns the probability grid per sentence. `load_span` rebuilds the model and loads `span_model.pt`.

**Q: Why store spans as start × width and not start × end?**
Start × end would be W×W, up to 123×123. Start × width is W×10, and almost every cell is a legal candidate. For the heatmap we just draw cell `[s, k]` at column `s+k`.

**Q: Why the mean of the words inside?**
Start and end alone miss the middle, for example "breast **and ovarian**". The mean summarises the whole phrase cheaply.

**Q: What does the width embedding add?**
A learned vector for each length from 1 to 10 words. It lets the model learn how common each length is, since short entities are far more frequent.

## src/sampling.py
- `span_targets` builds a label grid (1 = gold span) and a `hard` grid (overlaps a gold span but isn't the gold span). The overlap test is `start ≤ gold_end and end ≥ gold_start`.
- `sample_spans` keeps every positive and every hard negative, plus `10 × (number of positives)` random easy negatives per batch.
- A batch with no diseases still gets 10 easy negatives, so the model also learns to say "nothing here".

**Q: Why not train on all spans?**
- 99.5% of spans aren't diseases, so the loss would be dominated by trivially easy "no" answers.
- Down-sampling focuses training on the informative spans.

**Q: Is the class imbalance still a problem?**
- It's reduced, not gone: hard negatives average 43 per positive.
- The threshold tuning compensates.
- A `pos_weight` in the loss is a knob we could add if recall is low.

## src/decode.py
- `decode`: keep spans with P ≥ threshold, sort them by score, and accept each one unless it shares a word with a span already accepted (greedy, no overlaps).
- `decode_to_bio`: the accepted spans become `B-Disease I-Disease …`, which is always valid BIO.
- `tune_threshold` tries 0.30–0.70 and returns the best one. `saved_threshold` reads it back.

**Q: Why greedy and not exact?**
Nested or overlapping predictions are rare, and highest-score-first is simple, fast and easy to explain. The exact alternative is weighted interval scheduling by dynamic programming. It's rarely different in practice.

**Q: Why no overlaps?**
BIO tags can't represent nested or overlapping entities, and the dataset has none.

## src/train.py
- `--model baseline|span`, `--debug` (50 sentences, 1 epoch, saves to `checkpoints/debug`), and `--epochs`.
- **Baseline:** AdamW at 3e-5, cross-entropy computed by Hugging Face.
- **Span model:** AdamW with two parameter groups (encoder 3e-5, head 1e-3) and `BCEWithLogitsLoss` on the kept spans.
- **Both:**
  - a linear warmup over the first 10% of steps, then linear decay
  - gradient clipping at 1.0
  - after every epoch, seqeval span F1 on validation; the checkpoint is saved only when F1 improves (`metrics.json` records the epoch and F1)
- The seed is fixed with `torch.manual_seed(42)`, so initialisation and shuffling repeat exactly.

**Q: Why two learning rates?**
The encoder is already pretrained, so big steps would destroy what it knows. The span head starts from random, so it needs bigger steps to learn in 6 epochs.

**Q: Why warmup?**
Early gradients from the random head are large and noisy. Starting with a tiny learning rate protects BioBERT's pretrained weights.

**Q: Why gradient clipping?**
It caps occasional huge gradients, for example from a batch full of hard negatives, so one bad step can't wreck training.

**Q: Why BCEWithLogitsLoss rather than sigmoid + BCELoss?**
It combines the two in one numerically stable operation, so it never computes `log(0)`.

**Q: Why keep the best epoch rather than the last?**
In our baseline run, epoch 4 fitted the training set more closely but validation F1 dropped from 0.894 to 0.886. That's overfitting, so we keep the peak.

## src/evaluate.py
- Runs both models on the same 750 validation sentences. It tunes the span threshold and writes it to `metrics.json`.
- Reports span F1, precision and recall (seqeval), token accuracy and token macro F1 (the same sklearn calls as `validator.py`), and exact-span F1 by entity length (1, 2, 3, 4+).
- Prints 10 example errors per model:
  - **partial:** overlaps the gold span but has the wrong edges
  - **missed:** no prediction touches the gold span
  - **false positive:** a prediction touches no gold span
- Writes everything to `results.md`.

**Q: Why is token accuracy so high for both models?**
About 92% of tokens are `O`, so predicting all `O` already scores about 92%. That's why span F1 is the primary metric.

**Q: How is "F1 by length" computed?**
Gold spans count toward recall and predicted spans toward precision, each in its own length bucket. Only exact matches are true positives.

## src/predict.py
- Featurizes `test.csv` in order and prints how many sentences were truncated (0 on this data).
- The span model uses the tuned threshold and writes `submission/submission.csv`. `--model baseline` writes `submission_baseline.csv`.
- Asserts that every row has exactly as many tags as tokens, and only valid tags.

**Q: Why assert when the official validator exists?**
The validator silently pads or cuts bad rows. A repaired row is still a wrong prediction, so we'd rather crash loudly on Kaggle.

## backend/app.py
- FastAPI server that loads the tokenizer, both models, the threshold and the validation F1s once, at startup.
- Splits text with the regex `\w+|[^\w\s]`, i.e. words, with each punctuation mark on its own. This reproduces the dataset's tokens on all 6,993 sentences.
- **Endpoints:**
  - `/health`
  - `/predict`: the span model returns entities, top 15 candidates and the full span grid; the baseline returns entities scored by the mean confidence of their words' tags
  - `/compare`: both models, plus the indices where they disagree
  - `/examples`
- Input is validated to 1–2,000 characters. CORS is open because the frontend is a local file calling the tunnel URL.

**Q: The backend runs on Kaggle and uses a tunnel. Is that an external API?**
No. Cloudflare only forwards HTTP to our own server. Every prediction is computed by our models on Kaggle, with no third-party model involved.

**Q: Why regex tokenization?**
The model was trained on the dataset's tokens, so demo input must be split the same way. We checked the regex against every dataset sentence.

## frontend/index.html
- A single plain HTML/JS page with no build step. It connects to the backend URL and loads example chips.
- **Result view:** entities highlighted as continuous blocks, the BIO tag under each word, and the score on hover.
- **Span view:** a start×end heatmap (cream → soft terracotta = P(disease), with the chosen entity ringed in gold and the threshold marked on the legend), plus the top 15 table with the decoder's decision: kept, overlap, or below threshold.
- **Compare view:** both models side by side, with disagreeing words outlined.
- User text is inserted with `textContent` only, so it can't inject HTML.

**Q: How do I read the heatmap?**
Row = first word and column = last word of a phrase. A darker cell means the model is more confident the phrase is a disease. The ringed cell is what the decoder chose; the table explains why the other bright cells were dropped.

## notebooks/ and kaggle_bundle.sh
- `kaggle_bundle.sh` zips `src/`, `backend/` and the CSVs for upload as the `bioner-code` Kaggle dataset.
- `make_notebook.py` generates both notebooks from readable cell lists:
  - `kaggle_train.ipynb` runs every phase top to bottom.
  - `kaggle_demo.ipynb` serves the backend.
- Every `!command` is followed by `assert _exit_code == 0`, so a crashed step stops "Save & Run All" instead of continuing silently.
- The setup cell identifies our code by the `requirements.txt` next to `src/`, and reuses checkpoints from a previous training run if one is attached.

**Q: Why train on Kaggle and not locally?**
The rules allow Kaggle GPUs, and a T4 trains in minutes what a laptop does in hours. Running on Kaggle also keeps the laptop cool, and every result is reproducible from a clean notebook run.

---

## Design decisions we changed from the plan (be upfront)
1. **Biaffine projection:** we project to 128 dimensions before `U`, for stability.
2. **Threshold storage:** the tuned threshold is stored in `metrics.json`, not `config.py`.
3. **Decoder moved earlier:** `decode.py` was written in Phase 4 rather than Phase 5, because training needs it to score validation every epoch.
4. **Backend location:** the backend runs on Kaggle through a tunnel rather than on the laptop.
