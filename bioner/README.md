# Span-Based Biomedical NER (Team Lannisters)

Detect disease mentions in PubMed sentences and output BIO tags (`O`, `B-Disease`, `I-Disease`).

**Our idea.** A BIO tagger labels every word separately, so a single wrong tag breaks the whole entity. We **score whole phrases instead**:

1. BioBERT encodes the sentence.
2. Every candidate span of up to 10 words gets a probability of being a disease. The scorer is a biaffine start×end term plus a feed-forward head over `[h_start; h_end; mean; width]`.
3. A greedy decoder keeps the best non-overlapping spans and turns them into BIO tags.

The output is always a valid BIO sequence, and the loss is computed on whole spans, so the model is trained directly on boundaries.

We also train a standard BIO token classifier on the same encoder as a **baseline**.

## Results (validation split: 750 sentences, seed 42)

| metric | baseline (BIO) | span model |
|---|---|---|
| span F1 (primary) | 0.8940 * | *pending Kaggle run* |
| span precision | | |
| span recall | | |
| token accuracy | | |
| token macro F1 | | |
| F1 by length 1 / 2 / 3 / 4+ | | |

\* The baseline F1 of 0.8940 comes from an early full run on a laptop (best epoch 3 of 4). Replace this table with the one `src/evaluate.py` writes to `results.md` after the Kaggle run. That file also lists 10 example errors per model.

## Rules we follow
- **No external inference APIs.** Every prediction comes from our own models, trained and served on a Kaggle GPU.
- **Open models only.** The encoder is [`dmis-lab/biobert-v1.1`](https://huggingface.co/dmis-lab/biobert-v1.1).
- **Submission format.** `id,ner_tags`, with exactly one tag per token, in `test.csv` order. `predict.py` asserts the tag count on every row, and `utils/validator.py` checks the file again.

## Project layout
```
bioner/
├── data -> ../02_Biomedical_NER/dataset   symlink to the official data
├── src/
│   ├── config.py        all hyperparameters and paths
│   ├── explore.py       dataset statistics (standard library only)
│   ├── data.py          CSV parsing, subword alignment, BIO <-> spans, span enumeration, batching
│   ├── test_data.py     sanity tests for data.py
│   ├── baseline.py      BIO token classifier
│   ├── span_model.py    encoder -> word vectors -> span grid -> biaffine + FFN scores
│   ├── sampling.py      positives, hard negatives, sampled easy negatives
│   ├── decode.py        threshold + greedy non-overlap -> BIO; threshold tuning
│   ├── train.py         training loop for either model (--model span|baseline, --debug)
│   ├── evaluate.py      baseline vs span metrics, F1 by length, example errors -> results.md
│   └── predict.py       test set -> submission/submission.csv
├── backend/app.py       FastAPI server (runs on Kaggle, reached through a tunnel)
├── frontend/            demo UI: index.html (plain HTML/JS) + lion.svg (CC0 heraldic lion)
├── notebooks/
│   ├── make_notebook.py      generates both notebooks
│   ├── kaggle_train.ipynb    tests, training, evaluation, submission
│   └── kaggle_demo.ipynb     serves the backend for the demo
├── kaggle_bundle.sh     zips code + data for upload as a Kaggle dataset
├── checkpoints/         trained models (git-ignored), downloaded from Kaggle
└── submission/
```

## Setup (laptop)
The laptop only writes code, opens the frontend and runs the validator. **No model is loaded locally.**
```bash
cd bioner
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/explore.py            # dataset statistics, no model needed
```

## Train on Kaggle
1. Bundle the code: `sh kaggle_bundle.sh`. This writes `kaggle_bundle.zip`, about 4 MB, containing `src/`, `backend/` and the CSVs.
2. On Kaggle, go to **Datasets → New Dataset**, upload the zip, and name it `bioner-code`. After any code change, upload a **New Version**.
3. Go to **Code → New Notebook → File → Import Notebook** and pick `notebooks/kaggle_train.ipynb`.
4. In the right panel, set **Accelerator** to **GPU T4 x2** and **Internet** to **On**, and use **Add Input** to add `bioner-code`.
5. Click **Save Version → Save & Run All**. The run takes about 30–40 minutes and does the following:
   - data tests
   - baseline: debug run, then 4 epochs
   - span model: shape check, debug run, then 6 epochs
   - `evaluate.py`, which tunes the threshold and compares both models
   - `predict.py` for both models
   - zips the results

   Every shell cell checks its exit code, so a failure stops the run.

Run a single step interactively with `python src/train.py --model span --debug`, which uses 50 sentences and 1 epoch.

## Download checkpoints and validate the submission
1. Open the finished version, go to **Output**, and download `checkpoints.zip`, about 800 MB.
2. Unpack it and validate:
   ```bash
   cd ~/Desktop/PDNC && unzip -o ~/Downloads/checkpoints.zip
   #   -> bioner/checkpoints/{baseline,span}/, bioner/results.md, bioner/submission/
   cp bioner/submission/submission.csv 02_Biomedical_NER/submission/submission.csv
   cd 02_Biomedical_NER && ../bioner/.venv/bin/python utils/validator.py
   ```

Expect `[SUCCESS] ... ordering match` and **no** `Adjusted N sentences` warning. The validator checks the newest CSV in that folder, so keep only the file you want to submit there.

## Run the demo
**Backend, on Kaggle**
1. Import `notebooks/kaggle_demo.ipynb`.
2. Add two inputs: `bioner-code`, and the output of your finished `kaggle_train` run (under **Your Work**).
3. Set GPU and Internet On, and run the cells **interactively**.
4. The notebook starts FastAPI, tests every endpoint, opens a Cloudflare quick tunnel, and prints `BACKEND URL: https://….trycloudflare.com`.
5. Keep the tab open during the demo. A new session gives a new URL.

**Frontend, on the laptop**
1. Run `open frontend/index.html`, paste the URL, and click **Send raven**, which connects to the backend.
2. The page shows:
   - **Span model:** highlighted entities, the BIO tag under each word, a start×end heatmap, and the top 15 candidate spans with the decoder's decision for each.
   - **Baseline:** the same sentence view.
   - **Compare:** both models side by side, with disagreements outlined.

| Endpoint | Purpose |
|---|---|
| `GET /health` | device, model name, validation F1 of both models, threshold |
| `POST /predict {text, model}` | tokens, tags, entities with scores; the span model also returns the top 15 candidates and the span grid |
| `POST /compare {text}` | both models, plus the word indices where they disagree |
| `GET /examples` | 10 test sentences in which our model found a disease |

## Key hyperparameters (`src/config.py`)
- **Encoder:** BioBERT v1.1, `max_length` 256 subwords (the longest sentence is 146).
- **Span model:**
  - `MAX_SPAN_LEN = 10`, which covers 99.7% of gold entities
  - width embedding 32, biaffine projection 128, FFN hidden 256, dropout 0.2
  - learning rate 3e-5 for the encoder and 1e-3 for the head, 6 epochs, 10 easy negatives per positive
- **Baseline:** learning rate 3e-5, 4 epochs.
- **Both:** batch size 16, 10% linear warmup, gradient clip 1.0, seed 42, 15% validation split.
- **Threshold:** tuned on validation over 0.30–0.70 in steps of 0.05, and stored in `checkpoints/span/metrics.json`.

## Credits
- Heraldic lion rampant (`frontend/lion.svg`): "Heraldic Lion Rampant - Or" by Rode raaf, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Heraldic_Lion_Rampant_-_Or.svg), CC0 1.0 (public domain), recoloured. All other artwork (castle, compass rose, banners, seal, swords) is original SVG. No HBO or *Game of Thrones* artwork is used.
