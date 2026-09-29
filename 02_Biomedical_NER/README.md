# Biomedical Named Entity Recognition (BioNER)

> [!NOTE]
> ### AI Assistant Policy & Strict Rules
> - **AI Coding Assistance IS Allowed:** You may use AI coding assistants (ChatGPT, Claude, Cursor, Copilot) as pair programmers to help you write code, learn PyTorch/Transformers syntax, and debug errors.
> - **No Single-Prompt Code Dumps:** Work with your AI co-pilot step-by-step through the roadmap sections in `starter_notebook.ipynb` rather than asking for a monolithic solution.
> - **Mandatory Pitch & Defense:** Teams must present a 2-minute architectural pitch to faculty judges and explain their code. You must thoroughly understand every line you submit!
> - **NO EXTERNAL APIS FOR PREDICTIONS:** Using proprietary inference APIs (OpenAI, Anthropic, Gemini, Groq, etc.) to generate test predictions is **strictly prohibited**. All models must run locally or on Kaggle GPU.

## Description

Extracting clinical concepts, disease phenotypes, and medical conditions from unstructured scientific publications is a cornerstone of biomedical informatics, drug repurposing, and automated knowledge graph construction. Millions of PubMed abstracts exist, but unlocking actionable medical insights requires automating the identification of complex biomedical nomenclature.

### The Objective
Build a token-level sequence labeling model to detect and delineate disease mentions within peer-reviewed biomedical literature using standard **IOB / BIO sequence tagging**:
- `O`: Outside of any biomedical entity mention.
- `B-Disease`: Beginning token of a disease or phenotype entity mention.
- `I-Disease`: Continuation/interior token of a multi-token disease entity mention.

### Evaluation Metrics
Submissions are evaluated on:
- **Primary Metric:** **Entity-Level Span F1 Score (CoNLL / BioBERT Standard)**. An entity is counted as correct *if and only if* both its boundary span (start token to end token) and entity label match the ground truth exactly.
- **Secondary Metrics:** Token-Level Macro F1 and Token-Level Accuracy.

---

## Dataset & Directory Structure

```text
02_Biomedical_NER/
├── README.md
├── starter_notebook.ipynb               # Markdown guide detailing what to implement
├── dataset/
│   ├── train/
│   │   ├── train.csv                    # 5,000 sentences with tokens & JSON ner_tags
│   │   └── train.tsv                    # 5,000 sentences in standard CoNLL format (token \t tag)
│   └── test/
│       ├── test.csv                     # 1,993 unique test sentences WITHOUT ner_tags
│       ├── test.tsv                     # 1,993 sentences in CoNLL format (token only)
│       └── sample_submission.csv        # 1,993 rows illustrating submission schema
├── submission/
│   └── README.md                        # Drop your team submission CSV here
└── utils/
    └── validator.py                     # Automatic order-checking, token-aligning & submission tool
```

### Data Schema & Field Definitions (CSV)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | String | Unique sentence identifier (`BIONER_TR_00001` for train, `BIONER_TEST_0001` for test). |
| `text` | String | Reconstructed whitespace-delimited sentence. |
| `tokens` | String (JSON Array) | List of raw whitespace/punctuation tokens as strings. |
| `ner_tags` | String (JSON Array) | Corresponding IOB tags for each token (present only in `train.csv`). |

---

## Running on Kaggle & Submission Instructions

### Step 1: Upload Starter Notebook to Kaggle
1. Open [Kaggle](https://www.kaggle.com/) and upload [`starter_notebook.ipynb`](starter_notebook.ipynb).
2. Under **Notebook Options** in the right-hand panel, set **Accelerator** to **GPU T4 x2**.

### Step 2: Upload Dataset
- Click **Add Input** $\rightarrow$ **Upload Dataset** and upload `dataset/train/train.csv` (or `train.tsv`) and `dataset/test/test.csv` (or `test.tsv`).

### Step 3: Implement, Train & Export Predictions
- Follow the guided markdown sections in `starter_notebook.ipynb` to construct your sequence labeling pipeline.
- Export test predictions as space-delimited IOB tags:
  ```python
  sub = pd.DataFrame({
      'id': test_df['id'],
      'ner_tags': [" ".join(pred_tags) for pred_tags in all_sentence_preds]
  })
  sub.to_csv('submission.csv', index=False)
  ```

### Step 4: Validate Order Locally
1. Drop your downloaded file into `submission/submission.csv`.
2. Run the automated validator:
   ```bash
   python utils/validator.py
   ```
   `validator.py` checks token count alignment per sentence, ensures valid BIO tags, re-indexes row ordering if needed, and saves `submission/submission_verified.csv`.

---

## Tips

- **You should try** using domain-specific pretrained weights like `dmis-lab/biobert-v1.1` or `michiyasunaga/BioLinkBERT-base`.
- **You should try** tokenizing with `is_split_into_words=True` and labeling only the first subword token of each word while masking subsequent subwords with `-100`.
- **You should try** re-aligning subword predictions back to original word tokens during inference using `encoding.word_ids()`.
- **You should try** adding a CRF (Conditional Random Field) layer or Viterbi decoding to prevent illegal transitions like `O -> I-Disease`.
- **You should try** evaluating your model during training using `seqeval` to monitor entity-level span F1 rather than token accuracy.
