"""
Submission Validator & Evaluator for Problem 02: Biomedical Named Entity Recognition (BioNER)
- Automatically checks submission format, token counts, and valid BIO tags.
- Validates and enforces exact ID row ordering matching the official test set.
- Automatically saves the verified & ordered file into the `submission/` folder.
- If run by organizers (with evaluation key), computes entity span F1 and records in results.csv.

Usage:
    # Student verification:
    python utils/validator.py
    python utils/validator.py --file submission/my_submission.csv

    # Organizer evaluation:
    python utils/validator.py --file submission/my_submission.csv --eval
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
EVAL_KEY_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "organizer_evaluation_keys"))
DEFAULT_EVAL_KEY = os.path.join(EVAL_KEY_DIR, "02_bioner_solution.csv")

VALID_TAGS = ['O', 'B-Disease', 'I-Disease']

def parse_tags(val):
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    val = str(val).strip()
    if val.startswith("[") and val.endswith("]"):
        try:
            return [str(x).strip() for x in json.loads(val)]
        except Exception:
            pass
    return val.split()

def get_entities(tags):
    entities = []
    chunk_type = None
    chunk_start = None
    
    for i, tag in enumerate(tags):
        tag = tag.strip()
        if tag.startswith("B-"):
            if chunk_type is not None:
                entities.append((chunk_type, chunk_start, i - 1))
            chunk_type = tag[2:]
            chunk_start = i
        elif tag.startswith("I-"):
            current_type = tag[2:]
            if chunk_type == current_type:
                continue
            else:
                if chunk_type is not None:
                    entities.append((chunk_type, chunk_start, i - 1))
                chunk_type = current_type
                chunk_start = i
        else: # O
            if chunk_type is not None:
                entities.append((chunk_type, chunk_start, i - 1))
                chunk_type = None
                chunk_start = None
                
    if chunk_type is not None:
        entities.append((chunk_type, chunk_start, len(tags) - 1))
        
    return set(entities)

def find_candidate_file(explicit_file=None):
    if explicit_file:
        if os.path.exists(explicit_file):
            return explicit_file
        candidate = os.path.join(SUBMISSION_DIR, explicit_file)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Specified submission file not found: {explicit_file}")
    
    if os.path.exists(SUBMISSION_DIR):
        csv_files = [os.path.join(SUBMISSION_DIR, f) for f in os.listdir(SUBMISSION_DIR) if f.endswith('.csv') and f != 'submission_verified.csv']
        if csv_files:
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return csv_files[0]

    sample = os.path.join(DATASET_DIR, "test", "sample_submission.csv")
    if os.path.exists(sample):
        print("[INFO] No file found in submission/ folder. Running verification using dataset/test/sample_submission.csv.")
        return sample
        
    raise FileNotFoundError("No candidate submission CSV found! Please place your file in submission/submission.csv")

def validate_submission(file_path, do_eval=False, solution_key=None):
    print("="*65)
    print("BioNER SUBMISSION VALIDATOR & ORDER VERIFIER")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Submission: {file_path}")
    print("="*65)

    ref_test_path = os.path.join(DATASET_DIR, "test", "test.csv")
    if not os.path.exists(ref_test_path):
        raise FileNotFoundError(f"Reference test file missing: {ref_test_path}")

    ref_test_df = pd.read_csv(ref_test_path)
    expected_ids = ref_test_df['id'].tolist()
    expected_count = len(expected_ids)

    expected_token_counts = {}
    for _, row in ref_test_df.iterrows():
        toks = parse_tags(row['tokens']) if 'tokens' in row else parse_tags(row.get('ner_tags', ''))
        expected_token_counts[row['id']] = len(toks)

    try:
        sub_df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return False

    sub_df.columns = [c.strip().lower() for c in sub_df.columns]
    if 'id' not in sub_df.columns or 'ner_tags' not in sub_df.columns:
        print(f"[ERROR] Missing required columns! Expected ['id', 'ner_tags'], found {sub_df.columns.tolist()}")
        return False

    submitted_ids = sub_df['id'].tolist()
    is_ordered = (submitted_ids == expected_ids)
    
    if not is_ordered:
        print("[NOTICE] Submission IDs are out of order or mismatched. Re-aligning to official test sequence...")
        sub_indexed = sub_df.drop_duplicates(subset=['id']).set_index('id')
        missing_ids = set(expected_ids) - set(sub_indexed.index)
        if missing_ids:
            print(f"[WARNING] Submission is missing {len(missing_ids)} expected IDs! Missing sentences will default to all 'O'.")
            
        sub_ordered = sub_indexed.reindex(expected_ids).reset_index()
        sub_ordered.rename(columns={'index': 'id'}, inplace=True)
        sub_df = sub_ordered
        order_corrected = True
        print("[SUCCESS] Successfully re-ordered all rows to match official test sequence.")
    else:
        order_corrected = False
        print("[SUCCESS] Submission row count and ID ordering match test set exactly.")

    # Validate and Align Token Counts
    cleaned_tags = []
    token_mismatch_count = 0
    for _, row in sub_df.iterrows():
        sid = row['id']
        expected_len = expected_token_counts.get(sid, 0)
        raw_val = row['ner_tags']
        tags = parse_tags(raw_val) if pd.notnull(raw_val) else []
        
        normalized = []
        for t in tags:
            t_clean = t.strip()
            if t_clean in VALID_TAGS:
                normalized.append(t_clean)
            elif t_clean.upper() in ['B', 'B-DISEASE']:
                normalized.append('B-Disease')
            elif t_clean.upper() in ['I', 'I-DISEASE']:
                normalized.append('I-Disease')
            else:
                normalized.append('O')
                
        if len(normalized) != expected_len:
            token_mismatch_count += 1
            if len(normalized) < expected_len:
                normalized = normalized + ['O'] * (expected_len - len(normalized))
            else:
                normalized = normalized[:expected_len]
                
        cleaned_tags.append(" ".join(normalized))

    if token_mismatch_count > 0:
        print(f"[WARNING] Adjusted {token_mismatch_count} sentences where token tag counts differed from test tokens.")
    sub_df['ner_tags'] = cleaned_tags

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    verified_path = os.path.join(SUBMISSION_DIR, "submission_verified.csv")
    sub_df[['id', 'ner_tags']].to_csv(verified_path, index=False)
    print(f"[SAVED] Verified & ordered submission saved to: {verified_path}")
    print(f"[STATUS] Passed validation! Ready for evaluation ({len(sub_df)} sentences).")

    key_path = solution_key if solution_key else DEFAULT_EVAL_KEY
    if do_eval or (key_path and os.path.exists(key_path)):
        if os.path.exists(key_path):
            print("\n" + "-"*40)
            print("RUNNING BENCHMARK EVALUATION (ORGANIZER MODE):")
            gt_df = pd.read_csv(key_path)
            merged = pd.merge(gt_df, sub_df, on='id', suffixes=('_gt', '_pred'))
            gt_col = 'ner_tags_gt' if 'ner_tags_gt' in merged.columns else 'ner_tags'
            pred_col = 'ner_tags_pred' if 'ner_tags_pred' in merged.columns else 'ner_tags'

            all_true_tokens = []
            all_pred_tokens = []
            true_entities_total = 0
            pred_entities_total = 0
            matched_entities_total = 0

            for _, row in merged.iterrows():
                t_tags = parse_tags(row[gt_col])
                p_tags = parse_tags(row[pred_col])
                
                if len(p_tags) < len(t_tags):
                    p_tags = p_tags + ['O'] * (len(t_tags) - len(p_tags))
                elif len(p_tags) > len(t_tags):
                    p_tags = p_tags[:len(t_tags)]
                    
                all_true_tokens.extend(t_tags)
                all_pred_tokens.extend(p_tags)
                
                true_ents = get_entities(t_tags)
                pred_ents = get_entities(p_tags)
                
                true_entities_total += len(true_ents)
                pred_entities_total += len(pred_ents)
                matched_entities_total += len(true_ents.intersection(pred_ents))

            span_prec = matched_entities_total / pred_entities_total if pred_entities_total > 0 else 0.0
            span_rec = matched_entities_total / true_entities_total if true_entities_total > 0 else 0.0
            span_f1 = (2 * span_prec * span_rec / (span_prec + span_rec)) if (span_prec + span_rec) > 0 else 0.0

            token_acc = accuracy_score(all_true_tokens, all_pred_tokens)
            token_macro_f1 = f1_score(all_true_tokens, all_pred_tokens, average='macro', zero_division=0)

            print(f"  Matched Sentences:    {len(merged)} / {expected_count}")
            print(f"  True Spans:           {true_entities_total}")
            print(f"  Predicted Spans:      {pred_entities_total}")
            print(f"  Correct Spans:        {matched_entities_total}")
            print(f"  Span Precision:       {span_prec * 100:.2f}%")
            print(f"  Span Recall:          {span_rec * 100:.2f}%")
            print(f"  Span F1 (Primary):    {span_f1 * 100:.2f}%")
            print(f"  Token Accuracy:       {token_acc * 100:.2f}%")
            print(f"  Token Macro F1:       {token_macro_f1 * 100:.2f}%")
            print("-"*40)

            file_exists = os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = pd.DataFrame([{
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'submission_file': os.path.basename(file_path),
                    'matched_sentences': len(merged),
                    'total_sentences': expected_count,
                    'order_corrected': order_corrected,
                    'span_f1': round(span_f1, 4),
                    'span_precision': round(span_prec, 4),
                    'span_recall': round(span_rec, 4),
                    'token_accuracy': round(token_acc, 4),
                    'status': 'PASSED'
                }])
                writer.to_csv(f, header=not file_exists, index=False)
            print(f"[LOGGED] Evaluation score logged to: {RESULTS_CSV}")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate & Order BioNER Submissions")
    parser.add_argument('--file', type=str, default=None, help='Path to submission CSV')
    parser.add_argument('--eval', action='store_true', help='Evaluate if solution key is available')
    parser.add_argument('--solution_key', type=str, default=None, help='Path to solution key CSV')
    args = parser.parse_args()

    try:
        target = find_candidate_file(args.file)
        validate_submission(target, do_eval=args.eval, solution_key=args.solution_key)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
