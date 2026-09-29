"""All hyperparameters and paths in one place. Every other file imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the bioner/ folder
TRAIN_CSV = ROOT / "data" / "train" / "train.csv"
TEST_CSV = ROOT / "data" / "test" / "test.csv"
CKPT_DIR = ROOT / "checkpoints"
SUBMISSION_CSV = ROOT / "submission" / "submission.csv"

# Labels: index order matters, the baseline's output layer uses these ids
LABELS = ["O", "B-Disease", "I-Disease"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# Encoder
MODEL_NAME = "dmis-lab/biobert-v1.1"   # alternative: "michiyasunaga/BioLinkBERT-base"
MAX_LENGTH = 256                       # subwords; the longest sentence is 123 words

# Data split
VAL_FRACTION = 0.15
SEED = 42

# Baseline (BIO token classifier)
BASE_LR = 3e-5
BASE_EPOCHS = 4

# Span model
MAX_SPAN_LEN = 10          # covers 99.7% of gold entities
WIDTH_EMB_DIM = 32         # size of the learned span-length embedding
BIAFFINE_DIM = 128         # start/end vectors are projected 768 -> 128 before the biaffine product
FFN_HIDDEN = 256
DROPOUT = 0.2
ENCODER_LR = 3e-5
HEAD_LR = 1e-3             # the new span head learns faster than the pretrained encoder
SPAN_EPOCHS = 6
EASY_NEG_RATIO = 10        # at most this many easy negatives per positive, per batch
THRESHOLD = 0.5            # overwritten after tuning on validation (Phase 5)

# Shared training settings
BATCH_SIZE = 16
WARMUP_FRACTION = 0.10
GRAD_CLIP = 1.0
