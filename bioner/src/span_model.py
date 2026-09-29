"""Span model: BioBERT encoder -> word vectors -> a score for every phrase (start, end).

Shape legend:  B = batch size, T = subwords, W = words, H = 768 (BioBERT hidden size),
               L = MAX_SPAN_LEN = 10.
Spans are stored as a grid [start word s, width k], where the span is (s, s + k).
Grid cell [s, k] is the phrase from word s to word s + k, so the grid maps
directly onto the start x end heatmap in the demo.
"""
import torch
from torch import nn
from transformers import AutoModel

import config


class SpanModel(nn.Module):
    def __init__(self, encoder_name=config.MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        H, L = self.encoder.config.hidden_size, config.MAX_SPAN_LEN
        self.dropout = nn.Dropout(config.DROPOUT)
        self.width_emb = nn.Embedding(L, config.WIDTH_EMB_DIM)   # one learned vector per width 0..L-1

        # FFN head on [h_start; h_end; mean; width]  ->  1 score
        self.ffn = nn.Sequential(
            nn.Linear(3 * H + config.WIDTH_EMB_DIM, config.FFN_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FFN_HIDDEN, 1),
        )

        # Biaffine: project start and end words to a smaller size, then h_s^T U h_e.
        # U starts at zero, so at first the score comes from the FFN only.
        D = config.BIAFFINE_DIM
        self.start_proj = nn.Linear(H, D)
        self.end_proj = nn.Linear(H, D)
        self.U = nn.Parameter(torch.zeros(D, D))

        # widths 0..L-1; a buffer so it moves to the GPU together with the model
        self.register_buffer("widths", torch.arange(L), persistent=False)

    def forward(self, input_ids, attention_mask, word_index, word_mask):
        # 1. Encode: a context vector for every subword
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state   # (B, T, H)
        B, _, H = hidden.shape
        W, L = word_index.shape[1], len(self.widths)

        # 2. Word vectors: pick each word's first subword
        idx = word_index.unsqueeze(-1).expand(-1, -1, H)                         # (B, W, H)
        words = self.dropout(torch.gather(hidden, 1, idx))                       # (B, W, H)

        # 3. All spans as a grid: start s = row, end = s + k
        starts = torch.arange(W, device=words.device).unsqueeze(1)               # (W, 1)
        ends = starts + self.widths                                              # (W, L)
        span_mask = (ends < W) & word_mask.unsqueeze(-1)                         # (B, W, L) start is real, end inside sentence
        ends = ends.clamp(max=W - 1)                                             # keep indices legal; masked anyway
        span_mask &= torch.gather(word_mask, 1, ends.flatten().expand(B, -1)).view(B, W, L)  # end is a real word

        # 4. Span representation [h_start; h_end; mean; width_emb]
        h_start = words.unsqueeze(2).expand(-1, -1, L, -1)                       # (B, W, L, H)
        h_end = words[:, ends]                                                   # (B, W, L, H)
        # Mean via prefix sums: sum(s..e) = cs[e+1] - cs[s]; one subtraction per span, no loop
        cs = torch.cat([words.new_zeros(B, 1, H), words.cumsum(1)], dim=1)       # (B, W+1, H)
        span_sum = cs[:, ends + 1] - cs[:, starts]           # (B, W, L, H) - (B, W, 1, H) broadcasts
        mean = span_sum / (self.widths + 1).view(1, 1, L, 1)                     # (B, W, L, H)
        width = self.width_emb(self.widths).expand(B, W, L, -1)                  # (B, W, L, 32)
        span_vec = torch.cat([h_start, h_end, mean, width], dim=-1)              # (B, W, L, 3H+32)

        # 5. Scores: FFN + biaffine
        ffn_score = self.ffn(span_vec).squeeze(-1)                               # (B, W, L)
        s = self.start_proj(words)                                               # (B, W, D)
        e = self.end_proj(words)[:, ends]                                        # (B, W, L, D)
        biaffine = torch.einsum("bwd,de,bwle->bwl", s, self.U, e)               # (B, W, L)

        logits = ffn_score + biaffine                                            # (B, W, L)
        return logits, span_mask       # sigmoid(logits) = P(disease); only cells where mask is True are real spans


def load_span(ckpt_dir, device):
    """Rebuild the architecture, then load our trained weights over it."""
    model = SpanModel()
    model.load_state_dict(torch.load(ckpt_dir / "span_model.pt", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model, items, device):
    """Featurized sentences -> [(probs, mask)] per sentence, each (W_i, L) on the CPU.
    Kept separate from decoding so Phase 5 can try many thresholds without re-running BioBERT."""
    from data import collate
    model.eval()
    out = []
    for i in range(0, len(items), config.BATCH_SIZE):
        chunk = items[i:i + config.BATCH_SIZE]
        batch = {k: v.to(device) for k, v in collate(chunk).items() if k != "labels"}
        logits, mask = model(**batch)
        probs = torch.sigmoid(logits).cpu()
        for b, item in enumerate(chunk):
            n = len(item["word_index"])                   # drop this sentence's padding rows
            out.append((probs[b, :n], mask[b, :n].cpu()))
    return out


if __name__ == "__main__":
    # Shape check on two sentences: one forward pass, no training (safe on a laptop)
    from transformers import AutoTokenizer
    from data import collate, enumerate_spans, featurize, load_csv

    df = load_csv(config.TRAIN_CSV)[:2]
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    batch = collate([featurize(tok, t, g) for t, g in zip(df["tokens"], df["ner_tags"])])
    model = SpanModel().eval()
    with torch.no_grad():
        logits, mask = model(batch["input_ids"], batch["attention_mask"],
                             batch["word_index"], batch["word_mask"])
    n_words = [len(t) for t in df["tokens"]]
    print("input_ids", tuple(batch["input_ids"].shape), " word_index", tuple(batch["word_index"].shape))
    print("logits", tuple(logits.shape), " mask", tuple(mask.shape))
    for b, n in enumerate(n_words):
        # valid grid cells must be exactly the spans enumerate_spans lists
        cells = [(s, s + k) for s, k in mask[b].nonzero().tolist()]
        assert cells == enumerate_spans(n), b
        print(f"sentence {b}: {n} words -> {len(cells)} candidate spans")
    print("shape check passed")
