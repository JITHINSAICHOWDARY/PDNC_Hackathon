"""Which spans the loss sees in each batch.
positives      = gold disease spans                                   -> always kept, label 1
hard negatives = overlap a gold span but have the wrong start or end  -> always kept, label 0
easy negatives = everything else                                      -> random sample, label 0
"""
import torch

import config


def span_targets(gold_spans, W, device):
    """Gold spans -> two (B, W, L) grids: labels (1.0 = gold) and hard (True = hard negative)."""
    L = config.MAX_SPAN_LEN
    starts = torch.arange(W, device=device).unsqueeze(1)             # (W, 1)
    ends = starts + torch.arange(L, device=device)                   # (W, L)
    labels = torch.zeros(len(gold_spans), W, L, device=device)
    overlap = torch.zeros(len(gold_spans), W, L, dtype=torch.bool, device=device)
    for b, spans in enumerate(gold_spans):
        for gs, ge in spans:
            if ge < W and ge - gs < L:          # gold longer than L or cut by truncation: cannot be scored
                labels[b, gs, ge - gs] = 1.0
            overlap[b] |= (starts <= ge) & (ends >= gs)             # the two ranges share a word
    hard = overlap & (labels == 0)                                   # overlaps gold but is not gold
    return labels, hard


def sample_spans(labels, hard, span_mask, ratio=config.EASY_NEG_RATIO):
    """(B, W, L) bool mask of spans to train on in this batch."""
    pos = (labels == 1) & span_mask
    hard = hard & span_mask
    easy = span_mask & ~pos & ~hard
    keep = pos | hard
    n_easy = ratio * max(int(pos.sum()), 1)       # a batch with no diseases still gets `ratio` easy negatives
    easy_idx = easy.flatten().nonzero().squeeze(1)                   # positions of all easy negatives
    chosen = easy_idx[torch.randperm(len(easy_idx), device=easy_idx.device)[:n_easy]]
    keep.view(-1)[chosen] = True
    return keep


if __name__ == "__main__":
    # One 6-word sentence with a gold span (2, 3). Uses no model, so it is safe on a laptop.
    labels, hard = span_targets([[(2, 3)]], W=6, device="cpu")
    mask = torch.ones(1, 6, config.MAX_SPAN_LEN, dtype=torch.bool)
    mask[0] &= (torch.arange(6).unsqueeze(1) + torch.arange(config.MAX_SPAN_LEN)) < 6   # end inside sentence
    as_spans = lambda m: [(s, s + k) for _, s, k in m.nonzero().tolist()]
    assert as_spans(labels == 1) == [(2, 3)]
    hard_spans = as_spans(hard & mask)
    assert (2, 2) in hard_spans and (1, 3) in hard_spans and (3, 5) in hard_spans   # wrong edges
    assert (0, 1) not in hard_spans and (4, 5) not in hard_spans                    # no overlap
    keep = sample_spans(labels, hard, mask, ratio=2)
    n_easy = int((keep & mask & ~hard & (labels == 0)).sum())
    assert keep[0, 2, 1] and bool((keep | ~(hard & mask)).all()) and n_easy == 2
    print("gold:", as_spans(labels == 1), "\nhard negatives:", hard_spans, "\nsampling check passed")
