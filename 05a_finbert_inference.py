#!/usr/bin/env python3
"""
05a_finbert_inference.py

Runs FinBERT sentiment inference on the earnings call corpus and writes
per-document scores to data/finbert_sentiment.csv.

Separating inference from visualisation means the heavy computation can
be run once (or resumed after interruption) while the notebook
05_finbert_sentiment.ipynb just loads the cached CSV.

Usage
-----
    source /path/to/.venv/bin/activate

    # Full corpus (~1–2 h on CPU; seconds on GPU)
    python 05a_finbert_inference.py

    # Test on first 100 documents
    python 05a_finbert_inference.py --limit 100

    # Resume an interrupted run
    python 05a_finbert_inference.py --resume

    # Only score the full transcript (skip exec/analyst, 3× faster)
    python 05a_finbert_inference.py --fields full_text

Output
------
    data/finbert_sentiment.csv        — final results, one row per document
    data/.finbert_sentiment_ckpt.csv  — rolling checkpoint (deleted on success)
"""

import argparse
import re
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME      = "ProsusAI/finbert"
OUTPUT_PATH     = Path("data/finbert_sentiment.csv")
CHECKPOINT_PATH = Path("data/.finbert_sentiment_ckpt.csv")
MAX_CHUNK_TOKENS = 400   # BERT hard limit is 512; leave headroom for specials
BATCH_SIZE       = 32    # chunks per inference batch (increase if you have GPU RAM)
CHECKPOINT_EVERY = 200   # save checkpoint after every N documents
LABELS           = ["positive", "negative", "neutral"]
ALL_FIELDS       = ["full_text", "exec_text", "analyst_text"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def chunk_text(text: str, tokenizer, max_tokens: int = MAX_CHUNK_TOKENS) -> list:
    """Split text into sentence-aligned chunks of at most max_tokens tokens."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, current_len = [], [], 0
    for s in sentences:
        n = len(tokenizer.tokenize(s))
        if current_len + n > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(s)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def batch_infer(texts: list, tokenizer, model, device: str) -> np.ndarray:
    """Run FinBERT on a list of text strings. Returns (n, 3) probability array."""
    if not texts:
        return np.empty((0, 3))
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return softmax(logits, dim=-1).cpu().numpy()


def score_doc(text: str, tokenizer, model, device: str) -> dict:
    """Return averaged FinBERT scores for a single document."""
    chunks = chunk_text(text, tokenizer)
    if not chunks:
        return {"p_positive": np.nan, "p_negative": np.nan,
                "p_neutral": np.nan, "sentiment": None, "n_chunks": 0}
    all_probs = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_probs = batch_infer(chunks[i : i + BATCH_SIZE], tokenizer, model, device)
        all_probs.append(batch_probs)
    mean_probs = np.vstack(all_probs).mean(axis=0)
    return {
        "p_positive": float(mean_probs[0]),
        "p_negative": float(mean_probs[1]),
        "p_neutral":  float(mean_probs[2]),
        "sentiment":  LABELS[int(np.argmax(mean_probs))],
        "n_chunks":   len(chunks),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FinBERT batch inference on earnings call transcripts."
    )
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process at most N documents (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip documents already in the checkpoint file")
    parser.add_argument("--fields", default=",".join(ALL_FIELDS),
                        help=f"Comma-separated text fields to score "
                             f"(default: {','.join(ALL_FIELDS)})")
    args = parser.parse_args()
    fields = [f.strip() for f in args.fields.split(",")]

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    print("Loading corpus …")
    corpus = pd.read_csv("data/corpus_documents.csv",
                         usecols=["url"] + [f for f in fields
                                            if f in ["full_text","exec_text","analyst_text"]])
    if args.limit:
        corpus = corpus.head(args.limit)
        print(f"Limited to first {args.limit} documents.")

    total = len(corpus)
    print(f"Documents to process: {total:,}\n")

    # Load checkpoint
    done_urls: set = set()
    existing: list = []
    if args.resume and CHECKPOINT_PATH.exists():
        ckpt = pd.read_csv(CHECKPOINT_PATH)
        done_urls = set(ckpt["url"].tolist())
        existing  = ckpt.to_dict("records")
        print(f"Resuming — {len(done_urls):,} documents already processed.\n")

    results = list(existing)

    for i, (_, row) in enumerate(corpus.iterrows(), start=1):
        url = str(row.get("url", ""))
        if args.resume and url in done_urls:
            continue

        rec = {"url": url}
        for field in fields:
            text   = str(row.get(field) or "")
            scores = score_doc(text, tokenizer, model, device)
            for k, v in scores.items():
                rec[f"{field}__{k}"] = v

        results.append(rec)

        if i % 10 == 0:
            pct = 100 * i / total
            print(f"  {i:>5}/{total}  ({pct:.1f}%)")

        if len(results) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            print(f"  ↳ checkpoint saved ({len(results):,} docs)")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\nDone. {len(df):,} records → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
