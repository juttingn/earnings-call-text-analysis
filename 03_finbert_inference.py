#!/usr/bin/env python3
"""
03_finbert_inference.py

Runs FinBERT sentiment inference at the sentence level on the 250 georisk-flagged
transcripts. For each sentence in exec_text and analyst_text that matches a
geoeconomic risk pattern, a context window of [prev_sentence, sentence,
next_sentence] is scored with FinBERT.

Usage
-----
    source .venv/bin/activate
    python 03_finbert_inference.py           # full run (~5–20 min on MPS/GPU)
    python 03_finbert_inference.py --resume  # resume an interrupted run
    python 03_finbert_inference.py --limit 10  # test on first 10 transcripts

Output
------
    data/finbert_sentiment.csv        — one row per georisk sentence
    data/.finbert_sentiment_ckpt.csv  — rolling checkpoint (deleted on success)

Schema of output CSV
--------------------
    url, speaker_type, sent_idx, sentence_text, context_text,
    trade_risk, sanctions_risk, embargo_risk, geopolitical_risk,
    p_positive, p_negative, p_neutral, sentiment
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
MODEL_NAME       = "ProsusAI/finbert"
FLAGGED_PATH     = Path("data/geoeconomic_matches_flagged.csv")
CORPUS_PATH      = Path("data/corpus_documents.csv")
OUTPUT_PATH      = Path("data/finbert_sentiment.csv")
CHECKPOINT_PATH  = Path("data/.finbert_sentiment_ckpt.csv")
BATCH_SIZE       = 32    # context windows per inference batch
CHECKPOINT_EVERY = 50    # save checkpoint after every N transcripts
LABELS           = ["positive", "negative", "neutral"]

# ── Georisk pattern builder (mirrors 05_geoeconomic_search.ipynb) ──────────────

def make_pattern(*terms):
    pats = []
    for term in terms:
        term = term.strip()
        if term.startswith("RE:"):
            pats.append(term[3:])
            continue
        wildcard = term.endswith("*")
        stem = term[:-1] if wildcard else term
        words = stem.split()
        joined = r"\s+".join(re.escape(w) for w in words)
        pats.append(r"\b" + joined + (r"\w*" if wildcard else r"\b"))
    return re.compile("|".join(pats), re.IGNORECASE)


def hit(text, pat):
    return bool(pat.search(text)) if isinstance(text, str) else False


# Shared risk / uncertainty vocabulary
PAT_RISK = make_pattern(
    "risk*", "threat*", "caution*", "uncertaint*", "propos*", "future",
    "worr*", "concern*", "volatile", "tension*", "likel*", "probab*",
    "possib*", "chance*", "danger*", "fear*", "expect*", "potential",
    "rumor*", "prospect*",
)

# Trade
PAT_TRADE_A = make_pattern(
    "tariff*", "import dut*", "import barrier*", "trade treat*",
    "trade polic*", "trade act*", "trade agreement*", "trade relationship*",
    "GATT", "World Trade Organization", "WTO", "free trade",
    "RE:" + r"\b(?:anti-)?dumping\b",
)
PAT_TRADE_A_NOTARIFF = make_pattern(
    "import dut*", "import barrier*", "trade treat*", "trade polic*",
    "trade act*", "trade agreement*", "trade relationship*",
    "GATT", "World Trade Organization", "WTO", "free trade",
    "RE:" + r"\b(?:anti-)?dumping\b",
)
PAT_TARIFF      = make_pattern("tariff*")
PAT_TARIFF_EXCL = make_pattern(
    "feed-in", "MTA", "network*", "transportation", "adjustment*",
    "regulat*", "escalator",
)
PAT_TRADE_C = make_pattern("import*", "export*", "border*")
PAT_TRADE_D = make_pattern("ban*", "tax*", "subsid*", "control*")

# Sanctions / Embargo
PAT_SANCTIONS = make_pattern("sanctions", "asset freez*")
PAT_EMBARGO   = make_pattern("embargo*")

# Geopolitical
PAT_GEO_WAR      = make_pattern(
    "war", "conflict", "hostilities", "revolution*", "insurrection",
    "uprising", "revolt", "coup", "geopolitical",
)
PAT_GEO_THREAT   = make_pattern(
    "threat*", "warn*", "fear*", "risk*", "concern*", "danger*",
    "doubt*", "crisis", "trouble*", "dispute*", "tension*",
    "imminent*", "inevitable", "footing", "menace*", "brink", "scare", "peril*",
)
PAT_GEO_PEACE    = make_pattern("peace", "truce", "armistice", "treaty", "parley")
PAT_GEO_PEACE_B  = make_pattern(
    "threat*", "menace*", "reject*", "peril*", "boycott*", "disrupt*",
)
PAT_GEO_MILITARY = make_pattern(
    "military", "troops", "missile*", "weapon*", "bomb*", "warhead*",
    "RE:" + r"\barms\b",
)
PAT_GEO_ESCALATE = make_pattern(
    "buildup*", "RE:" + r"\bbuild\-up\w*",
    "sanction*", "blockade*", "embargo", "quarantine", "ultimatum", "mobilize*",
)
PAT_GEO_TERROR   = make_pattern("terror*", "guerrilla*", "hostage*")
PAT_GEO_ACTORS   = make_pattern(
    "allies*", "enemy*", "insurgent*", "foe*", "army", "navy",
    "aerial", "troops", "rebels",
)
PAT_GEO_OFFENSIVE = make_pattern(
    "advance*", "attack*", "strike*", "drive*", "shell*",
    "offensive", "invasion", "invade*", "clash*", "raid*", "launch*",
)
PAT_GEO_OUTBREAK = make_pattern(
    "begin*", "start*", "declar*", "begun", "began",
    "outbreak", "RE:" + r"\bbroke\s+out\b",
    "breakout", "proclamation", "launch*",
)


# ── Sentence-level georisk classifiers ────────────────────────────────────────

def match_trade_sent(s):
    arm1 = hit(s, PAT_TRADE_A) and hit(s, PAT_RISK)
    if arm1:
        tariff_only = hit(s, PAT_TARIFF) and not hit(s, PAT_TRADE_A_NOTARIFF)
        if tariff_only and hit(s, PAT_TARIFF_EXCL):
            arm1 = False
    return arm1 or (hit(s, PAT_TRADE_C) and hit(s, PAT_TRADE_D))


def match_sanctions_sent(s):
    return hit(s, PAT_SANCTIONS) and hit(s, PAT_RISK)


def match_embargo_sent(s):
    return hit(s, PAT_EMBARGO) and hit(s, PAT_RISK)


def match_geopolitical_sent(s):
    return (
        (hit(s, PAT_GEO_WAR)      and hit(s, PAT_GEO_THREAT))   or
        (hit(s, PAT_GEO_PEACE)    and hit(s, PAT_GEO_PEACE_B))  or
        (hit(s, PAT_GEO_MILITARY) and hit(s, PAT_GEO_ESCALATE)) or
        (hit(s, PAT_GEO_TERROR)   and hit(s, PAT_GEO_THREAT))   or
        (hit(s, PAT_GEO_ACTORS)   and hit(s, PAT_GEO_OFFENSIVE)) or
        (hit(s, PAT_GEO_WAR)      and hit(s, PAT_GEO_OUTBREAK))
    )


def georisk_flags(s):
    return {
        "trade_risk":        match_trade_sent(s),
        "sanctions_risk":    match_sanctions_sent(s),
        "embargo_risk":      match_embargo_sent(s),
        "geopolitical_risk": match_geopolitical_sent(s),
    }


def is_georisk(s):
    f = georisk_flags(s)
    return any(f.values())


# ── FinBERT helpers ────────────────────────────────────────────────────────────

def split_sentences(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return re.split(r"(?<=[.!?])\s+", text.strip())


def build_context(sentences, idx):
    """Concatenate prev + current + next sentence as the scoring window."""
    parts = []
    if idx > 0:
        parts.append(sentences[idx - 1])
    parts.append(sentences[idx])
    if idx < len(sentences) - 1:
        parts.append(sentences[idx + 1])
    return " ".join(parts)


def batch_infer(texts, tokenizer, model, device):
    """Run FinBERT on a list of text strings. Returns (n, 3) probability array."""
    if not texts:
        return np.empty((0, 3))
    enc = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=512, padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return softmax(logits, dim=-1).cpu().numpy()


def score_transcript(url, exec_text, analyst_text, tokenizer, model, device):
    """
    Find all georisk sentences in exec_text and analyst_text, score each with
    its ±1 sentence context window, and return a list of result dicts.
    """
    records = []
    for speaker_type, text in [("exec", exec_text), ("analyst", analyst_text)]:
        sentences = split_sentences(text)
        if not sentences:
            continue

        # Collect matching sentences
        matches = []
        for i, sent in enumerate(sentences):
            if is_georisk(sent):
                flags = georisk_flags(sent)
                context = build_context(sentences, i)
                matches.append({
                    "sent_idx":      i,
                    "sentence_text": sent,
                    "context_text":  context,
                    **flags,
                })

        if not matches:
            continue

        # Batch inference over all context windows for this speaker block
        contexts = [m["context_text"] for m in matches]
        all_probs = []
        for i in range(0, len(contexts), BATCH_SIZE):
            probs = batch_infer(contexts[i : i + BATCH_SIZE], tokenizer, model, device)
            all_probs.append(probs)
        probs_arr = np.vstack(all_probs)

        for m, prob in zip(matches, probs_arr):
            records.append({
                "url":          url,
                "speaker_type": speaker_type,
                **m,
                "p_positive": float(prob[0]),
                "p_negative": float(prob[1]),
                "p_neutral":  float(prob[2]),
                "sentiment":  LABELS[int(np.argmax(prob))],
            })

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentence-level FinBERT inference on georisk-flagged transcripts."
    )
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process at most N transcripts (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip transcripts already in the checkpoint file")
    args = parser.parse_args()

    device = ("mps"  if torch.backends.mps.is_available()  else
              "cuda" if torch.cuda.is_available()           else "cpu")
    print(f"Device : {device}")

    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    # Load the 250 flagged transcript URLs
    print(f"Loading flagged transcripts from {FLAGGED_PATH} …")
    flagged = pd.read_csv(FLAGGED_PATH, usecols=["url"])
    flagged_urls = set(flagged["url"].tolist())
    print(f"  {len(flagged_urls):,} flagged transcripts")

    # Load corpus text, filtered to flagged URLs only
    print(f"Loading corpus text from {CORPUS_PATH} …")
    corpus = pd.read_csv(CORPUS_PATH,
                         usecols=["url", "exec_text", "analyst_text"])
    corpus = corpus[corpus["url"].isin(flagged_urls)].reset_index(drop=True)
    print(f"  {len(corpus):,} transcripts matched in corpus")

    if args.limit:
        corpus = corpus.head(args.limit)
        print(f"  Limited to first {args.limit} transcripts.")

    # Resume: skip already-processed URLs
    done_urls: set = set()
    existing: list = []
    if args.resume and CHECKPOINT_PATH.exists():
        ckpt      = pd.read_csv(CHECKPOINT_PATH)
        done_urls = set(ckpt["url"].tolist())
        existing  = ckpt.to_dict("records")
        print(f"Resuming — {len(done_urls):,} transcripts already processed.\n")

    results = list(existing)
    total   = len(corpus)

    for i, (_, row) in enumerate(corpus.iterrows(), start=1):
        url = str(row["url"])
        if args.resume and url in done_urls:
            continue

        recs = score_transcript(
            url,
            str(row.get("exec_text") or ""),
            str(row.get("analyst_text") or ""),
            tokenizer, model, device,
        )
        results.extend(recs)

        if i % 10 == 0 or i == total:
            print(f"  {i:>4}/{total}  ({100*i/total:.1f}%)  "
                  f"sentences so far: {len(results):,}")

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            print(f"  ↳ checkpoint saved ({len(results):,} sentences, {i} transcripts)")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\nDone.")
    print(f"  Transcripts processed : {total:,}")
    print(f"  Georisk sentences found: {len(df):,}")
    print(f"  Output → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
