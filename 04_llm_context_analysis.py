#!/usr/bin/env python3
"""
04_llm_context_analysis.py

For each transcript flagged as containing geoeconomic risk language,
uses an LLM via OpenRouter to classify the context in which that risk is
discussed. Results are written to data/geoeconomic_context.json.

Usage
-----
    source /path/to/.venv/bin/activate
    export OPENROUTER_API_KEY="sk-or-v1-..."

    # Full run (1,626 documents)
    python 04_llm_context_analysis.py

    # Test on first 5 documents
    python 04_llm_context_analysis.py --limit 5

    # Resume an interrupted run
    python 04_llm_context_analysis.py --resume

    # Use a different OpenRouter model
    python 04_llm_context_analysis.py --model meta-llama/llama-3.3-70b-instruct:free

Output
------
    data/geoeconomic_context.json        — final results, one object per doc
    data/.geoeconomic_context_ckpt.json  — rolling checkpoint (auto-deleted on
                                           clean completion)
"""

import os
import re
import json
import time
import argparse
import pandas as pd
from openai import OpenAI
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "google/gemma-3-27b-it:free"
OPENROUTER_BASE   = "https://openrouter.ai/api/v1"
OUTPUT_PATH       = Path("data/geoeconomic_context.json")
CHECKPOINT_PATH   = Path("data/.geoeconomic_context_ckpt.json")
CONTEXT_SENTENCES = 2       # extra sentences before/after each keyword hit
MAX_EXCERPT_CHARS = 4_000   # hard cap on excerpt length sent to the model
REQUESTS_PER_SEC  = 0.33    # ~3 s between requests; free models have low limits
CHECKPOINT_EVERY  = 20      # save checkpoint after every N new documents
MAX_RETRIES       = 3       # retry on transient errors before recording failure

# Broad pattern for locating sentences that warrant inclusion in excerpts.
# Intentionally wider than the strict dictionary in notebook 03 — the LLM
# does the precise contextual judgement.
_EXCERPT_RE = re.compile(
    r'\b(?:'
    r'tariff|trade[\s\-]{0,5}(?:war|polic|tension|disput|deal|agreement)|'
    r'free[\s\-]{0,5}trade|wto|protectioni|deglobali|decoupl|reshoring|'
    r'sanction|embargo|export[\s\-]{0,5}control|import[\s\-]{0,5}restrict|'
    r'anti[\s\-]?dumping|countervailing|'
    r'geopolit|geoecon|supply[\s\-]{0,10}chain[\s\-]{0,20}(?:disrupt|risk|divers)|'
    r'russia|ukraine|taiwan|'
    r'china[\s\-]{0,20}(?:risk|tension|tariff|trade|war|decoupl)|'
    r'military[\s\-]{0,10}(?:action|escalat|conflict)|invasion|'
    r'arms[\s\-]{0,5}embargo|war(?!\w)'
    r')',
    re.IGNORECASE,
)

# ── Prompt templates ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a financial analyst assistant that reads earnings call excerpts "
    "and extracts structured information. You respond ONLY with a single, "
    "minified, valid JSON object — no markdown fences, no explanation, "
    "no extra text of any kind."
)

_Q_TEMPLATE = """\
Below are excerpts from an earnings call transcript.

Company: {company} ({ticker})
Reporting period: {period}
Risk categories flagged: {risk_types}

=== EXCERPTS (sentences containing geoeconomic language) ===
{excerpts}
=== END EXCERPTS ===

Using ONLY the information in the excerpts above, answer all eight questions \
by returning a single JSON object. Definitions and decision rules for each \
field are given below.

────────────────────────────────────────────────────────────
FIELD DEFINITIONS
────────────────────────────────────────────────────────────

"firm_operations_relevance"  [boolean]
  true  — The geoeconomic risk is explicitly linked to THIS company's own
          business: its revenues, cost structure, profit margins, specific
          products, named customers or suppliers, manufacturing sites, or
          geographic markets the firm operates in.
  false — The risk is cited only as general economic background or as a
          sector-wide issue without a stated connection to this firm's
          concrete operations.

"macro_context"  [boolean]
  true  — Executives or analysts place the risk in a broader macroeconomic,
          industry-wide, or geopolitical frame (e.g., "the whole industry
          faces…", "global supply chains are affected…", "the macro
          environment remains uncertain…").
  false — The discussion is confined entirely to this company's situation
          with no reference to broader trends.

"speaker_attribution"  [string: "executive_only" | "analyst_only" | "both"]
  Who raises or substantively discusses the geoeconomic risk topic?
  "executive_only" — Only management (CEO, CFO, COO, President, etc.)
                     discusses it.
  "analyst_only"   — Only external sell-side analysts bring it up.
  "both"           — Both management and analysts engage with the topic.
  If attribution is ambiguous, choose the most likely option based on
  conversational cues (e.g., "your exposure to…" = analyst addressing mgmt).

"response_discussed"  [boolean]
  true  — Management describes at least one specific, concrete action —
          already taken or firmly committed to — directly in response to
          the geoeconomic risk.
  false — No concrete response is described, OR statements are limited to
          generic monitoring language ("we are watching," "we remain
          cautious," "it is too early to tell").

"increase_investments"  [boolean]
  true  — The firm explicitly mentions INCREASING capital expenditure,
          accelerating investment, building new capacity, expanding
          manufacturing, or stockpiling inventory as a direct response
          to geoeconomic risk (e.g., reshoring production, building
          strategic buffer stock, constructing domestic facilities).
  false — No such response is mentioned.

"decrease_investments"  [boolean]
  true  — The firm explicitly mentions CUTTING, DEFERRING, or CANCELLING
          capital expenditure or investment programmes because of
          geoeconomic risk (e.g., pausing a plant expansion, reducing
          capex guidance, pulling back from a market).
  false — No such response is mentioned.

"find_new_suppliers"  [boolean]
  true  — The firm mentions sourcing from alternative suppliers, qualifying
          new vendors, diversifying its supply base, or shifting
          procurement away from tariff-affected or sanctioned regions.
  false — No such response is mentioned.

"stop_exports"  [boolean]
  true  — The firm mentions halting, reducing, or redirecting exports;
          withdrawing from specific customer markets; restricting
          shipments; or reducing commercial exposure to sanctioned,
          embargoed, or conflict-affected regions.
  false — No such response is mentioned.

────────────────────────────────────────────────────────────
Return ONLY this JSON (replace the placeholder values with your answers):
{{"firm_operations_relevance": true, "macro_context": true, \
"speaker_attribution": "both", "response_discussed": false, \
"increase_investments": false, "decrease_investments": false, \
"find_new_suppliers": false, "stop_exports": false}}\
"""


# ── Helper functions ───────────────────────────────────────────────────────────

def extract_excerpts(text: str) -> str:
    """Return sentences surrounding keyword hits, deduplicated and joined."""
    if not isinstance(text, str) or not text.strip():
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    include = set()
    for i, s in enumerate(sentences):
        if _EXCERPT_RE.search(s):
            for j in range(
                max(0, i - CONTEXT_SENTENCES),
                min(len(sentences), i + CONTEXT_SENTENCES + 1),
            ):
                include.add(j)
    if not include:
        return ""
    return " ".join(sentences[i] for i in sorted(include))[:MAX_EXCERPT_CHARS]


def get_risk_labels(row) -> list:
    labels = []
    for col, label in [
        ("trade_risk",        "trade policy / tariffs"),
        ("sanctions_risk",    "sanctions"),
        ("embargo_risk",      "embargo"),
        ("geopolitical_risk", "geopolitical risk"),
    ]:
        if row.get(col, False):
            labels.append(label)
    return labels


def call_llm(client, model: str, company: str, ticker: str,
             period: str, risk_types: list, excerpts: str) -> dict:
    """Call OpenRouter and parse JSON response. Raises on parse failure."""
    body = _Q_TEMPLATE.format(
        company=company,
        ticker=ticker,
        period=period,
        risk_types=", ".join(risk_types) if risk_types else "geoeconomic risk",
        excerpts=excerpts or "(no excerpts extracted — full text unavailable)",
    )
    # Prepend system instruction directly in the user message so the prompt
    # works with models that do not support a separate "system" role
    # (e.g. Google Gemma on AI Studio).
    full_prompt = SYSTEM_PROMPT + "\n\n" + body

    response = client.chat.completions.create(
        extra_body={},
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=300,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences the model may add despite instructions
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    # Some models wrap the JSON in extra commentary — extract the first {...}
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based context classification of geoeconomic risk mentions."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N documents (useful for testing)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip documents already present in the checkpoint file",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "Export it before running: export OPENROUTER_API_KEY='sk-or-v1-...'"
        )

    client = OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=api_key,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading flagged transcripts …")
    flagged = pd.read_csv("data/geoeconomic_matches_flagged.csv")

    print("Loading corpus documents …")
    corpus = pd.read_csv(
        "data/corpus_documents.csv",
        usecols=["url", "full_text", "exec_text", "analyst_text"],
    )

    df = flagged.merge(corpus, on="url", how="left")

    if args.limit:
        # For testing: prefer actual earnings calls with full metadata
        has_full_meta = df["ticker"].notna() & df["reporting_period"].notna()
        df = df[has_full_meta].head(args.limit)
        print(f"Limited to first {args.limit} documents with full metadata for testing.")

    total = len(df)
    print(f"Documents to process: {total}\n")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    results: dict = {}
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} documents already processed.\n")

    # ── Process ────────────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        url = str(row.get("url", ""))

        if args.resume and url in results:
            continue

        company    = str(row.get("company_name", "Unknown Company"))
        ticker     = str(row.get("ticker", "N/A"))
        period     = str(row.get("reporting_period", "Unknown"))
        risk_types = get_risk_labels(row)

        excerpts = extract_excerpts(str(row.get("full_text") or ""))
        if not excerpts:
            combined = " ".join([
                str(row.get("exec_text") or ""),
                str(row.get("analyst_text") or ""),
            ])
            excerpts = extract_excerpts(combined) or combined[:MAX_EXCERPT_CHARS]

        done = len(results)
        print(f"  [{done + 1:>4}/{total}] {company} ({ticker}) | {period}")

        record = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                answer = call_llm(
                    client, args.model,
                    company, ticker, period, risk_types, excerpts,
                )
                record = {
                    "url":              url,
                    "company_name":     company,
                    "ticker":           ticker,
                    "reporting_period": period,
                    "risk_types":       risk_types,
                    "excerpts":         excerpts,
                    **answer,
                }
                break
            except Exception as exc:
                msg = str(exc)
                print(f"           attempt {attempt}/{MAX_RETRIES} failed: {msg}")
                if attempt < MAX_RETRIES:
                    # longer back-off for rate-limit errors
                    wait = 30 if "429" in msg else 5 * attempt
                    print(f"           ↳ waiting {wait}s …")
                    time.sleep(wait)

        if record is None:
            record = {
                "url":              url,
                "company_name":     company,
                "ticker":           ticker,
                "reporting_period": period,
                "risk_types":       risk_types,
                "excerpts":         excerpts,
                "error":            "all retries exhausted",
            }

        results[url] = record

        if len(results) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f)
            print(f"           ↳ checkpoint saved ({len(results)} docs)")

        time.sleep(1.0 / REQUESTS_PER_SEC)

    # ── Write final output ─────────────────────────────────────────────────────
    output_list = list(results.values())
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    n_ok  = sum(1 for r in output_list if "error" not in r)
    n_err = len(output_list) - n_ok
    print(f"\nDone.")
    print(f"  Processed : {len(output_list):,} documents")
    print(f"  Succeeded : {n_ok:,}")
    print(f"  Errors    : {n_err:,}")
    print(f"  Output    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
