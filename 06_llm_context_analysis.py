#!/usr/bin/env python3
"""
06_llm_context_analysis.py

For each transcript flagged as containing geoeconomic risk language,
uses an LLM to classify the context in which that risk is discussed.
Results are written to data/geoeconomic_context.json.

The script uses the Anthropic API by default (set ANTHROPIC_API_KEY).
As a free alternative, OpenRouter (https://openrouter.ai) also works —
set OPENROUTER_API_KEY and pass --provider openrouter.

Usage
-----
    source /path/to/.venv/bin/activate

    # Anthropic (recommended — fast, reliable)
    export ANTHROPIC_API_KEY="sk-ant-..."
    python 06_llm_context_analysis.py

    # OpenRouter free tier (50 req/day without credits)
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python 06_llm_context_analysis.py --provider openrouter

    # Test on first 10 documents
    python 06_llm_context_analysis.py --limit 10

    # Resume an interrupted run
    python 06_llm_context_analysis.py --resume

Output
------
    data/geoeconomic_context.json        — final results, one object per doc
    data/.geoeconomic_context_ckpt.json  — rolling checkpoint (auto-deleted
                                           on clean completion)
"""

import os
import re
import json
import time
import argparse
import pandas as pd
import anthropic
from openai import OpenAI
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_ANTHROPIC  = "claude-haiku-4-5-20251001"
DEFAULT_MODEL_OPENROUTER = "google/gemma-3-27b-it:free"
OPENROUTER_BASE          = "https://openrouter.ai/api/v1"

OUTPUT_PATH       = Path("data/geoeconomic_context.json")
CHECKPOINT_PATH   = Path("data/.geoeconomic_context_ckpt.json")
CONTEXT_SENTENCES = 2
MAX_EXCERPT_CHARS = 4_000
CHECKPOINT_EVERY  = 20
MAX_RETRIES       = 3

# Rate limits: Anthropic Haiku is fast; OpenRouter free tier is slow
RATE_LIMITS = {"anthropic": 2.0, "openrouter": 0.25}   # requests per second

# ── Keyword pattern for excerpt extraction ────────────────────────────────────
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
by returning a single JSON object. Definitions and decision rules:

"firm_operations_relevance"  [boolean]
  true  — The geoeconomic risk is explicitly linked to THIS company's own
          business: revenues, costs, margins, supply chain, products,
          named customers or suppliers, manufacturing sites, or geographic
          markets the firm operates in.
  false — The risk is cited only as general background with no stated
          connection to this firm's concrete operations.

"macro_context"  [boolean]
  true  — Executives or analysts place the risk in a broader macroeconomic,
          industry-wide, or geopolitical frame.
  false — Discussion is confined to this company with no broader framing.

"speaker_attribution"  [string: "executive_only" | "analyst_only" | "both"]
  Who substantively discusses the geoeconomic risk?
  "executive_only" — Only management (CEO, CFO, etc.).
  "analyst_only"   — Only external sell-side analysts.
  "both"           — Both parties engage with the topic.

"response_discussed"  [boolean]
  true  — Management describes at least one specific, concrete action in
          response to the risk. Generic reassurances do NOT count.
  false — No concrete response described.

"increase_investments"  [boolean]
  true  — Firm mentions INCREASING capex, building new capacity, or
          stockpiling inventory as a direct response to geoeconomic risk.

"decrease_investments"  [boolean]
  true  — Firm mentions CUTTING, DEFERRING, or CANCELLING capex or
          investment programmes because of geoeconomic risk.

"find_new_suppliers"  [boolean]
  true  — Firm mentions sourcing from alternative suppliers, diversifying
          supply base, or shifting procurement away from affected regions.

"stop_exports"  [boolean]
  true  — Firm mentions halting, reducing, or redirecting exports;
          withdrawing from markets; or reducing exposure to sanctioned or
          conflict-affected regions.

Return ONLY this JSON:
{{"firm_operations_relevance": true, "macro_context": true, \
"speaker_attribution": "both", "response_discussed": false, \
"increase_investments": false, "decrease_investments": false, \
"find_new_suppliers": false, "stop_exports": false}}\
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_excerpts(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    include = set()
    for i, s in enumerate(sentences):
        if _EXCERPT_RE.search(s):
            for j in range(max(0, i - CONTEXT_SENTENCES),
                           min(len(sentences), i + CONTEXT_SENTENCES + 1)):
                include.add(j)
    if not include:
        return ""
    return " ".join(sentences[i] for i in sorted(include))[:MAX_EXCERPT_CHARS]


def get_risk_labels(row) -> list:
    labels = []
    for col, label in [("trade_risk", "trade policy / tariffs"),
                        ("sanctions_risk", "sanctions"),
                        ("embargo_risk", "embargo"),
                        ("geopolitical_risk", "geopolitical risk")]:
        if row.get(col, False):
            labels.append(label)
    return labels


def call_anthropic(client, model, company, ticker, period, risk_types, excerpts):
    prompt = _Q_TEMPLATE.format(
        company=company, ticker=ticker, period=period,
        risk_types=", ".join(risk_types) or "geoeconomic risk",
        excerpts=excerpts or "(no excerpts extracted)",
    )
    msg = client.messages.create(
        model=model, max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    return json.loads(match.group(0) if match else raw)


def call_openrouter(client, model, company, ticker, period, risk_types, excerpts):
    prompt = SYSTEM_PROMPT + "\n\n" + _Q_TEMPLATE.format(
        company=company, ticker=ticker, period=period,
        risk_types=", ".join(risk_types) or "geoeconomic risk",
        excerpts=excerpts or "(no excerpts extracted)",
    )
    response = client.chat.completions.create(
        extra_body={}, model=model, max_tokens=300, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    return json.loads(match.group(0) if match else raw)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openrouter"],
                        default="anthropic")
    parser.add_argument("--model", default=None,
                        help="Override the default model for the chosen provider")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    provider = args.provider

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set.\n"
                "Export it: export ANTHROPIC_API_KEY='sk-ant-...'\n"
                "Or use OpenRouter (free): --provider openrouter"
            )
        client = anthropic.Anthropic(api_key=api_key)
        model  = args.model or DEFAULT_MODEL_ANTHROPIC
        call   = call_anthropic
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set.\n"
                "Get a free key at https://openrouter.ai and export it:\n"
                "export OPENROUTER_API_KEY='sk-or-v1-...'"
            )
        client = OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)
        model  = args.model or DEFAULT_MODEL_OPENROUTER
        call   = call_openrouter

    rps = RATE_LIMITS[provider]
    print(f"Provider : {provider}  |  Model: {model}  |  Rate: {rps} req/s")

    print("Loading flagged transcripts …")
    flagged = pd.read_csv("data/geoeconomic_matches_flagged.csv")
    corpus  = pd.read_csv("data/corpus_documents.csv",
                          usecols=["url", "full_text", "exec_text", "analyst_text"])
    df = flagged.merge(corpus, on="url", how="left")

    if args.limit:
        has_meta = df["ticker"].notna() & df["reporting_period"].notna()
        df = df[has_meta].head(args.limit)
        print(f"Limited to first {args.limit} documents with full metadata.")

    total = len(df)
    print(f"Documents to process: {total:,}\n")

    results: dict = {}
    if args.resume and CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            results = json.load(f)
        print(f"Resuming — {len(results):,} already done.\n")

    for _, row in df.iterrows():
        url = str(row.get("url", ""))
        if args.resume and url in results:
            continue

        company    = str(row.get("company_name", "Unknown"))
        ticker     = str(row.get("ticker", "N/A"))
        period     = str(row.get("reporting_period", "Unknown"))
        risk_types = get_risk_labels(row)

        excerpts = extract_excerpts(str(row.get("full_text") or ""))
        if not excerpts:
            combined = " ".join([str(row.get("exec_text") or ""),
                                 str(row.get("analyst_text") or "")])
            excerpts = extract_excerpts(combined) or combined[:MAX_EXCERPT_CHARS]

        done = len(results)
        print(f"  [{done+1:>4}/{total}] {company} ({ticker}) | {period}")

        record = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                answer = call(client, model, company, ticker,
                              period, risk_types, excerpts)
                record = {"url": url, "company_name": company,
                          "ticker": ticker, "reporting_period": period,
                          "risk_types": risk_types, "excerpts": excerpts,
                          **answer}
                break
            except Exception as exc:
                msg = str(exc)
                print(f"           attempt {attempt}/{MAX_RETRIES}: {msg[:120]}")
                if attempt < MAX_RETRIES:
                    wait = 30 if "429" in msg else 5 * attempt
                    print(f"           ↳ waiting {wait}s …")
                    time.sleep(wait)

        if record is None:
            record = {"url": url, "company_name": company, "ticker": ticker,
                      "reporting_period": period, "risk_types": risk_types,
                      "excerpts": excerpts, "error": "all retries exhausted"}

        results[url] = record

        if len(results) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f)
            print(f"           ↳ checkpoint ({len(results):,} docs)")

        time.sleep(1.0 / rps)

    output_list = list(results.values())
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    n_ok  = sum(1 for r in output_list if "error" not in r)
    n_err = len(output_list) - n_ok
    print(f"\nDone.  Processed: {len(output_list):,}  |  OK: {n_ok:,}  |  Errors: {n_err:,}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
