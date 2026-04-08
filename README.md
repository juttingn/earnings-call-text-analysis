# Earnings Call Transcript — Text Analysis

A computational text analysis of U.S. earnings call transcripts, combining
R and Python to explore the topical structure of corporate financial
communication and, more specifically, the discussions surrounding
**geoeconomic risk** in firm-level discourse.

This project is a direct continuation of the web scraper developed in
[juttingn/earnings-call-transcripts](https://github.com/juttingn/earnings-call-transcripts),
which collected ~2,900 full transcripts from investing.com. Where the scraper
built the corpus, this repository analyses it.

---

## Research motivation

This project links directly to my Master Thesis
([available here](https://juttingn.github.io/personal-site/thesis.pdf)),
in which I estimated the **geoeconomic risk exposure** of U.S. listed companies
over time and examined how that exposure shaped firms' investment responses to
Russia's invasion of Ukraine in 2022. That analysis relied on the
[NL Analytics](https://www.nlanalytics.com) platform, which provides only
dictionary-based keyword counts on earnings call transcripts — a useful but
inherently limited approach. By gaining full access to the raw transcripts, I
can now apply more sophisticated methods to the same underlying material.

The analysis follows a deliberate progression from broad exploration to targeted
precision, with each step motivated by the limitations of the one before it:

**Step 1 — Exploratory topic modelling (LDA, scripts 01–02).**
The starting point is purely inductive: with no prior assumptions about what
matters in these transcripts, LDA maps the latent topical structure of the full
corpus. This reveals what executives and analysts talk about most, how topics
vary across sectors and reporting quarters, and whether management and analyst
speech occupy different corners of the topical space. The output is a descriptive
map of earnings-call discourse that motivates the more focused work that follows.

**Step 2 — Narrowing to geoeconomic risk (dictionary search, script 05).**
The thesis research question is about *geoeconomic risk* specifically — how firms
perceive and respond to trade policy uncertainty, sanctions, and geopolitical
conflict. A sentence-level keyword search flags the 250 transcripts (8.6% of the
corpus) where a geoeconomic term and a risk/uncertainty term co-occur in the same
sentence. This is a deliberate step back toward the dictionary-based approach of
the original thesis, but applied more carefully: sentence-level co-occurrence
avoids the false positives that arise when both term types appear in entirely
unrelated passages of the same document.

**Step 3 — Sentiment as a first characterisation (FinBERT, scripts 03–04).**
With the geoeconomic-flagged transcripts identified, a natural first question is
whether they are discussed in a more negative tone than the rest of the corpus.
FinBERT — a BERT model fine-tuned on financial news text — assigns
positive/negative/neutral scores to each transcript. The comparison between
flagged and non-flagged documents provides a quick read on whether geoeconomic
risk language is associated with a discernibly more negative register. In
practice, the sentiment signal turns out to be only weakly discriminating: while
flagged transcripts do score somewhat more negatively on average, the
distributions overlap heavily. This limits what can be inferred from tone alone
and motivates a more precise approach.

**Step 4 — Structured LLM annotation (scripts 06–07).**
Because sentiment alone cannot tell us *how* geoeconomic risk is discussed —
whether it is tied to the firm's own operations, who raises it, and what
management plans to do about it — the final step uses a large language model
(`claude-haiku-4-5-20251001`) to classify each flagged transcript against eight
structured questions. This transforms the raw text into actionable variables:
firm-level relevance, macro framing, speaker attribution, and four concrete
response types (increase investment, cut investment, diversify suppliers, reduce
exports). Where FinBERT measures *tone*, the LLM annotation measures *content*,
producing a richer and more interpretable picture of how U.S.-listed firms
engage with geoeconomic risk in their earnings calls.

---

## A note on sample balance

> **All cross-quarter comparisons in this analysis are illustrative only.**
>
> The corpus is dominated by **Q4 2025 earnings calls** (~1,850 transcripts out
> of ~2,900 total). The remaining quarters each contain between a handful and a
> few hundred transcripts, reflecting companies with non-calendar fiscal years
> (Q2 2025: 6, Q3 2025: 6, Q1 2026: 149, Q2 2026: 125, Q3 2026: 163,
> Q4 2026: 15; plus ~590 undated conference presentations). Q1 2025 is entirely
> absent from the corpus. Ideally this analysis would be run on a corpus with a
> comparable number of transcripts per quarter so that temporal trends are not
> confounded with sample composition. All temporal figures include n= labels for
> this reason; interpret quarter-to-quarter differences with caution.

---

## Pipeline

```
00_prepare_corpus.R              (R)
        ↓
   data/corpus_documents.csv
   data/corpus_speaker_level.csv
        ↓
01_topic_modeling.ipynb          (Python / gensim)
        ↓
   data/doc_topic_distributions.csv
   data/topic_terms.csv
   data/role_topic_comparison.csv
        ↓
02_visualize_topics.R            (R / ggplot2)
        ↓
   figures/07, 09
        ↓
03_finbert_inference.py          (Python / HuggingFace FinBERT, standalone batch)
        ↓
   data/finbert_sentiment.csv
        ↓
04_finbert_sentiment.ipynb       (Python / FinBERT visualisations)
        ↓
   data/finbert_results.csv
   figures/14–18
        ↓
05_geoeconomic_search.ipynb      (Python / regex, sentence-level matching)
        ↓
   data/geoeconomic_matches.csv
   data/geoeconomic_matches_flagged.csv
   figures/10, 11, 12
        ↓
06_llm_context_analysis.py       (Python / Anthropic API or OpenRouter)
        ↓
   data/geoeconomic_context.json
        ↓
07_llm_results_viz.ipynb         (Python / matplotlib)
        ↓
   figures/19–24
```

---

## Scripts

### `00_prepare_corpus.R`

Loads all individual transcript RDS files from the scraper's `transcripts_raw/`
directory, classifies each speaker turn by role type (executive, analyst,
investor-relations, operator), and aggregates text at the transcript level.

**Outputs:**
- `data/corpus_documents.csv` — one row per transcript with full text,
  executive-only text, analyst-only text, and metadata
- `data/corpus_speaker_level.csv` — one row per speaker turn with role label

**Run:**
```bash
/usr/local/bin/Rscript 00_prepare_corpus.R
```

---

### `01_topic_modeling.ipynb`

The main topic-modeling notebook. Preprocesses the corpus (tokenization,
custom earnings-call stopword removal, lemmatization), fits an LDA model
using [gensim](https://radimrehurek.com/gensim/), selects the optimal number
of topics (K=20) via coherence scoring, and compares the topic footprints of
executive and analyst speech.

**Requires:** the `.venv` Python 3.10 environment from the scraper project.

**Outputs:**
- `data/doc_topic_distributions.csv` — γ matrix: per-document topic proportions
- `data/topic_terms.csv` — β matrix: top 30 terms per topic with probabilities
- `data/role_topic_comparison.csv` — topic proportions by speaker role

---

### `02_visualize_topics.R`

Reads the LDA outputs and produces publication-quality figures using ggplot2.
All available reporting quarters are shown; n= labels flag sample sizes so
the Q4 2025 dominance is immediately apparent.

**Required R packages:** `dplyr`, `tidyr`, `readr`, `lubridate`, `ggplot2`,
`forcats`, `stringr`, `scales`

**Run:**
```bash
/usr/local/bin/Rscript 02_visualize_topics.R
```

---

### `03_finbert_inference.py`

Standalone batch inference script that runs FinBERT on all corpus documents.
Detects MPS (Apple Silicon), CUDA, or CPU automatically. Saves results to
`data/finbert_sentiment.csv` with a rolling checkpoint so long runs can be
safely interrupted and resumed with `--resume`.

**Run:**
```bash
source /path/to/.venv/bin/activate
python 03_finbert_inference.py            # full corpus
python 03_finbert_inference.py --limit 100 --resume
```

**Output:** `data/finbert_sentiment.csv`

---

### `04_finbert_sentiment.ipynb`

Loads `data/finbert_sentiment.csv` (generated by `03_finbert_inference.py`)
and produces publication-quality sentiment figures.

**Why FinBERT?** Standard sentiment tools (VADER, TextBlob) are trained on
general-purpose text and struggle with financial language: words like
"impairment," "write-down," or "guidance cut" carry clear negative meaning in
financial discourse but may not score accordingly in a generic model.
FinBERT ([Araci, 2019](https://arxiv.org/abs/1908.10063)) is a BERT-base model
fine-tuned on ~10,000 financial news sentences (Reuters, Bloomberg) and the
Financial PhraseBank. It assigns **positive / negative / neutral** labels with
calibrated probabilities and substantially outperforms generic models on
financial sentiment benchmarks.

**How it complements LDA:** LDA reveals *what topics* are discussed; FinBERT
reveals *with what tone*. Together they provide a richer picture — e.g.,
identifying which LDA topics are discussed most negatively, whether executives
speak more positively than analysts, and whether geoeconomic-flagged transcripts
carry a meaningfully more negative tone.

Because BERT has a 512-token limit, transcripts are split into 400-token
sentence-aligned chunks; scores are averaged across chunks to produce a
document-level estimate.

**Outputs:**
- `data/finbert_results.csv` — merged: scores + LDA topics + geoeconomic flags
- `figures/14–18` — sentiment distributions and comparisons

---

### `05_geoeconomic_search.ipynb`

Dictionary-based keyword search across all 2,907 transcripts. A transcript is
flagged only if at least one geoeconomic risk term and one uncertainty term
appear in the **same sentence** — sentence-level co-occurrence rather than a
weaker document-level AND. This avoids false positives from transcripts that
happen to mention both term types in completely unrelated passages.

**Four risk categories:**

| Category | Sentence-level logic |
|---|---|
| **Trade risk** | (trade-policy term AND risk term) OR (trade-flow term AND trade-restriction term) |
| **Sanctions risk** | sanction/penalty term AND risk term |
| **Embargo risk** | embargo/export-ban term AND risk term |
| **Geopolitical risk** | geopolitical term AND risk term (six sub-conditions) |

**Results (sentence-level matching):**
- 250 documents flagged (out of 2,907 total, 8.6%)
- Trade risk: 205 documents
- Geopolitical risk: 60 documents
- Sanctions risk: 2 documents
- Embargo risk: 0 documents

**Outputs:**
- `data/geoeconomic_matches.csv` — all documents with per-category flags
- `data/geoeconomic_matches_flagged.csv` — 250 flagged documents only
- `figures/10–12` — match counts by category, by-period breakdown, top companies

---

### `06_llm_context_analysis.py`

For each flagged transcript, uses an LLM to classify the *context* in which
geoeconomic risk is discussed. Extracts sentences surrounding keyword hits and
asks eight structured questions. Supports the Anthropic API (default, fast) and
OpenRouter (free tier, 50 req/day without credits) as a fallback.

**Questions answered for each transcript:**

| Field | Question |
|---|---|
| `firm_operations_relevance` | Is the risk tied to this firm's own revenues, costs, supply chain, or operations? |
| `macro_context` | Is the risk placed in a broader macroeconomic or industry-wide frame? |
| `speaker_attribution` | Who raises the topic — executives only, analysts only, or both? |
| `response_discussed` | Does management describe concrete actions in response? |
| `increase_investments` | Increasing capex or expanding capacity as a response? |
| `decrease_investments` | Cutting or deferring investment as a response? |
| `find_new_suppliers` | Diversifying supply base or sourcing from alternative regions? |
| `stop_exports` | Halting or reducing exports or withdrawing from affected markets? |

**Run:**
```bash
# Anthropic (recommended)
export ANTHROPIC_API_KEY="sk-ant-..."
python 06_llm_context_analysis.py
python 06_llm_context_analysis.py --limit 20   # test
python 06_llm_context_analysis.py --resume     # resume if interrupted

# OpenRouter free tier (50 req/day without credits)
export OPENROUTER_API_KEY="sk-or-v1-..."
python 06_llm_context_analysis.py --provider openrouter
```

**Output:** `data/geoeconomic_context.json`

---

### `07_llm_results_viz.ipynb`

Visualises the LLM classification results from `data/geoeconomic_context.json`.

**Outputs:**
- `figures/19–24` — binary flag overview, speaker attribution, response types,
  risk × context heatmap, temporal breakdown, top companies by discussion depth

---

## Output figures

| File | Description |
|---|---|
| `figures/00_corpus_overview.png` | Corpus size by reporting period and event type |
| `figures/01_top_terms_corpus.png` | Top 30 terms after stopword removal |
| `figures/02_coherence_scores.png` | LDA coherence curve for K selection |
| `figures/03_topic_top_terms.png` | Top 10 terms per topic (grid) |
| `figures/04_executive_vs_analyst_topics.png` | Topic fingerprints by speaker role |
| `figures/07_topic_heatmap_by_period.png` | Topic × period heatmap (all quarters, n= labels) |
| `figures/09_dominant_topic_by_period.png` | Dominant topic share per reporting period (all quarters, n= labels) |
| `figures/10_georisk_match_counts.png` | Geoeconomic risk match counts by category |
| `figures/11_georisk_by_period.png` | Flagged documents by reporting period and category (all quarters, n= labels) |
| `figures/12_georisk_top_companies.png` | Top 20 companies by geoeconomic risk mentions |
| `figures/14_finbert_sentiment_dist.png` | Overall sentiment distribution (FinBERT) |
| `figures/15_finbert_by_topic.png` | Net sentiment by LDA topic |
| `figures/16_finbert_by_period.png` | Net sentiment by reporting quarter |
| `figures/17_finbert_exec_vs_analyst.png` | Executive vs analyst net sentiment |
| `figures/18_finbert_georisk_vs_not.png` | Geoeconomic-flagged vs non-flagged sentiment |
| `figures/19_llm_binary_flags.png` | LLM context flags: share of True across all 7 questions |
| `figures/20_llm_speaker_attribution.png` | Speaker attribution overall and by risk category |
| `figures/21_llm_response_types.png` | Management response types (among transcripts with concrete response) |
| `figures/22_llm_risk_context_heatmap.png` | Risk category × context flag heatmap |
| `figures/23_llm_relevance_by_period.png` | Firm relevance vs. macro framing by reporting period |
| `figures/24_llm_top_companies_depth.png` | Top 20 companies by geoeconomic risk discussion depth |

---

## Project structure

```
.
├── 00_prepare_corpus.R           # R: corpus assembly and speaker classification
├── 01_topic_modeling.ipynb       # Python: LDA topic modeling
├── 02_visualize_topics.R         # R: ggplot2 visualizations (figures 07, 09)
├── 03_finbert_inference.py       # Python: FinBERT batch inference (standalone)
├── 04_finbert_sentiment.ipynb    # Python: FinBERT visualizations (figures 14–18)
├── 05_geoeconomic_search.ipynb   # Python: sentence-level geoeconomic risk search
├── 06_llm_context_analysis.py    # Python: LLM context classification (Anthropic / OpenRouter)
├── 07_llm_results_viz.ipynb      # Python: LLM results visualizations (figures 19–24)
├── data/
│   ├── doc_topic_distributions.csv    # γ matrix (document–topic)
│   ├── topic_terms.csv                # β matrix (topic–term)
│   ├── role_topic_comparison.csv      # topic proportions by speaker role
│   ├── top_docs_per_topic.csv         # most representative docs per topic
│   ├── geoeconomic_matches.csv        # all docs with per-category flags
│   ├── geoeconomic_matches_flagged.csv # flagged docs only (250)
│   ├── geoeconomic_context.json       # LLM context classification results (250 docs)
│   ├── finbert_sentiment.csv          # raw FinBERT scores per document
│   └── finbert_results.csv            # merged: scores + topics + geo flags
└── figures/                      # all PNG outputs (00–04, 07, 09–12, 14–24)
```

Files generated at runtime (excluded from version control):
```
data/corpus_documents.csv        # regenerate with 00_prepare_corpus.R
data/corpus_speaker_level.csv    # regenerate with 00_prepare_corpus.R
```

---

## Environment

- **R:** `/usr/local/bin/Rscript` (system R; do **not** use the Anaconda R at `/Users/.../anaconda3/bin/R`)
- **Python:** `.venv/` inside the scraper project directory (Python 3.10, NumPy < 2); the Anaconda base environment has a NumPy 2.0 incompatibility. Always activate the dedicated `.venv`.
- `06_llm_context_analysis.py` writes a rolling checkpoint every 20 documents so a long run can be safely interrupted and resumed with `--resume`.
- `03_finbert_inference.py` caches inference results to `data/finbert_sentiment.csv` so re-runs of `04_finbert_sentiment.ipynb` are instant.
