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

The analysis unfolds in three parts:

1. **Broad topic exploration** (Part 1): apply unsupervised topic modeling
   (LDA) to the full corpus to map the latent topical structure of earnings
   call discourse, understand which sectors dominate, how topics are
   distributed across the Q4 2025 reporting season, and whether executive
   speech and analyst questions reflect distinct topical priorities.

2. **Geoeconomic risk search** (Part 2): a dictionary-based keyword search
   that flags transcripts where a geoeconomic risk term and an uncertainty
   term co-occur **in the same sentence**, followed by an LLM-based context
   classifier that characterises how and by whom the risk is discussed.

3. **Sentiment analysis with FinBERT** (Part 3): a domain-adapted BERT model
   assigns positive/negative/neutral sentiment scores to each transcript,
   enabling comparison of tone across topics, reporting periods, and speaker
   roles.

---

## A note on sample balance

> **All cross-quarter comparisons in this analysis are illustrative only.**
>
> The corpus is dominated by **Q4 2025 earnings calls** (~1,850 transcripts out
> of ~2,900 total). The remaining quarters each contain between a handful and a
> few hundred transcripts, reflecting companies with non-calendar fiscal years.
> Ideally this analysis would be run on a corpus with a comparable number of
> transcripts per quarter so that temporal trends are not confounded with sample
> composition. Figures showing distributions by reporting period include n=
> labels for this reason; interpret quarter-to-quarter differences with caution.

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
03_geoeconomic_search.ipynb      (Python / regex)
        ↓
   data/geoeconomic_matches.csv
   data/geoeconomic_matches_flagged.csv
   figures/10, 11, 12
        ↓
04_llm_context_analysis.py       (Python / OpenRouter API)
        ↓
   data/geoeconomic_context.json
        ↓
05_finbert_sentiment.ipynb       (Python / HuggingFace FinBERT)
        ↓
   data/finbert_sentiment.csv
   data/finbert_results.csv
   figures/14–18
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

### `03_geoeconomic_search.ipynb`

Dictionary-based keyword search across all 2,907 transcripts. A transcript is
flagged only if at least one geoeconomic term and one risk/uncertainty term
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
- `data/geoeconomic_matches_flagged.csv` — 250 matched documents only
- `figures/10–12` — bar chart, by-period breakdown, top companies

---

### `04_llm_context_analysis.py`

For each flagged transcript, uses an LLM (via
[OpenRouter](https://openrouter.ai)) to classify the *context* in which
geoeconomic risk is discussed. Extracts sentences surrounding keyword hits and
asks a structured set of questions.

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
export OPENROUTER_API_KEY="sk-or-v1-..."
python 04_llm_context_analysis.py --limit 20   # test
python 04_llm_context_analysis.py              # full run (250 docs)
python 04_llm_context_analysis.py --resume     # resume if interrupted
```

**Output:** `data/geoeconomic_context.json`

---

### `05_finbert_sentiment.ipynb`

Complements the LDA topic model with a **FinBERT sentiment analysis**.

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
- `data/finbert_sentiment.csv` — raw scores per document
- `data/finbert_results.csv` — merged with LDA topics and geoeconomic flags
- `figures/14–18` — sentiment distributions and comparisons

**Run (requires `.venv`, GPU recommended for speed):**
```bash
source /path/to/.venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=earnings-venv \
  --ExecutePreprocessor.timeout=7200 \
  05_finbert_sentiment.ipynb
```

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

---

## Project structure

```
.
├── 00_prepare_corpus.R           # R: corpus assembly and speaker classification
├── 01_topic_modeling.ipynb       # Python: LDA topic modeling
├── 02_visualize_topics.R         # R: ggplot2 visualizations (figures 07, 09)
├── 03_geoeconomic_search.ipynb   # Python: sentence-level geoeconomic risk search
├── 04_llm_context_analysis.py    # Python: LLM context classification (OpenRouter)
├── 05_finbert_sentiment.ipynb    # Python: FinBERT sentiment analysis
├── data/
│   ├── doc_topic_distributions.csv    # γ matrix (document–topic)
│   ├── topic_terms.csv                # β matrix (topic–term)
│   ├── role_topic_comparison.csv      # topic proportions by speaker role
│   ├── top_docs_per_topic.csv         # most representative docs per topic
│   ├── geoeconomic_matches.csv        # all docs with per-category flags
│   ├── geoeconomic_matches_flagged.csv # flagged docs only (250)
│   ├── geoeconomic_context.json       # LLM context classification results
│   ├── finbert_sentiment.csv          # raw FinBERT scores per document
│   └── finbert_results.csv            # merged: scores + topics + geo flags
└── figures/                      # all PNG outputs (00–04, 07, 09–12, 14–18)
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
- `04_llm_context_analysis.py` writes a rolling checkpoint every 20 documents so a long run can be safely interrupted and resumed with `--resume`.
- `05_finbert_sentiment.ipynb` caches inference results to `data/finbert_sentiment.csv` so re-runs are instant.
