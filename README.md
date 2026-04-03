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

The analysis unfolds in two parts:

1. **Broad topic exploration** (Part 1): apply unsupervised topic modeling
   (LDA) to the full corpus to map the latent topical structure of earnings
   call discourse, understand which sectors dominate, how topics are
   distributed across the Q4 2025 reporting season, and whether executive
   speech and analyst questions reflect distinct topical priorities.

2. **Geoeconomic risk deep-dive** (Part 2): zoom in on the documents most
   associated with geoeconomic language — trade policy, sanctions, embargoes,
   geopolitical uncertainty — and use a large language model to classify the
   context in which that risk is discussed at the firm level. This extends the
   Master Thesis analysis to a richer linguistic window, moving beyond keyword
   counts to structured inference about firm-level risk exposure and response.

---

## Pipeline

```
00_prepare_corpus.R          (R)
        ↓
   data/corpus_documents.csv
   data/corpus_speaker_level.csv
        ↓
01_topic_modeling.ipynb      (Python / gensim)
        ↓
   data/doc_topic_distributions.csv
   data/topic_terms.csv
   data/role_topic_comparison.csv
        ↓
02_visualize_topics.R        (R / ggplot2)
        ↓
   figures/00–09  (PNG outputs)
        ↓
03_geoeconomic_search.ipynb  (Python / regex)
        ↓
   data/geoeconomic_matches.csv
   data/geoeconomic_matches_flagged.csv
   figures/10–13  (PNG outputs)
        ↓
04_llm_context_analysis.py   (Python / Claude API)
        ↓
   data/geoeconomic_context.json
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
of topics via coherence scoring, and compares the topic footprints of
executive and analyst speech.

**Requires:** the `.venv` Python 3.10 environment from the scraper project
(or any environment with `gensim`, `nltk`, `pandas`, `matplotlib`,
`scikit-learn`).

**Outputs:**
- `data/doc_topic_distributions.csv` — γ matrix: per-document topic proportions
- `data/topic_terms.csv` — β matrix: top 30 terms per topic with probabilities
- `data/role_topic_comparison.csv` — topic proportions by speaker role

**Run in Jupyter:**
```bash
source /path/to/.venv/bin/activate
jupyter lab 01_topic_modeling.ipynb
```

---

### `02_visualize_topics.R`

Reads the LDA outputs and produces a series of publication-quality figures
using ggplot2. Temporal charts are restricted to Q4 2025 and earlier, since
the corpus is dominated by the Q4 2025 earnings season.

**Required R packages:** `dplyr`, `tidyr`, `readr`, `lubridate`, `ggplot2`,
`forcats`, `stringr`, `tidytext`, `scales`

**Run:**
```bash
/usr/local/bin/Rscript 02_visualize_topics.R
```

---

### `03_geoeconomic_search.ipynb`

Dictionary-based keyword search across all 2,907 transcripts. Flags documents
that contain a co-occurrence of at least one topic keyword and one
risk/uncertainty keyword from four geoeconomic risk categories:

| Category | Logic |
|---|---|
| **Trade risk** | (trade-policy term AND risk/uncertainty term) OR (trade-flow term AND trade-restriction term) |
| **Sanctions risk** | sanction/penalty term AND risk/uncertainty term |
| **Embargo risk** | embargo/export-ban term AND risk/uncertainty term |
| **Geopolitical risk** | geopolitical term AND risk/uncertainty term |

**Results (Q4 2025 corpus):**
- 1,626 documents flagged (out of 2,907 total, 55.9%)
- Trade risk: 1,512 documents
- Geopolitical risk: 354 documents
- Sanctions risk: 9 documents
- Embargo risk: 1 document

**Outputs:**
- `data/geoeconomic_matches.csv` — all 2,907 documents with per-category flags
- `data/geoeconomic_matches_flagged.csv` — 1,626 matched documents only
- `figures/10–13` — match-count bar chart, by-period breakdown, top companies,
  category co-occurrence matrix

---

### `04_llm_context_analysis.py`

For each of the 1,626 flagged transcripts, uses the Claude API to classify
the *context* in which geoeconomic risk is discussed. Extracts the sentences
surrounding keyword hits and asks a structured set of questions, returning a
JSON record per document.

**Questions answered for each transcript:**

| Field | Question |
|---|---|
| `firm_operations_relevance` | Is the risk discussed in direct relation to this firm's own revenues, costs, supply chain, or operations? |
| `macro_context` | Is the risk placed in a broader macroeconomic or industry-wide frame? |
| `speaker_attribution` | Who raises the topic — executives only, analysts only, or both? |
| `response_discussed` | Does management describe concrete actions taken or planned in response? |
| `increase_investments` | Does the firm mention increasing capex or expanding capacity as a response? |
| `decrease_investments` | Does the firm mention cutting or deferring investment as a response? |
| `find_new_suppliers` | Does the firm mention diversifying its supply base or sourcing from alternative regions? |
| `stop_exports` | Does the firm mention halting or reducing exports or withdrawing from affected markets? |

**Requires:** `ANTHROPIC_API_KEY` environment variable set.

**Run:**
```bash
source /path/to/.venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."

# Full run (~1,626 documents)
python 04_llm_context_analysis.py

# Test on a small sample first
python 04_llm_context_analysis.py --limit 20

# Resume an interrupted run
python 04_llm_context_analysis.py --resume

# Use a more capable model
python 04_llm_context_analysis.py --model claude-sonnet-4-6
```

**Output:**
- `data/geoeconomic_context.json` — array of JSON objects, one per document

---

## Output figures

| File | Description |
|---|---|
| `figures/00_corpus_overview.png` | Corpus size by reporting period and event type |
| `figures/01_top_terms_corpus.png` | Top 30 terms after stopword removal |
| `figures/02_coherence_scores.png` | LDA coherence curve for K selection |
| `figures/03_topic_top_terms.png` | Top 10 terms per topic (grid) |
| `figures/04_executive_vs_analyst_topics.png` | Topic fingerprints by speaker role |
| `figures/05_topic_by_reporting_period.png` | Topic composition per reporting quarter |
| `figures/06_weekly_topic_trajectories.png` | Weekly topic trajectories (Q4 2025 season) |
| `figures/07_topic_heatmap_by_period.png` | Topic × period heatmap |
| `figures/08_topic_terms_dotplot.png` | β matrix dot plot |
| `figures/09_dominant_topic_by_period.png` | Dominant topic share per reporting period |
| `figures/10_georisk_match_counts.png` | Geoeconomic risk match counts by category |
| `figures/11_georisk_by_period.png` | Flagged documents by reporting period and category |
| `figures/12_georisk_top_companies.png` | Top 20 companies by geoeconomic risk mentions |
| `figures/13_georisk_cooccurrence.png` | Co-occurrence matrix across risk categories |

---

## Project structure

```
.
├── 00_prepare_corpus.R           # R: corpus assembly and speaker classification
├── 01_topic_modeling.ipynb       # Python: LDA topic modeling
├── 02_visualize_topics.R         # R: ggplot2 visualization
├── 03_geoeconomic_search.ipynb   # Python: dictionary-based geoeconomic risk search
├── 04_llm_context_analysis.py    # Python: LLM context classification (Claude API)
├── data/
│   ├── doc_topic_distributions.csv    # γ matrix (document–topic)
│   ├── topic_terms.csv                # β matrix (topic–term)
│   ├── role_topic_comparison.csv      # topic proportions by speaker role
│   ├── top_docs_per_topic.csv         # most representative docs per topic
│   ├── geoeconomic_matches.csv        # all docs with per-category flags
│   ├── geoeconomic_matches_flagged.csv # flagged docs only (1,626)
│   └── geoeconomic_context.json       # LLM context classification results
└── figures/                      # all PNG outputs (00–13)
```

Files generated at runtime (excluded from version control):
```
data/corpus_documents.csv        # regenerate with 00_prepare_corpus.R
data/corpus_speaker_level.csv    # regenerate with 00_prepare_corpus.R
```

---

## Notes

- The corpus was scraped in February–March 2026 and is therefore dominated by
  **Q4 2025 earnings calls** (~1,850 transcripts). The remaining ~450 calls
  reflect companies with non-calendar fiscal years (Q1–Q3 2026 designations).
- About 590 entries are **investor conference presentations** rather than
  quarterly earnings calls; they are included in the topic model but excluded
  from quarterly temporal comparisons.
- The Anaconda Python environment on the development machine has a NumPy 2.0
  incompatibility with older compiled packages. Always use the dedicated `.venv`
  (Python 3.10) from the scraper project directory.
- `04_llm_context_analysis.py` writes a rolling checkpoint every 50 documents
  so that a long run can be safely interrupted and resumed with `--resume`.
