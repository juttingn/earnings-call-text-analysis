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

1. **Broad topic exploration** (this repository, Part 1): apply unsupervised
   topic modeling (LDA) to the full corpus to map the latent topical structure
   of earnings call discourse, understand which sectors dominate, how topics
   are distributed across the Q4 2025 reporting season, and whether executive
   speech and analyst questions reflect distinct topical priorities.

2. **Geoeconomic risk deep-dive** (Part 2, forthcoming): zoom in on the topics
   and documents most associated with geoeconomic language — trade policy,
   supply chain disruption, geopolitical uncertainty, sanctions exposure — and
   track how the salience of these themes has evolved across companies and
   quarters. This will extend the Master Thesis analysis to a richer linguistic
   and temporal window than was previously available.

---

## Pipeline

The analysis uses a combined **R + Python** workflow:

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
   figures/  (PNG outputs)
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

The main analysis notebook. Preprocesses the corpus (tokenization, custom
earnings-call stopword removal, lemmatization), fits an LDA model using
[gensim](https://radimrehurek.com/gensim/), selects the optimal number of
topics via coherence scoring, and compares the topic footprints of executive
and analyst speech.

**Requires:** the `.venv` Python 3.10 environment from the scraper project
(or any environment with `gensim`, `nltk`, `pandas`, `matplotlib`, `scikit-learn`).

**Outputs:**
- `data/doc_topic_distributions.csv` — γ matrix: per-document topic proportions
- `data/topic_terms.csv` — β matrix: top 30 terms per topic with probabilities
- `data/role_topic_comparison.csv` — topic proportions by speaker role

**Run in Jupyter:**
```bash
source /path/to/.venv/bin/activate
jupyter lab 01_topic_modeling.ipynb
```

Or execute non-interactively:
```bash
jupyter nbconvert --to notebook --execute --inplace 01_topic_modeling.ipynb
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

---

## Project structure

```
.
├── 00_prepare_corpus.R          # R: corpus assembly and speaker classification
├── 01_topic_modeling.ipynb      # Python: LDA topic modeling
├── 02_visualize_topics.R        # R: ggplot2 visualization
├── data/
│   ├── doc_topic_distributions.csv   # γ matrix (document–topic)
│   ├── topic_terms.csv               # β matrix (topic–term)
│   ├── role_topic_comparison.csv     # topic proportions by speaker role
│   └── top_docs_per_topic.csv        # most representative docs per topic
└── figures/                     # all PNG outputs
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
