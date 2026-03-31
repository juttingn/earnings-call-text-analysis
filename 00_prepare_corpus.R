# =============================================================================
# 00_prepare_corpus.R
# Corpus preparation for topic modeling of earnings call transcripts
#
# Reads all individual transcript RDS files from transcripts_raw/,
# aggregates text at the document (transcript) level, classifies speakers
# by role type, and exports two CSVs for downstream Python analysis.
#
# Outputs:
#   data/corpus_documents.csv      — one row per transcript (for LDA)
#   data/corpus_speaker_level.csv  — one row per speaker turn (for role analysis)
# =============================================================================

library(dplyr)
library(stringr)
library(lubridate)
library(readr)

# --- Configuration -----------------------------------------------------------

RAW_DIR  <- "transcripts_raw"
DATA_DIR <- "data"

# --- Load all RDS files -------------------------------------------------------

cat("Loading RDS files...\n")

files <- list.files(RAW_DIR, pattern = "[.]rds$", full.names = TRUE)
cat("  Found", length(files), "files\n")

# Read each file, adding slug identifier derived from filename
corpus_raw <- lapply(files, function(f) {
  df <- tryCatch(readRDS(f), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) return(NULL)
  df$slug <- tools::file_path_sans_ext(basename(f))
  df
})

# Remove NULL entries (failed reads) and bind
corpus_raw <- bind_rows(Filter(Negate(is.null), corpus_raw))
cat("  Loaded", nrow(corpus_raw), "speaker turns across",
    length(unique(corpus_raw$slug)), "transcripts\n")

# --- Classify speaker roles --------------------------------------------------
# We distinguish three communities in earnings call discourse:
#   "executive" — company management (CEO, CFO, COO, President, etc.)
#   "analyst"   — sell-side analysts asking questions
#   "operator"  — call operator / moderator (formulaic, non-substantive)

classify_role <- function(role) {
  if (is.na(role)) return("other")
  role_lc <- tolower(role)

  if (str_detect(role_lc, "operator|moderator")) {
    return("operator")
  } else if (str_detect(role_lc, "analyst|research|coverage")) {
    return("analyst")
  } else if (str_detect(role_lc,
    "chief|president|ceo|cfo|coo|cto|chairman|director|officer|founder|partner|head|vice|svp|evp|vp|treasurer|secretary|general counsel")) {
    return("executive")
  } else if (str_detect(role_lc, "investor relation")) {
    return("ir")  # investor relations — facilitators, not analysts or execs
  } else {
    return("other")
  }
}

corpus_raw <- corpus_raw %>%
  mutate(role_type = sapply(speaker_role, classify_role))

cat("  Role distribution:\n")
print(table(corpus_raw$role_type))

# --- Speaker-level export (for role comparison analysis) ---------------------

corpus_speaker <- corpus_raw %>%
  select(slug, url, company_name, ticker, quarter, call_year, call_date,
         speaker, speaker_role, role_type, text) %>%
  filter(!is.na(text), nchar(trimws(text)) > 20)  # drop near-empty turns

write_csv(corpus_speaker,
          file.path(DATA_DIR, "corpus_speaker_level.csv"))
cat("  Saved corpus_speaker_level.csv:", nrow(corpus_speaker), "rows\n")

# --- Document-level aggregation ----------------------------------------------
# One row per transcript: concatenate all text, plus separate executive/analyst text.
# We also compute a word count and a document date for temporal analysis.

corpus_docs <- corpus_raw %>%
  filter(!is.na(text)) %>%
  group_by(slug, url, company_name, ticker, quarter, call_year, call_date) %>%
  summarise(
    # Full transcript text (all speakers except operator)
    full_text    = paste(text[role_type != "operator"], collapse = " "),
    # Executive speech only
    exec_text    = paste(text[role_type == "executive"], collapse = " "),
    # Analyst questions only
    analyst_text = paste(text[role_type == "analyst"],   collapse = " "),
    # Metadata
    n_speaker_turns = n(),
    n_exec_turns    = sum(role_type == "executive"),
    n_analyst_turns = sum(role_type == "analyst"),
    .groups = "drop"
  ) %>%
  mutate(
    # Word count as a document-length proxy
    n_words = str_count(full_text, "\\S+"),
    # Publication month (when the transcript appeared online)
    call_month = floor_date(as.Date(call_date), "month"),
    # Week within the reporting season (for granular temporal analysis)
    call_week  = floor_date(as.Date(call_date), "week"),
    # Reporting period = the fiscal quarter being discussed
    # Format: "Q4 2025", "Q1 2026", etc. NA for conferences / non-quarterly events
    reporting_period = if_else(
      !is.na(quarter) & !is.na(call_year),
      paste(quarter, call_year),
      NA_character_
    ),
    # Event type: earnings call vs investor conference
    event_type = if_else(
      !is.na(quarter) & !is.na(call_year),
      "earnings_call",
      "investor_conference"
    )
  ) %>%
  # Keep only documents with a minimum amount of text
  filter(n_words >= 100)

cat("\nDocument-level corpus:\n")
cat("  Documents:", nrow(corpus_docs), "\n")
cat("  Date range:", as.character(min(corpus_docs$call_date, na.rm = TRUE)),
    "to", as.character(max(corpus_docs$call_date, na.rm = TRUE)), "\n")
cat("  Median words per transcript:", median(corpus_docs$n_words, na.rm = TRUE), "\n")
cat("  Reporting period distribution:\n")
print(sort(table(corpus_docs$reporting_period, useNA = "ifany"), decreasing = TRUE))
cat("  Event type distribution:\n")
print(table(corpus_docs$event_type, useNA = "ifany"))

write_csv(corpus_docs,
          file.path(DATA_DIR, "corpus_documents.csv"))
cat("\nSaved data/corpus_documents.csv:", nrow(corpus_docs), "documents\n")
cat("Done.\n")
