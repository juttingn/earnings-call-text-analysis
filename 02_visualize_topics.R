# =============================================================================
# 02_visualize_topics.R
# Temporal and structural visualization of LDA topic model outputs
#
# Reads the document-topic distribution matrix (gamma) and topic-term
# matrix (beta) exported by 01_topic_modeling.ipynb, and produces a
# series of publication-quality figures saved to figures/.
#
# Figures produced:
#   figures/05_monthly_topic_prevalence.png  — stacked area chart
#   figures/06_topic_trajectories.png        — faceted monthly line chart
#   figures/07_topic_heatmap_monthly.png     — topic × month heatmap
#   figures/08_topic_terms_dotplot.png       — top terms per topic (dot plot)
#   figures/09_dominant_topic_calendar.png   — dominant topic share over time
# =============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(ggplot2)
library(forcats)
library(stringr)
library(tidytext)  # for reorder_within() and scale_y_reordered()

# --- Load data ----------------------------------------------------------------

gamma <- read_csv("data/doc_topic_distributions.csv",
                  show_col_types = FALSE)
beta  <- read_csv("data/topic_terms.csv",
                  show_col_types = FALSE)

cat("Gamma matrix:", nrow(gamma), "documents,",
    sum(str_starts(names(gamma), "topic_")), "topics\n")
cat("Beta matrix: ", nrow(beta), "rows\n")

# Identify topic columns
topic_cols <- names(gamma)[str_starts(names(gamma), "topic_")]
K <- length(topic_cols)

# Ensure date columns are proper types
gamma <- gamma %>%
  mutate(
    call_date        = as.Date(call_date),
    call_month       = as.Date(call_month),
    call_week        = as.Date(call_week),
    # Order reporting periods chronologically
    reporting_period = factor(reporting_period,
      levels = c("Q2 2025", "Q3 2025", "Q4 2025",
                 "Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"))
  )

# --- Helper: topic label lookup ----------------------------------------------
# If the Python notebook produced a "dominant_label" column we can use it;
# otherwise fall back to generic labels. Users should update TOPIC_LABELS
# in the Python notebook for meaningful labels here.

if ("dominant_label" %in% names(gamma) &&
    !all(str_starts(gamma$dominant_label, "Topic "))) {
  label_map <- gamma %>%
    select(dominant_topic, dominant_label) %>%
    distinct() %>%
    rename(topic_id = dominant_topic, label = dominant_label)
} else {
  label_map <- tibble(
    topic_id = 0:(K - 1),
    label    = paste0("Topic ", 0:(K - 1))
  )
}

cat("\nTopic labels:\n")
print(label_map)

# =============================================================================
# Figure 1: Stacked bar chart — topic composition by reporting period
# Primary temporal axis: the fiscal quarter reported (Q4 2025, Q1 2026, etc.)
# This is more analytically meaningful than publication month for this corpus.
# =============================================================================

# Pivot gamma to long format, join labels
gamma_long <- gamma %>%
  select(reporting_period, call_week, all_of(topic_cols)) %>%
  pivot_longer(cols = all_of(topic_cols),
               names_to = "topic_col",
               values_to = "probability") %>%
  mutate(topic_id = as.integer(str_extract(topic_col, "[0-9]+"))) %>%
  left_join(label_map, by = "topic_id")

# Aggregate by reporting period
period_topics <- gamma_long %>%
  filter(!is.na(reporting_period)) %>%
  group_by(reporting_period, topic_id, label) %>%
  summarise(mean_prob = mean(probability, na.rm = TRUE), .groups = "drop") %>%
  group_by(reporting_period) %>%
  mutate(share = mean_prob / sum(mean_prob)) %>%
  ungroup()

# Also compute by week for the within-season analysis
weekly_topics <- gamma_long %>%
  filter(!is.na(call_week)) %>%
  group_by(call_week, topic_id, label) %>%
  summarise(mean_prob = mean(probability, na.rm = TRUE), .groups = "drop") %>%
  group_by(call_week) %>%
  mutate(share = mean_prob / sum(mean_prob)) %>%
  ungroup() %>%
  # Keep only weeks with at least 10 documents to avoid sparse week artefacts
  left_join(
    gamma %>% filter(!is.na(call_week)) %>% count(call_week),
    by = "call_week"
  ) %>%
  filter(n >= 10)

n_topics <- length(unique(period_topics$topic_id))
topic_colors <- setNames(
  colorRampPalette(c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                     "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
                     "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
                     "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"))(n_topics),
  unique(period_topics$label)
)

# Document counts per period (for annotation)
period_counts <- gamma %>%
  filter(!is.na(reporting_period)) %>%
  count(reporting_period)

p1 <- ggplot(period_topics,
             aes(x = reporting_period, y = share, fill = label)) +
  geom_bar(stat = "identity", position = "stack",
           colour = "white", linewidth = 0.3) +
  geom_text(
    data = period_counts,
    aes(x = reporting_period, y = 1.03, label = paste0("n=", n)),
    inherit.aes = FALSE, size = 3, colour = "grey40"
  ) +
  scale_fill_manual(values = topic_colors, name = "Topic") +
  scale_y_continuous(labels = scales::percent, expand = c(0, 0),
                     limits = c(0, 1.08)) +
  labs(
    title    = "Topic Composition by Reporting Period",
    subtitle = "Mean topic probability share per fiscal quarter reported",
    x        = "Reporting period (quarter & year)",
    y        = "Share of topic probability"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position  = "right",
    legend.key.size  = unit(0.4, "cm"),
    legend.text      = element_text(size = 8),
    axis.text.x      = element_text(angle = 30, hjust = 1),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey40", size = 10)
  )

ggsave("figures/05_topic_by_reporting_period.png", p1,
       width = 10, height = 6, dpi = 150, bg = "white")
cat("Saved figures/05_topic_by_reporting_period.png\n")

# =============================================================================
# Figure 2: Weekly topic trajectories within the Q4 2025 reporting season
# Publication date (week of posting) tracks the cadence of Q4 2025 reporting.
# =============================================================================

if (nrow(weekly_topics) > 0) {

  p2 <- ggplot(weekly_topics,
               aes(x = call_week, y = share, colour = label, group = label)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.8) +
    scale_colour_manual(values = topic_colors, name = "Topic") +
    scale_x_date(date_breaks = "1 week", date_labels = "%d %b",
                 expand = c(0.02, 0)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title    = "Weekly Topic Trajectories — Q4 2025 Reporting Season",
      subtitle = "Topic share by week of transcript publication (weeks with ≥10 transcripts)",
      x        = "Week of publication",
      y        = "Share of topic probability"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position  = "right",
      legend.key.size  = unit(0.4, "cm"),
      legend.text      = element_text(size = 8),
      axis.text.x      = element_text(angle = 45, hjust = 1),
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(colour = "grey40", size = 10)
    )

  ggsave("figures/06_weekly_topic_trajectories.png", p2,
         width = 13, height = 6, dpi = 150, bg = "white")
  cat("Saved figures/06_weekly_topic_trajectories.png\n")
}

# =============================================================================
# Figure 3: Heatmap — topic × reporting period
# =============================================================================

heatmap_data <- period_topics %>%
  filter(!is.na(reporting_period))

p3 <- ggplot(heatmap_data,
             aes(x = reporting_period,
                 y = fct_reorder(label, -topic_id),
                 fill = mean_prob)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = scales::percent(mean_prob, accuracy = 0.1)),
            size = 2.5, colour = ifelse(heatmap_data$mean_prob > 0.07,
                                        "white", "grey30")) +
  scale_fill_gradientn(
    colours = c("#EFF3FF", "#6BAED6", "#08519C"),
    name    = "Mean\nprobability"
  ) +
  labs(
    title    = "Topic–Period Prevalence Heatmap",
    subtitle = "Mean document-topic probability by reporting quarter",
    x        = "Reporting period",
    y        = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x     = element_text(angle = 30, hjust = 1, size = 9),
    axis.text.y     = element_text(size = 9),
    panel.grid      = element_blank(),
    legend.position = "right",
    plot.title      = element_text(face = "bold", size = 13),
    plot.subtitle   = element_text(colour = "grey40", size = 10)
  )

ggsave("figures/07_topic_heatmap_by_period.png", p3,
       width = max(8, nlevels(heatmap_data$reporting_period) * 1.3 + 3),
       height = n_topics * 0.55 + 2,
       dpi = 150, bg = "white")
cat("Saved figures/07_topic_heatmap_by_period.png\n")

# =============================================================================
# Figure 4: Dot plot — top terms per topic (beta matrix)
# A lollipop/dot plot showing the probability of the top N terms in each topic.
# =============================================================================

N_TERMS <- 10  # terms to show per topic

top_terms <- beta %>%
  left_join(label_map, by = c("topic" = "topic_id")) %>%
  rename(label = label.y) %>%
  group_by(label) %>%
  slice_max(probability, n = N_TERMS) %>%
  ungroup() %>%
  mutate(label = fct_reorder(label, topic),
         term  = reorder_within(term, probability, label))

p4 <- ggplot(top_terms, aes(x = probability, y = term, colour = label)) +
  geom_segment(aes(x = 0, xend = probability, yend = term),
               colour = "grey80", linewidth = 0.6) +
  geom_point(size = 2.2) +
  facet_wrap(~ label, scales = "free_y", ncol = 4) +
  scale_y_reordered() +
  scale_colour_manual(values = topic_colors, guide = "none") +
  labs(
    title    = paste0("Top ", N_TERMS, " Terms per LDA Topic"),
    subtitle = "Term probability within each topic (β)",
    x        = "β (term probability within topic)",
    y        = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(
    strip.text       = element_text(face = "bold", size = 8),
    axis.text.x      = element_text(size = 7),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey40", size = 10)
  )

ggsave("figures/08_topic_terms_dotplot.png", p4,
       width = 14,
       height = ceiling(n_topics / 4) * (N_TERMS * 0.32 + 1.5),
       dpi = 150, bg = "white")
cat("Saved figures/08_topic_terms_dotplot.png\n")

# =============================================================================
# Figure 5: Dominant topic share over time (bar chart by dominant topic)
# Counts how many documents have each topic as dominant in each month.
# Complements the probability-based charts above.
# =============================================================================

if ("dominant_topic" %in% names(gamma)) {

  dominant_by_period <- gamma %>%
    filter(!is.na(reporting_period)) %>%
    left_join(label_map, by = c("dominant_topic" = "topic_id")) %>%
    group_by(reporting_period, label) %>%
    summarise(n_docs = n(), .groups = "drop") %>%
    group_by(reporting_period) %>%
    mutate(share = n_docs / sum(n_docs)) %>%
    ungroup()

  p5 <- ggplot(dominant_by_period,
               aes(x = reporting_period, y = share, fill = label)) +
    geom_bar(stat = "identity", position = "stack",
             colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = topic_colors, name = "Dominant topic") +
    scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
    labs(
      title    = "Dominant Topic Distribution by Reporting Period",
      subtitle = "Share of transcripts where each topic has the highest probability",
      x        = "Reporting period",
      y        = "Share of transcripts"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position  = "right",
      legend.key.size  = unit(0.4, "cm"),
      legend.text      = element_text(size = 8),
      axis.text.x      = element_text(angle = 30, hjust = 1),
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(colour = "grey40", size = 10)
    )

  ggsave("figures/09_dominant_topic_by_period.png", p5,
         width = 10, height = 6, dpi = 150, bg = "white")
  cat("Saved figures/09_dominant_topic_by_period.png\n")
}

# =============================================================================
# Summary table: top 5 documents per topic
# Useful for manually validating topic labels by reading representative calls.
# =============================================================================

if ("dominant_topic" %in% names(gamma)) {

  top_docs <- gamma %>%
    left_join(label_map, by = c("dominant_topic" = "topic_id")) %>%
    group_by(dominant_topic, label) %>%
    slice_max(dominant_prob, n = 5) %>%
    select(label, company_name, ticker, call_date, dominant_prob) %>%
    arrange(dominant_topic, desc(dominant_prob)) %>%
    ungroup()

  write_csv(top_docs, "data/top_docs_per_topic.csv")
  cat("\nSaved data/top_docs_per_topic.csv\n")
  cat("\nTop representative documents per topic:\n")
  print(top_docs, n = Inf)
}

cat("\nAll figures saved to figures/. Done.\n")
