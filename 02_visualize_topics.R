# =============================================================================
# 02_visualize_topics.R
# Structural and temporal visualization of LDA topic model outputs
#
# Produces:
#   figures/07_topic_heatmap_by_period.png   — topic × period heatmap
#   figures/09_dominant_topic_by_period.png  — dominant topic share per quarter
# =============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(ggplot2)
library(forcats)
library(stringr)
library(scales)

# --- Load data ----------------------------------------------------------------

gamma <- read_csv("data/doc_topic_distributions.csv", show_col_types = FALSE)
beta  <- read_csv("data/topic_terms.csv",              show_col_types = FALSE)

cat("Gamma matrix:", nrow(gamma), "documents,",
    sum(str_starts(names(gamma), "topic_")), "topics\n")
cat("Beta matrix: ", nrow(beta), "rows\n")

topic_cols <- names(gamma)[str_starts(names(gamma), "topic_")]
K <- length(topic_cols)

# All possible reporting periods in chronological order
# Q1 2025 has no transcripts in this corpus; start from Q2 2025
PERIOD_LEVELS <- c("Q2 2025", "Q3 2025", "Q4 2025",
                   "Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026")

gamma <- gamma %>%
  mutate(
    call_date        = as.Date(call_date),
    call_week        = as.Date(call_week),
    reporting_period = factor(reporting_period, levels = PERIOD_LEVELS)
  )

# Restrict to periods actually present in data
available_periods <- levels(droplevels(
  gamma$reporting_period[!is.na(gamma$reporting_period)]
))
cat("Reporting periods present:", paste(available_periods, collapse = ", "), "\n")

# --- Topic label lookup -------------------------------------------------------

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

# --- Custom ggplot2 theme -----------------------------------------------------

theme_earnings <- function(base_size = 11) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 2,
                                      hjust = 0, margin = margin(b = 4)),
      plot.subtitle    = element_text(colour = "#555555", size = base_size - 0.5,
                                      hjust = 0, lineheight = 1.3,
                                      margin = margin(b = 12)),
      plot.caption     = element_text(colour = "#888888", size = base_size - 2,
                                      hjust = 0, margin = margin(t = 8),
                                      face = "italic"),
      axis.text        = element_text(colour = "#333333", size = base_size - 1),
      axis.title       = element_text(colour = "#222222", size = base_size),
      legend.text      = element_text(size = base_size - 2),
      legend.title     = element_text(face = "bold", size = base_size - 1),
      legend.key.size  = unit(0.42, "cm"),
      panel.grid.major = element_line(colour = "#EBEBEB", linewidth = 0.45),
      panel.grid.minor = element_blank(),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      strip.text       = element_text(face = "bold", size = base_size - 1,
                                      colour = "#222222"),
      plot.margin      = margin(14, 16, 12, 14)
    )
}

# --- Colour palette -----------------------------------------------------------

n_topics <- length(unique(label_map$label))
topic_colors_base <- c(
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
  "#c49c94","#f7b6d2","#dbdb8d","#9edae5","#393b79"
)

# --- Pivot to long format and aggregate by reporting period ------------------

gamma_long <- gamma %>%
  select(reporting_period, all_of(topic_cols)) %>%
  pivot_longer(cols = all_of(topic_cols),
               names_to  = "topic_col",
               values_to = "probability") %>%
  mutate(topic_id = as.integer(str_extract(topic_col, "[0-9]+"))) %>%
  left_join(label_map, by = "topic_id")

period_topics <- gamma_long %>%
  filter(!is.na(reporting_period)) %>%
  group_by(reporting_period, topic_id, label) %>%
  summarise(mean_prob = mean(probability, na.rm = TRUE), .groups = "drop") %>%
  group_by(reporting_period) %>%
  mutate(share = mean_prob / sum(mean_prob)) %>%
  ungroup()

# Document counts per period — used for n= labels
period_counts <- gamma %>%
  filter(!is.na(reporting_period)) %>%
  count(reporting_period, name = "n")

n_topics_actual <- length(unique(period_topics$label))
topic_colors <- setNames(
  colorRampPalette(topic_colors_base)(n_topics_actual),
  unique(period_topics$label)
)

# =============================================================================
# Figure 0: Topic composition stacked bar  (all available reporting periods)
# =============================================================================

stacked_data <- period_topics %>%
  filter(!is.na(reporting_period)) %>%
  left_join(period_counts, by = "reporting_period") %>%
  mutate(
    period_label = paste0(as.character(reporting_period), "\n(n=", n, ")"),
    period_label = factor(
      period_label,
      levels = unique(period_label[order(reporting_period)])
    )
  )

p0 <- ggplot(stacked_data,
             aes(x = period_label,
                 y = share,
                 fill = fct_reorder(label, topic_id))) +
  geom_bar(stat = "identity", position = "stack",
           colour = "white", linewidth = 0.25) +
  scale_fill_manual(values = topic_colors, name = "Topic") +
  scale_y_continuous(labels = scales::percent,
                     expand = expansion(mult = c(0, 0.01))) +
  labs(
    title    = "Topic Composition by Reporting Period",
    subtitle = "Mean topic probability share per fiscal quarter reported",
    x        = "Reporting period (quarter & year)",
    y        = "Share of topic probability",
    caption  = "The corpus is dominated by Q4 2025 (~1,850 transcripts). Cross-quarter comparisons are illustrative only."
  ) +
  theme_earnings(base_size = 11) +
  theme(
    axis.text.x     = element_text(angle = 30, hjust = 1, lineheight = 1.2),
    legend.position = "right"
  )

n_periods_stacked <- length(unique(stacked_data$period_label))
ggsave("figures/05_topic_by_reporting_period.png", p0,
       width  = max(10, n_periods_stacked * 1.6 + 3),
       height = 6.5,
       dpi = 160, bg = "white")
cat("Saved figures/05_topic_by_reporting_period.png\n")

# =============================================================================
# Figure 1: Topic × period heatmap  (all available reporting periods)
# =============================================================================

heatmap_data <- period_topics %>%
  filter(!is.na(reporting_period)) %>%
  left_join(period_counts, by = "reporting_period") %>%
  mutate(
    period_label = paste0(as.character(reporting_period), "\n(n=", n, ")"),
    period_label = factor(
      period_label,
      levels = unique(period_label[order(reporting_period)])
    )
  )

p1 <- ggplot(heatmap_data,
             aes(x = period_label,
                 y = fct_reorder(label, -topic_id),
                 fill = mean_prob)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(
    aes(label = scales::percent(mean_prob, accuracy = 0.1)),
    size   = 2.4,
    colour = ifelse(heatmap_data$mean_prob > 0.07, "white", "#444444")
  ) +
  scale_fill_gradientn(
    colours = c("#EFF3FF", "#6BAED6", "#2171B5", "#084594"),
    name    = "Mean\nprobability",
    labels  = scales::percent_format(accuracy = 1)
  ) +
  labs(
    title    = "Topic–Period Prevalence Heatmap",
    subtitle = "Mean document-topic probability by reporting quarter\nAll available quarters shown; n= gives transcript count per quarter",
    x        = NULL,
    y        = NULL,
    caption  = "The corpus is dominated by Q4 2025 (~1,850 transcripts). Cross-quarter comparisons are illustrative only."
  ) +
  theme_earnings(base_size = 10) +
  theme(
    axis.text.x    = element_text(angle = 0, hjust = 0.5, size = 9,
                                  lineheight = 1.2),
    axis.text.y    = element_text(size = 9),
    panel.grid     = element_blank(),
    legend.position = "right"
  )

n_periods_plot <- length(unique(heatmap_data$period_label))
ggsave("figures/07_topic_heatmap_by_period.png", p1,
       width  = max(9, n_periods_plot * 1.7 + 3),
       height = n_topics_actual * 0.52 + 2.5,
       dpi = 160, bg = "white")
cat("Saved figures/07_topic_heatmap_by_period.png\n")

# =============================================================================
# Figure 2: Dominant topic share per reporting period  (all available periods)
# =============================================================================

if ("dominant_topic" %in% names(gamma)) {

  dominant_by_period <- gamma %>%
    filter(!is.na(reporting_period)) %>%
    left_join(label_map, by = c("dominant_topic" = "topic_id")) %>%
    left_join(period_counts, by = "reporting_period") %>%
    group_by(reporting_period, label, n) %>%
    summarise(n_docs = n(), .groups = "drop") %>%
    group_by(reporting_period) %>%
    mutate(share = n_docs / sum(n_docs)) %>%
    ungroup() %>%
    mutate(
      period_label = paste0(as.character(reporting_period), "\n(n=", n, ")"),
      period_label = factor(
        period_label,
        levels = unique(period_label[order(reporting_period)])
      )
    )

  p2 <- ggplot(dominant_by_period,
               aes(x = period_label, y = share, fill = label)) +
    geom_bar(stat = "identity", position = "stack",
             colour = "white", linewidth = 0.3) +
    scale_fill_manual(values = topic_colors, name = "Dominant\ntopic") +
    scale_y_continuous(labels = scales::percent,
                       expand = expansion(mult = c(0, 0.01))) +
    labs(
      title    = "Dominant Topic Distribution by Reporting Period",
      subtitle = "Share of transcripts where each topic has the highest probability\nAll available quarters shown; n= gives transcript count per quarter",
      x        = NULL,
      y        = "Share of transcripts",
      caption  = "The corpus is dominated by Q4 2025 (~1,850 transcripts). Cross-quarter comparisons are illustrative only."
    ) +
    theme_earnings(base_size = 11) +
    theme(
      axis.text.x    = element_text(angle = 0, hjust = 0.5, lineheight = 1.2),
      legend.position = "right"
    )

  ggsave("figures/09_dominant_topic_by_period.png", p2,
         width  = max(10, n_periods_plot * 1.6 + 3),
         height = 6.5,
         dpi = 160, bg = "white")
  cat("Saved figures/09_dominant_topic_by_period.png\n")
}

# =============================================================================
# Summary table: top 5 documents per topic
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
