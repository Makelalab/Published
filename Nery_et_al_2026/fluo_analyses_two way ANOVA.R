# =============================================================
# Statistical analysis: Inducible ClpX-HaloTag expression
# Two-way ANOVA (genotype x galactose) + Tukey HSD post-hoc
# =============================================================

# ---- 0. Install/load required packages ----
required_packages <- c("dplyr", "ggplot2", "ggsignif")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(dplyr)
library(ggplot2)
library(ggsignif)

# ---- 1. Load your data ----
# Expected format: a data frame / CSV with columns:
#   genotype  : "clpx" or "control"
#   galactose : "0", "0.25", "0.5", "1"   (as factor/character, not numeric)
#   bio_rep   : biological replicate ID (1, 2, 3)
#   tech_rep  : technical replicate ID (1, 2, 3) -- repeated measurement of
#               the SAME biological sample
#   fluor_norm: normalized fluorescence (RFU/OD600), one row per technical
#               measurement
#
# Example of loading from CSV:
# df_raw <- read.csv("your_data.csv")

# Example synthetic structure (REPLACE with your real data):
# 2 genotypes x 4 galactose levels x 3 biological reps x 3 technical reps = 72 rows
set.seed(1)
df_raw <- expand.grid(
  genotype  = c("clpx", "control"),
  galactose = c("0", "0.25", "0.5", "1"),
  bio_rep   = 1:3,
  tech_rep  = 1:3
)

# Assign realistic fake fluorescence values (REPLACE with your real measurements)
base_means <- c(
  "clpx.0" = 28, "clpx.0.25" = 81, "clpx.0.5" = 103, "clpx.1" = 102,
  "control.0" = 1.5, "control.0.25" = 1.5, "control.0.5" = 1.5, "control.1" = 1.5
)
df_raw$fluor_norm <- mapply(function(g, gal) {
  key <- paste(g, gal, sep = ".")
  rnorm(1, mean = base_means[key], sd = ifelse(g == "clpx", 4, 0.5))
}, df_raw$genotype, df_raw$galactose)

df_raw$genotype  <- factor(df_raw$genotype, levels = c("control", "clpx"))
df_raw$galactose <- factor(df_raw$galactose, levels = c("0", "0.25", "0.5", "1"))
df_raw$bio_rep    <- factor(df_raw$bio_rep)
df_raw$tech_rep   <- factor(df_raw$tech_rep)

# ---- 1b. Aggregate technical replicates within each biological replicate ----
# IMPORTANT: statistical tests (ANOVA, Tukey HSD) should be run on the
# biological-replicate means, NOT on all technical replicates individually,
# to avoid pseudoreplication (treating technical replicates as if they were
# independent biological observations inflates the sample size and produces
# artificially small p-values).
df <- df_raw %>%
  group_by(genotype, galactose, bio_rep) %>%
  summarise(fluor_norm = mean(fluor_norm), .groups = "drop")

# 'df' now has n = 3 (one value per biological replicate) per genotype x
# galactose combination, and is what feeds into the ANOVA/Tukey HSD below.
# 'df_raw' (all technical replicates) is kept separately for plotting.

# ---- 2. Two-way ANOVA (genotype x galactose) ----
model_2way <- aov(fluor_norm ~ genotype * galactose, data = df)
cat("\n===== Two-way ANOVA (genotype x galactose) =====\n")
print(summary(model_2way))

# ---- 3. Tukey HSD post-hoc, within clpx only (dose-response) ----
df_clpx <- df %>% filter(genotype == "clpx")
model_1way_clpx <- aov(fluor_norm ~ galactose, data = df_clpx)

tukey_result <- TukeyHSD(model_1way_clpx)
cat("\n===== Tukey HSD (within clpx, pairwise galactose levels) =====\n")
print(tukey_result)

# ---- 4. Summary table of means +/- SD per group ----
summary_table <- df %>%
  group_by(genotype, galactose) %>%
  summarise(
    mean_fluor = mean(fluor_norm),
    sd_fluor   = sd(fluor_norm),
    n          = n(),
    .groups = "drop"
  )

cat("\n===== Summary table (mean +/- SD per group) =====\n")
print(summary_table)

# ---- 5. Publication-quality plot ----
# Bars: mean +/- SD calculated from BIOLOGICAL replicate means (df, n=3)
# Points: ALL technical replicates (df_raw), colored by biological replicate,
#         to visually show both biological and technical variability

# Convert galactose to a numeric-labelled factor for correct axis ordering
df$galactose     <- factor(df$galactose, levels = c("0", "0.25", "0.5", "1"),
                            labels = c("0%", "0.25%", "0.5%", "1%"))
df_raw$galactose <- factor(df_raw$galactose, levels = c("0", "0.25", "0.5", "1"),
                            labels = c("0%", "0.25%", "0.5%", "1%"))

# Summary table (mean +/- SD across biological replicates, n=3)
summary_table <- df %>%
  group_by(genotype, galactose) %>%
  summarise(
    mean_fluor = mean(fluor_norm),
    sd_fluor   = sd(fluor_norm),
    n          = n(),
    .groups = "drop"
  )

cat("\n===== Summary table (mean +/- SD across biological replicates) =====\n")
print(summary_table)

genotype_labels <- c("control" = "Control", "clpx" = "ClpX-HaloTag")
bio_rep_colors  <- c("1" = "#E69F00", "2" = "#56B4E9", "3" = "#009E73")  # colorblind-friendly

p <- ggplot() +
  # Bars: mean fluorescence per group (from biological replicate means)
  geom_bar(
    data = summary_table,
    aes(x = galactose, y = mean_fluor, fill = genotype),
    stat = "identity", position = position_dodge(width = 0.75),
    width = 0.65, color = "black", linewidth = 0.4, alpha = 0.5
  ) +
  # Error bars: mean +/- SD (across biological replicates)
  geom_errorbar(
    data = summary_table,
    aes(x = galactose, ymin = mean_fluor - sd_fluor, ymax = mean_fluor + sd_fluor,
        group = genotype),
    position = position_dodge(width = 0.75), width = 0.2, linewidth = 0.4
  ) +
  # Individual technical replicate points, jittered within each group,
  # colored by biological replicate
  geom_point(
    data = df_raw,
    aes(x = galactose, y = fluor_norm, group = genotype, color = bio_rep),
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.75),
    size = 2, alpha = 0.85, stroke = 0.3
  ) +
  scale_fill_manual(
    values = c("control" = "#B0B0B0", "clpx" = "#2E86AB"),
    labels = genotype_labels,
    name = NULL
  ) +
  scale_color_manual(
    values = bio_rep_colors,
    name = "Biological\nreplicate"
  ) +
  labs(
    x = "Galactose concentration",
    y = expression(paste("Normalized fluorescence (RFU / OD"[600], ")")),
    title = NULL
  ) +
  theme_classic(base_size = 13) +
  theme(
    axis.text = element_text(color = "black"),
    axis.title = element_text(face = "plain"),
    legend.position = "right",
    legend.background = element_blank(),
    legend.key = element_blank(),
    plot.margin = margin(10, 15, 10, 10)
  ) +
  guides(fill = guide_legend(order = 1), color = guide_legend(order = 2))

# ---- 5b. Add significance brackets based on Tukey HSD results (within clpx) ----
# Brackets compare galactose levels WITHIN the clpx strain only (since that is
# where the Tukey HSD post-hoc test was performed). Update the comparisons,
# labels, and y-positions below to match your actual Tukey HSD p-values.
#
# Convention used for annotations:
#   p <= 0.001  -> "***"
#   p <= 0.01   -> "**"
#   p <= 0.05   -> "*"
#   p > 0.05    -> "ns"
#
# IMPORTANT CAVEAT: because bars are dodged by genotype, geom_signif's
# default bracket placement (which works off the x-axis category, not the
# dodged sub-bar) may need manual x-position tweaking to sit exactly above
# the clpx bars rather than spanning the full category width. If brackets
# look misaligned when you run this, either (a) nudge the bracket x-range
# manually, or (b) skip geom_signif and add brackets/asterisks in
# Illustrator/Inkscape/PowerPoint after exporting the PDF -- this is common
# practice for dodged-bar figures and often faster than fighting the layout.

max_y <- max(df_raw$fluor_norm, na.rm = TRUE)

p <- p +
  geom_signif(
    data = data.frame(genotype = "clpx"),
    comparisons = list(c("0%", "0.25%")),
    annotations = "**",           # non-induced vs 0.25%, p_adj <= 0.006
    y_position = max_y * 1.08,
    tip_length = 0.01, textsize = 4, vjust = 0.4
  ) +
  geom_signif(
    data = data.frame(genotype = "clpx"),
    comparisons = list(c("0%", "0.5%")),
    annotations = "**",           # non-induced vs 0.5%, p_adj <= 0.006
    y_position = max_y * 1.18,
    tip_length = 0.01, textsize = 4, vjust = 0.4
  ) +
  geom_signif(
    data = data.frame(genotype = "clpx"),
    comparisons = list(c("0%", "1%")),
    annotations = "**",           # non-induced vs 1%, p_adj <= 0.006
    y_position = max_y * 1.28,
    tip_length = 0.01, textsize = 4, vjust = 0.4
  ) +
  geom_signif(
    data = data.frame(genotype = "clpx"),
    comparisons = list(c("0.5%", "1%")),
    annotations = "ns",           # 0.5% vs 1%, p = 0.999
    y_position = max_y * 1.08,
    tip_length = 0.01, textsize = 3.5, vjust = -0.1
  ) +
  # Extend y-axis so brackets are not cut off
  coord_cartesian(ylim = c(0, max_y * 1.4), clip = "off")

print(p)

# Save at publication resolution (300 dpi), sized for a single-column figure
ggsave(
  filename = "clpx_fluorescence_induction.png",
  plot = p, width = 5.5, height = 4.6, units = "in", dpi = 300
)
ggsave(
  filename = "clpx_fluorescence_induction.pdf",
  plot = p, width = 5.5, height = 4.6, units = "in"
)

# =============================================================
# NOTES:
# - Replace the synthetic 'df_raw' above (Section 1) with your real data,
#   e.g. df_raw <- read.csv("your_file.csv")
# - Required columns: genotype, galactose, bio_rep, tech_rep, fluor_norm
#   (one row per technical measurement -- 3 technical reps per biological
#   rep, 3 biological reps per genotype x galactose combination)
# - Statistics (ANOVA, Tukey HSD) run on biological replicate means (n=3)
#   to avoid pseudoreplication; the plot shows all technical replicate
#   points, colored by biological replicate, for full transparency
# =============================================================
