# Paired comparison of the fusion variants against the baseline.
#
# Every variant runs on the SAME subsample within a repeat, so the runs are
# paired: comparing means alone hides that the per-repeat spread (f1_std around
# 0.13 at n=100) is much larger than the effect we are looking for. What matters
# is how often the prior wins on the same draw, and whether the mean of the
# per-repeat differences is distinguishable from zero.
#
# Note on power: with only 5 paired repeats the smallest attainable two-sided
# Wilcoxon p-value is 2/2^5 = 0.0625, so no result can reach 0.05 no matter how
# consistent it is. At least 6 repeats are needed, and 8-10 for comfort.
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\paired_test_bnlearn.R --tag=long

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

project_root <- "E:/Causal-LLM"
tag <- get_arg("tag", "centered")
baseline_variant <- get_arg("baseline", "baseline")

path <- sprintf("%s/results/bnlearn_fusion_detail_%s.csv", project_root, tag)

if (!file.exists(path)) {
  cat("missing", path, "\n")
  quit(status = 1)
}

detail <- read.csv(path, stringsAsFactors = FALSE)

# Average over targets within each repeat, so a repeat is one observation.
per_repeat <- aggregate(
  f1 ~ network + sample_size + beta + variant + repeat_index,
  data = detail, FUN = mean
)

base <- per_repeat[per_repeat$variant == baseline_variant, ]
keys <- unique(per_repeat[
  per_repeat$variant != baseline_variant,
  c("network", "sample_size", "beta", "variant")
])

results <- data.frame()

for (i in seq_len(nrow(keys))) {
  key <- keys[i, ]

  treated <- per_repeat[
    per_repeat$network == key$network &
    per_repeat$sample_size == key$sample_size &
    per_repeat$beta == key$beta &
    per_repeat$variant == key$variant,
  ]

  reference <- base[
    base$network == key$network &
    base$sample_size == key$sample_size &
    base$beta == key$beta,
  ]

  merged <- merge(
    treated[, c("repeat_index", "f1")],
    reference[, c("repeat_index", "f1")],
    by = "repeat_index", suffixes = c("_variant", "_baseline")
  )

  delta <- merged$f1_variant - merged$f1_baseline
  n <- length(delta)
  wins <- sum(delta > 0)

  p_value <- tryCatch(
    suppressWarnings(wilcox.test(delta, exact = TRUE)$p.value),
    error = function(e) NA
  )
  if (length(p_value) == 0 || is.na(p_value)) {
    p_value <- tryCatch(
      suppressWarnings(wilcox.test(delta)$p.value),
      error = function(e) NA
    )
  }

  results <- rbind(results, data.frame(
    network = key$network,
    sample_size = key$sample_size,
    beta = key$beta,
    variant = key$variant,
    repeats = n,
    baseline_f1 = round(mean(merged$f1_baseline), 4),
    variant_f1 = round(mean(merged$f1_variant), 4),
    mean_delta = round(mean(delta), 4),
    sd_delta = round(sd(delta), 4),
    wins = sprintf("%d/%d", wins, n),
    wilcox_p = round(p_value, 4)
  ))
}

results <- results[order(
  results$network, results$sample_size, results$beta, results$variant
), ]

cat(sprintf("paired comparison, %d repeats, tag=%s\n\n", max(per_repeat$repeat_index), tag))
print(results, row.names = FALSE)

write.csv(
  results,
  sprintf("%s/results/bnlearn_fusion_paired_%s.csv", project_root, tag),
  row.names = FALSE
)
cat(sprintf("\nwritten: results/bnlearn_fusion_paired_%s.csv\n", tag))
