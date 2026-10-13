# The validation experiment the plan actually asks for.
#
# The plan requires IAMB *and* HITON-MB, the prior injected into the
# conditional independence test, evaluated at sample sizes up to 500 where the
# prior is supposed to matter. Earlier rounds used only IAMB because bnlearn
# dropped HITON-MB; the port in R/hiton_mb.R now reproduces the MATLAB baseline
# (mean |delta F1| = 0.0037), so both algorithms can be run.
#
# Two metrics are reported for every cell:
#
#   raw       F1 at alpha = 0.05, i.e. what the algorithm naturally returns;
#   matched   F1 after ranking candidates along an alpha path and keeping the
#             top-k, with k = the size of the true Markov blanket. This removes
#             the "just select more variables" confound, which earlier rounds
#             showed can flip the sign of the conclusion.
#
# Variants: baseline, prior-thresh (p scaled by the prior), random-thresh
# (the same scores shuffled across pairs - the control that decides whether any
# improvement is semantic or just a perturbation).
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\run_fusion_both.R --repeats=30

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

project_root <- "E:/Causal-LLM"
source(sprintf("%s/R/hiton_mb.R", project_root))
suppressPackageStartupMessages(library(bnlearn))

networks <- strsplit(get_arg("networks", "ASIA,CHILD"), ",")[[1]]
sizes <- as.integer(strsplit(get_arg("sizes", "100,500"), ",")[[1]])
repeats <- as.integer(get_arg("repeats", "30"))
beta <- as.numeric(get_arg("beta", "0.5"))
alpha <- as.numeric(get_arg("alpha", "0.05"))
max_k <- as.integer(get_arg("max-k", "3"))
seed <- as.integer(get_arg("seed", "20261012"))
mode <- get_arg("mode", "real")
tag <- get_arg("tag", sprintf("both_n%s", mode))

alphas <- as.numeric(strsplit(
  get_arg("alphas", "1e-06,1e-04,0.001,0.01,0.05,0.2"), ","
)[[1]])

# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

load_prior <- function(network) {
  path <- sprintf(
    "%s/data/%s/%s_LLM_prior_%s_summary.csv", project_root, network, network, mode
  )
  if (!file.exists(path)) return(NULL)
  table <- read.csv(path, stringsAsFactors = FALSE)
  scores <- list()
  for (i in seq_len(nrow(table))) {
    a <- table$variable_a[i]; b <- table$variable_b[i]
    confidence <- suppressWarnings(as.numeric(table$final_confidence[i]))
    if (is.na(confidence)) confidence <- 0
    confidence <- min(max(confidence, 0), 1)
    relation <- as.character(table$final_relation[i])
    direct <- tolower(as.character(table$direct_edge[i])) == "true"
    score <- if (direct || relation %in% c("A_to_B", "B_to_A")) confidence
             else if (relation == "no_direct_edge") -confidence
             else 0
    scores[[paste(sort(c(a, b)), collapse = "|")]] <- score
  }
  scores
}

score_of <- function(scores, a, b) {
  if (is.null(scores)) return(0)
  key <- paste(sort(c(as.character(a), as.character(b))), collapse = "|")
  value <- scores[[key]]
  if (is.null(value)) 0 else value
}

shuffle_scores <- function(scores, seed) {
  if (is.null(scores) || length(scores) == 0) return(scores)
  values <- unlist(scores, use.names = FALSE)
  set.seed(seed)
  values <- sample(values, length(values), replace = FALSE)
  out <- as.list(values)
  names(out) <- names(scores)
  out
}

center_scores <- function(scores) {
  if (is.null(scores) || length(scores) == 0) return(scores)
  shift <- mean(unlist(scores, use.names = FALSE), na.rm = TRUE)
  lapply(scores, function(value) value - shift)
}

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

score_set <- function(predicted, truth) {
  predicted <- unique(predicted)
  tp <- length(intersect(predicted, truth))
  precision <- if (length(predicted) > 0) tp / length(predicted) else 0
  recall <- if (length(truth) > 0) tp / length(truth) else 0
  f1 <- if (precision + recall > 0) {
    2 * precision * recall / (precision + recall)
  } else 0
  c(precision = precision, recall = recall, f1 = f1)
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

detail <- data.frame()

for (network in networks) {

  truth <- as.matrix(read.table(sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )))

  # numeric-coded data, as the MATLAB driver uses
  full_raw <- as.matrix(read.table(
    sprintf("%s/data/%s/%s_n5000_num.txt", project_root, network, network),
    header = FALSE
  ))
  labels <- colnames(read.table(
    sprintf("%s/data/%s/%s_n100.txt", project_root, network, network),
    header = TRUE, sep = "\t", nrows = 1, check.names = FALSE
  ))
  colnames(full_raw) <- labels

  prior_real <- load_prior(network)
  cat(sprintf(
    "%s: %d variables, prior pairs = %d\n", network, ncol(full_raw),
    if (is.null(prior_real)) 0 else length(prior_real)
  ))

  for (size in sizes) {
    for (repeat_index in seq_len(repeats)) {

      set.seed(seed + repeat_index)
      rows <- sample(nrow(full_raw), size, replace = FALSE)
      sample_raw <- full_raw[rows, , drop = FALSE]

      # --- coding for the HITON-MB port (MATLAB convention) ---
      codes <- matrix(0L, nrow = nrow(sample_raw), ncol = ncol(sample_raw))
      ns <- integer(ncol(sample_raw))
      for (column in seq_len(ncol(sample_raw))) {
        values <- sort(unique(sample_raw[, column]))
        ns[column] <- length(values)
        codes[, column] <- match(sample_raw[, column], values)
      }
      valid <- ns >= 2
      codes_valid <- codes[, valid, drop = FALSE]
      ns_valid <- ns[valid]
      names_valid <- labels[valid]
      valid_index <- which(valid)

      # --- frame for bnlearn's IAMB (full level set kept) ---
      frame <- as.data.frame(sample_raw, stringsAsFactors = FALSE)
      for (name in labels) {
        frame[[name]] <- factor(
          frame[[name]], levels = sort(unique(full_raw[, name]))
        )
      }

      prior_centered <- center_scores(prior_real)
      shuffled <- shuffle_scores(prior_real, seed + 1000 + repeat_index)
      shuffled_centered <- center_scores(shuffled)

      variants <- list(
        baseline = list(scores = NULL, use = FALSE),
        "prior-thresh" = list(scores = prior_centered, use = TRUE),
        "random-thresh" = list(scores = shuffled_centered, use = TRUE)
      )

      for (variant in names(variants)) {

        scores <- variants[[variant]]$scores
        use_prior <- variants[[variant]]$use

        for (target_index in seq_along(labels)) {

          target_name <- labels[target_index]
          true_set <- labels[which(truth[target_index, ] > 0)]
          k <- max(1, length(true_set))

          # ============================================================
          # IAMB through bnlearn
          # ============================================================
          scale_bnlearn <- if (use_prior) {
            function(x, y) exp(-beta * score_of(scores, x, y))
          } else NULL

          run_iamb <- function(alpha_value) {
            if (is.null(scale_bnlearn)) {
              learn.mb(frame, node = target_name, method = "iamb",
                       test = "mi", alpha = alpha_value)
            } else {
              fun <- function(x, y, z, data, args) {
                r <- if (length(z) > 0) {
                  ci.test(x, y, z, data = data, test = "mi")
                } else {
                  ci.test(x, y, data = data, test = "mi")
                }
                p <- as.numeric(r$p.value); s <- as.numeric(r$statistic)
                if (!is.finite(p)) p <- 1
                if (!is.finite(s)) s <- 0
                c(s, min(1, p * scale_bnlearn(x, y)))
              }
              learn.mb(frame, node = target_name, method = "iamb",
                       test = "custom-test", fun = fun, alpha = alpha_value)
            }
          }

          # ============================================================
          # HITON-MB through the R port
          # ============================================================
          scale_hiton <- if (use_prior) {
            function(x, y) {
              exp(-beta * score_of(scores, names_valid[x], names_valid[y]))
            }
          } else NULL

          # hiton_mb works on the surviving variables only, so the target index
          # has to be mapped into that working set and the result mapped back to
          # variable names. Forgetting either step silently empties every result:
          # the raw set came back as integer working indices and was then
          # filtered by variable names, which removes everything.
          target_local <- match(target_name, names_valid)

          run_hiton <- function(alpha_value) {
            if (is.na(target_local)) return(character(0))
            test <- make_g2(codes_valid, ns_valid, alpha = alpha_value,
                            scale = scale_hiton)
            names_valid[hiton_mb(test, target_local, seq_along(names_valid),
                                 max_k, alpha_value)]
          }

          for (method in c("IAMB", "HITON-MB")) {

            runner <- if (method == "IAMB") run_iamb else run_hiton

            # alpha path -> ranking for the size-matched metric
            entry <- rep(Inf, length(labels))
            names(entry) <- labels
            raw_set <- character(0)

            for (alpha_value in alphas) {
              found <- tryCatch(
                withCallingHandlers(
                  runner(alpha_value),
                  warning = function(condition) {
                    if (grepl("levels that are not observed",
                              conditionMessage(condition))) {
                      invokeRestart("muffleWarning")
                    }
                  }
                ),
                error = function(error) character(0)
              )
              if (is.null(found)) found <- character(0)
              found <- found[found %in% labels]
              for (name in found) {
                if (is.infinite(entry[[name]])) entry[[name]] <- alpha_value
              }
              if (abs(alpha_value - alpha) < 1e-12) raw_set <- found
            }

            ranking <- labels[order(entry[labels], labels)]
            top_k <- ranking[seq_len(min(k, length(ranking)))]

            raw_metrics <- score_set(raw_set, true_set)
            matched_metrics <- score_set(top_k, true_set)

            detail <- rbind(detail,
              data.frame(
                network = network, sample_size = size,
                repeat_index = repeat_index, method = method,
                variant = variant, budget = "raw",
                precision = raw_metrics[["precision"]],
                recall = raw_metrics[["recall"]],
                f1 = raw_metrics[["f1"]],
                n_selected = length(raw_set)
              ),
              data.frame(
                network = network, sample_size = size,
                repeat_index = repeat_index, method = method,
                variant = variant, budget = "matched",
                precision = matched_metrics[["precision"]],
                recall = matched_metrics[["recall"]],
                f1 = matched_metrics[["f1"]],
                n_selected = length(top_k)
              )
            )
          }
        }
      }

      if (repeat_index %% 5 == 0) {
        cat(sprintf("  %s n=%d repeat %d/%d done\n",
                    network, size, repeat_index, repeats))
      }
    }
  }
}

dir.create(sprintf("%s/results", project_root), showWarnings = FALSE)
write.csv(
  detail,
  sprintf("%s/results/fusion_both_detail_%s.csv", project_root, tag),
  row.names = FALSE
)

# ---------------------------------------------------------------------------
# Summary: mean over targets and repeats, plus a paired test against baseline
# ---------------------------------------------------------------------------

keys <- unique(detail[, c("network", "sample_size", "method", "variant", "budget")])
rows <- data.frame()
paired <- data.frame()

for (i in seq_len(nrow(keys))) {
  key <- keys[i, ]
  block <- detail[
    detail$network == key$network & detail$sample_size == key$sample_size &
    detail$method == key$method & detail$variant == key$variant &
    detail$budget == key$budget, ]

  per_repeat <- aggregate(f1 ~ repeat_index, block, mean)

  rows <- rbind(rows, data.frame(
    network = key$network, sample_size = key$sample_size,
    method = key$method, variant = key$variant, budget = key$budget,
    precision = round(mean(block$precision), 4),
    recall = round(mean(block$recall), 4),
    f1 = round(mean(block$f1), 4),
    f1_std = round(sd(per_repeat$f1), 4),
    mean_selected = round(mean(block$n_selected), 2)
  ))

  if (key$variant != "baseline") {
    base <- aggregate(
      f1 ~ repeat_index,
      detail[detail$network == key$network &
             detail$sample_size == key$sample_size &
             detail$method == key$method &
             detail$variant == "baseline" &
             detail$budget == key$budget, ],
      mean
    )
    merged <- merge(
      per_repeat, base, by = "repeat_index", suffixes = c("_v", "_b")
    )
    delta <- merged$f1_v - merged$f1_b
    p <- tryCatch(suppressWarnings(wilcox.test(delta)$p.value),
                  error = function(e) NA)
    paired <- rbind(paired, data.frame(
      network = key$network, sample_size = key$sample_size,
      method = key$method, variant = key$variant, budget = key$budget,
      baseline_f1 = round(mean(merged$f1_b), 4),
      variant_f1 = round(mean(merged$f1_v), 4),
      mean_delta = round(mean(delta), 4),
      wins = sprintf("%d/%d", sum(delta > 0), length(delta)),
      wilcox_p = round(p, 4)
    ))
  }
}

rows <- rows[order(rows$network, rows$sample_size, rows$method, rows$budget,
                   rows$variant), ]
write.csv(rows, sprintf("%s/results/fusion_both_summary_%s.csv", project_root, tag),
          row.names = FALSE)
write.csv(paired,
          sprintf("%s/results/fusion_both_paired_%s.csv", project_root, tag),
          row.names = FALSE)

cat("\n")
print(rows, row.names = FALSE)
cat("\npaired vs baseline:\n")
print(paired, row.names = FALSE)
