# Prior-guided Markov blanket discovery, implemented on top of bnlearn.
#
# The plan asks for the language-model prior to be injected as an adjustment of
# the conditional independence test threshold. bnlearn 5.2.1 no longer accepts a
# bare function for `test`, but it does expose "custom-test": a user function
# with signature function(x, y, z, data, args) returning c(statistic, p.value).
# Wrapping the built-in mutual information test lets us scale the p-value:
#
#     p_effective = p * exp(-beta * prior_score)
#
# so a pair the model believes is directly connected becomes easier to accept
# (smaller p), and a pair it believes has no edge becomes harder. beta = 0
# reproduces the plain data-driven baseline exactly.
#
# Variants:
#   baseline        bnlearn's own test, no prior
#   prior-thresh    real prior (real variable names) or anonymous prior (mode)
#   random-thresh   control: the same scores, shuffled across pairs
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\run_fusion_bnlearn.R --mode=real
#
# ASCII-only source on purpose (R on a Chinese Windows locale).

suppressPackageStartupMessages(library(bnlearn))

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

# 仓库根目录：本机 Windows 与 Colab/Linux 都能用。
# 优先环境变量 CAUSAL_LLM_ROOT；否则本机的 E:/Causal-LLM；再否则用当前工作目录。
project_root <- Sys.getenv(
  "CAUSAL_LLM_ROOT",
  if (dir.exists("E:/Causal-LLM")) "E:/Causal-LLM" else normalizePath(".")
)

networks <- strsplit(get_arg("networks", "ASIA,CHILD,INSURANCE,ALARM"), ",")[[1]]
sizes <- as.integer(strsplit(get_arg("sizes", "100,500"), ",")[[1]])
repeats <- as.integer(get_arg("repeats", "5"))
mode <- get_arg("mode", "real")
# Either a single --beta or a sweep via --betas=0,0.25,0.5,1,2.
# beta = 0 keeps the p-value untouched, so prior-thresh must then reproduce the
# baseline exactly - a useful self-check of the custom-test plumbing.
betas <- as.numeric(strsplit(
  get_arg("betas", get_arg("beta", "1.0")), ","
)[[1]])
alpha <- as.numeric(get_arg("alpha", "0.05"))
base_test <- get_arg("test", "mi")
seed <- as.integer(get_arg("seed", "20261012"))
tag <- get_arg("tag", mode)

# ---------------------------------------------------------------------------
# Data: subsample from the largest file so every repeat is an independent draw
# ---------------------------------------------------------------------------

build_levels <- function(network) {
  path <- sprintf("%s/data/%s/%s_n5000.txt", project_root, network, network)
  frame <- read.table(
    path, header = TRUE, sep = "\t",
    colClasses = "character", check.names = FALSE
  )
  lapply(frame, function(column) sort(unique(as.character(column))))
}

load_full <- function(network, levels_map) {
  path <- sprintf("%s/data/%s/%s_n5000.txt", project_root, network, network)
  frame <- read.table(
    path, header = TRUE, sep = "\t",
    colClasses = "character", check.names = FALSE
  )
  for (name in names(frame)) {
    frame[[name]] <- factor(frame[[name]], levels = levels_map[[name]])
  }
  frame
}

# ---------------------------------------------------------------------------
# Prior scores
# ---------------------------------------------------------------------------

load_prior <- function(network, mode) {
  path <- sprintf(
    "%s/data/%s/%s_LLM_prior_%s_summary.csv",
    project_root, network, network, mode
  )
  if (!file.exists(path)) return(NULL)

  table <- read.csv(path, stringsAsFactors = FALSE)
  scores <- list()

  for (i in seq_len(nrow(table))) {
    a <- table$variable_a[i]
    b <- table$variable_b[i]
    confidence <- suppressWarnings(as.numeric(table$final_confidence[i]))
    if (is.na(confidence)) confidence <- 0
    confidence <- min(max(confidence, 0), 1)

    relation <- as.character(table$final_relation[i])
    direct <- tolower(as.character(table$direct_edge[i])) == "true"

    score <- if (direct || relation %in% c("A_to_B", "B_to_A")) {
      confidence
    } else if (relation == "no_direct_edge") {
      -confidence
    } else {
      0
    }

    scores[[paste(sort(c(a, b)), collapse = "|")]] <- score
  }

  scores
}

lookup <- function(scores, x, y) {
  if (is.null(scores)) return(0)
  key <- paste(sort(c(as.character(x), as.character(y))), collapse = "|")
  value <- scores[[key]]
  if (is.null(value)) 0 else value
}

# ---------------------------------------------------------------------------
# Custom conditional independence tests
# ---------------------------------------------------------------------------

make_test <- function(scale_for) {
  function(x, y, z, data, args) {
    result <- if (length(z) > 0) {
      ci.test(x, y, z, data = data, test = base_test)
    } else {
      ci.test(x, y, data = data, test = base_test)
    }
    p <- as.numeric(result$p.value)
    statistic <- as.numeric(result$statistic)
    # On sparse tables ci.test can come back with a non-finite p-value.
    # Treat that as "no evidence of dependence" rather than letting NA leak
    # into the search, which is what produced the warnings.
    if (!is.finite(p)) p <- 1
    if (!is.finite(statistic)) statistic <- 0
    c(statistic, min(1, p * scale_for(x, y)))
  }
}

shuffle_scores <- function(scores, seed) {
  # Watch out: reordering a named list with scores[perm] keeps each value
  # attached to its own name, so the mapping does not change at all.
  # The values themselves have to be permuted while the keys stay put.
  if (is.null(scores) || length(scores) == 0) return(scores)
  values <- unlist(scores, use.names = FALSE)
  set.seed(seed)
  values <- sample(values, length(values), replace = FALSE)
  shuffled <- as.list(values)
  names(shuffled) <- names(scores)
  shuffled
}

# Plain prior adjustment multiplies the p-value by exp(-beta * score). Because
# the language model answers "direct edge" far more often than "no edge", the
# average score is positive, so the adjustment systematically shrinks p-values
# and the Markov blanket simply grows. That conflates "accept more variables"
# with "accept the right variables".
#
# Centring removes exactly that bias: subtract the mean score over all pairs, so
# the average multiplier becomes 1. Pairs the model favours still get loosened,
# but now they are paid for by pairs the model rejects being tightened.
center_scores <- function(scores, verbose = FALSE) {
  if (is.null(scores) || length(scores) == 0) return(scores)
  values <- unlist(scores, use.names = FALSE)
  shift <- mean(values, na.rm = TRUE)
  if (verbose) {
    cat(sprintf("  prior mean score = %+.3f (centred around it)\n", shift))
  }
  lapply(scores, function(value) value - shift)
}

# We deliberately keep the full level set of every variable across sample sizes,
# so bnlearn warns "variable X has levels that are not observed in the data"
# whenever a small sample happens to miss a rare level (for example "tub" in
# ASIA n=100 is always "no"). The warning is expected and the run is still
# correct, so it is muffled - but only this one message, everything else still
# surfaces.
quiet_expected_warnings <- function(expression) {
  withCallingHandlers(
    expression,
    warning = function(condition) {
      if (grepl("levels that are not observed", conditionMessage(condition))) {
        invokeRestart("muffleWarning")
      }
    }
  )
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

score <- function(predicted, truth) {
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

  truth_path <- sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )
  if (!file.exists(truth_path)) {
    cat(sprintf("%s: no truth matrix, skipped\n", network))
    next
  }

  truth_matrix <- as.matrix(read.table(truth_path))
  levels_map <- build_levels(network)
  full <- load_full(network, levels_map)
  variables <- names(full)
  prior <- load_prior(network, mode)
  prior_centered <- center_scores(prior, verbose = TRUE)

  cat(sprintf(
    "%s: %d variables, prior covers %d pairs\n",
    network, length(variables), if (is.null(prior)) 0 else length(prior)
  ))

  for (size in sizes) {
    for (repeat_index in seq_len(repeats)) {

      set.seed(seed + repeat_index)
      rows <- if (size >= nrow(full)) {
        seq_len(nrow(full))
      } else {
        sample(nrow(full), size, replace = FALSE)
      }
      sample_frame <- full[rows, , drop = FALSE]

      # Prior scores shuffled across pairs: the control condition.
      shuffled <- shuffle_scores(prior, seed + 1000 + repeat_index)

      for (beta in betas) {

        shuffled_centered <- center_scores(shuffled)

        variants <- list(
          baseline = NULL,
          "prior-thresh" = if (is.null(prior)) NULL else make_test(
            function(x, y) exp(-beta * lookup(prior, x, y))
          ),
          "random-thresh" = if (is.null(shuffled)) NULL else make_test(
            function(x, y) exp(-beta * lookup(shuffled, x, y))
          ),
          "prior-centered" = if (is.null(prior_centered)) NULL else make_test(
            function(x, y) exp(-beta * lookup(prior_centered, x, y))
          ),
          "random-centered" = if (is.null(shuffled_centered)) NULL else make_test(
            function(x, y) exp(-beta * lookup(shuffled_centered, x, y))
          )
        )

        for (variant in names(variants)) {

          test_definition <- variants[[variant]]

          started <- Sys.time()

          for (index in seq_along(variables)) {

            target <- variables[index]
            true_set <- variables[truth_matrix[index, ] == 1]

            mb <- tryCatch(
              quiet_expected_warnings(
                if (is.null(test_definition)) {
                  learn.mb(sample_frame, node = target, method = "iamb", test = base_test)
                } else {
                  learn.mb(sample_frame, node = target, method = "iamb",
                           test = "custom-test", fun = test_definition)
                }
              ),
              error = function(error) character(0)
            )

            if (is.null(mb)) mb <- character(0)

            metrics <- score(mb, true_set)

            detail <- rbind(detail, data.frame(
              network = network,
              sample_size = size,
              repeat_index = repeat_index,
              beta = beta,
              variant = variant,
              target = target,
              precision = metrics[["precision"]],
              recall = metrics[["recall"]],
              f1 = metrics[["f1"]],
              mb_size = length(mb)
            ))
          }

          elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
          cat(sprintf("  n=%-5d beta=%-5.2f %-13s done in %.1fs\n",
                      size, beta, variant, elapsed))
        }
      }
    }
  }
}

if (nrow(detail) == 0) {
  cat("no results\n")
  quit(status = 1)
}

dir.create(sprintf("%s/results", project_root), showWarnings = FALSE)

write.csv(
  detail,
  sprintf("%s/results/bnlearn_fusion_detail_%s.csv", project_root, tag),
  row.names = FALSE
)

keys <- unique(detail[, c("network", "sample_size", "beta", "variant")])
summary_table <- data.frame()

for (i in seq_len(nrow(keys))) {
  key <- keys[i, ]
  rows <- detail[
    detail$network == key$network &
    detail$sample_size == key$sample_size &
    detail$beta == key$beta &
    detail$variant == key$variant,
  ]
  # Every variant is run with both MB learners here (iamb only), so the summary
  # carries a method column to keep the same shape as the other result tables.
  summary_table <- rbind(summary_table, data.frame(
    network = key$network,
    sample_size = key$sample_size,
    beta = key$beta,
    method = "IAMB",
    variant = key$variant,
    precision = round(mean(rows$precision), 4),
    recall = round(mean(rows$recall), 4),
    f1 = round(mean(rows$f1), 4),
    f1_std = round(sd(aggregate(f1 ~ repeat_index, rows, mean)$f1), 4),
    mean_mb_size = round(mean(rows$mb_size), 2)
  ))
}

summary_table <- summary_table[order(
  summary_table$network, summary_table$sample_size,
  summary_table$beta, summary_table$variant
), ]

write.csv(
  summary_table,
  sprintf("%s/results/bnlearn_fusion_summary_%s.csv", project_root, tag),
  row.names = FALSE
)

cat("\n")
print(summary_table, row.names = FALSE)
cat(sprintf(
  "\nwritten: results/bnlearn_fusion_detail_%s.csv, results/bnlearn_fusion_summary_%s.csv\n",
  tag, tag
))
