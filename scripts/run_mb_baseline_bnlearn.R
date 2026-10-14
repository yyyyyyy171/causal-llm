# Markov blanket recovery baselines, using the bnlearn package itself.
#
# Week 4 of the plan asks for IAMB and HITON-MB on the five bnlearn networks,
# at sample sizes 100 / 500 / 1000 / 5000, scored against the true Markov
# blanket of every target variable.
#
# IMPORTANT - availability of the algorithms in bnlearn 5.2.1:
#
#   learn.mb() supports: gs, iamb, fast.iamb, inter.iamb, iamb.fdr
#   HITON-MB is NOT available any more: the old hiton.mb() was removed and
#   si.hiton.pc() is a parents-and-children learner behind the new cluster
#   API, not a Markov blanket learner. There is no Rtools on this machine,
#   so an older bnlearn (which still had hiton.mb) cannot be built from source.
#
# Therefore: IAMB comes from bnlearn (canonical), while HITON-MB keeps using
# the existing MATLAB / Causal-Learner implementation that produced
# results/BN_baseline_summary.csv.
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\run_mb_baseline_bnlearn.R
#   ... --vanilla scripts\run_mb_baseline_bnlearn.R --networks=ASIA --sizes=100,500
#
# The file is ASCII-only on purpose: R on a Chinese Windows locale can fail to
# parse non-ASCII source files.

suppressPackageStartupMessages(library(bnlearn))

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

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

networks <- strsplit(
  get_arg("networks", "ALARM,ASIA,CHILD,INSURANCE,HAILFINDER"), ","
)[[1]]
sizes <- as.integer(strsplit(get_arg("sizes", "100,500,1000,5000"), ",")[[1]])
methods <- strsplit(get_arg("methods", "iamb,hiton.mb"), ",")[[1]]
tests <- strsplit(get_arg("tests", "mi"), ",")[[1]]
alpha <- as.numeric(get_arg("alpha", "0.05"))

# ---------------------------------------------------------------------------
# Helpers
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

# The level sets of the variables, taken from the largest sample so that every
# sample size uses the same factor levels. This matters: in ASIA n=100 the
# column "tub" happens to contain a single value ("no"), and bnlearn refuses a
# variable with fewer than two levels ("variable tub in the data must have at
# least two levels"). Declaring the levels explicitly keeps all sample sizes
# comparable and is also the statistically honest thing to do.
build_levels <- function(network) {
  path <- sprintf("%s/data/%s/%s_n5000.txt", project_root, network, network)
  if (!file.exists(path)) {
    path <- sprintf("%s/data/%s/%s_n1000.txt", project_root, network, network)
  }
  if (!file.exists(path)) return(NULL)
  frame <- read.table(
    path, header = TRUE, sep = "\t",
    stringsAsFactors = FALSE, check.names = FALSE
  )
  lapply(frame, function(column) sort(unique(as.character(column))))
}

load_dataset <- function(network, n, levels_map) {
  path <- sprintf("%s/data/%s/%s_n%d.txt", project_root, network, network, n)
  if (!file.exists(path)) return(NULL)
  frame <- read.table(
    path,
    header = TRUE,
    sep = "\t",
    colClasses = "character",
    check.names = FALSE
  )

  # Two traps solved here:
  #   * a column holding only TRUE/FALSE is read as logical and bnlearn rejects
  #     logical variables ("variable X is not supported") - that made every
  #     ALARM target fail on the first attempt;
  #   * a column that happens to be constant within one sample would collapse to
  #     a single level.
  # Reading as character and applying the shared level set fixes both.
  for (name in names(frame)) {
    levels_for_column <- if (!is.null(levels_map) && name %in% names(levels_map)) {
      levels_map[[name]]
    } else {
      sort(unique(frame[[name]]))
    }
    frame[[name]] <- factor(frame[[name]], levels = levels_for_column)
  }

  frame
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

detail <- data.frame()

for (network in networks) {

  truth_path <- sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )

  if (!file.exists(truth_path)) {
    cat(sprintf("%s: missing truth matrix, skipped\n", network))
    next
  }

  truth_matrix <- as.matrix(read.table(truth_path))
  levels_map <- build_levels(network)

  for (n in sizes) {

    frame <- load_dataset(network, n, levels_map)

    if (is.null(frame)) {
      cat(sprintf("%s n=%d: missing data, skipped\n", network, n))
      next
    }

    variables <- names(frame)

    if (nrow(truth_matrix) != length(variables)) {
      cat(sprintf(
        "%s n=%d: truth matrix is %dx%d but data has %d columns, skipped\n",
        network, n, nrow(truth_matrix), ncol(truth_matrix), length(variables)
      ))
      next
    }

    for (method in methods) {

      for (test in tests) {

        started <- Sys.time()
        failures <- character(0)

        for (index in seq_along(variables)) {

          target <- variables[index]
          true_set <- variables[truth_matrix[index, ] == 1]

          # learn.mb(data, node, method, ...) is the 5.x entry point;
          # the raw iamb(data, cluster, ...) form needs an internal cluster object.
          mb <- tryCatch(
            learn.mb(
              frame,
              node = target,
              method = method,
              test = test,
              alpha = alpha
            ),
            error = function(error) {
              failures <<- c(failures, target)
              character(0)
            }
          )

          if (is.null(mb)) mb <- character(0)

          metrics <- score(mb, true_set)

          detail <- rbind(detail, data.frame(
            network = network,
            sample_size = n,
            method = method,
            test = test,
            target = target,
            precision = metrics[["precision"]],
            recall = metrics[["recall"]],
            f1 = metrics[["f1"]],
            mb_size = length(mb),
            true_size = length(true_set)
          ))
        }

        elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))

        cat(sprintf(
          "%-11s n=%-5d %-9s test=%-3s done in %.1fs%s\n",
          network, n, method, test, elapsed,
          if (length(failures)) {
            sprintf("  (%d targets errored: %s)",
                    length(failures),
                    paste(utils::head(failures, 3), collapse = ","))
          } else ""
        ))
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

if (nrow(detail) == 0) {
  cat("no results\n")
  quit(status = 1)
}

dir.create(sprintf("%s/results", project_root), showWarnings = FALSE)

write.csv(
  detail,
  sprintf("%s/results/bnlearn_mb_detail.csv", project_root),
  row.names = FALSE
)

# Summary: average over targets (and over sizes is left to the reader)
keys <- unique(detail[, c("network", "sample_size", "method", "test")])

summary <- data.frame()

for (i in seq_len(nrow(keys))) {
  key <- keys[i, ]
  rows <- detail[
    detail$network == key$network &
    detail$sample_size == key$sample_size &
    detail$method == key$method &
    detail$test == key$test,
  ]
  summary <- rbind(summary, data.frame(
    network = key$network,
    sample_size = key$sample_size,
    method = key$method,
    test = key$test,
    precision = round(mean(rows$precision), 4),
    recall = round(mean(rows$recall), 4),
    f1 = round(mean(rows$f1), 4),
    mean_mb_size = round(mean(rows$mb_size), 2),
    targets = nrow(rows)
  ))
}

summary <- summary[order(
  summary$network, summary$sample_size, summary$method, summary$test
), ]

write.csv(
  summary,
  sprintf("%s/results/bnlearn_mb_summary.csv", project_root),
  row.names = FALSE
)

cat("\n")
print(summary, row.names = FALSE)
cat("\nwritten: results/bnlearn_mb_detail.csv, results/bnlearn_mb_summary.csv\n")
