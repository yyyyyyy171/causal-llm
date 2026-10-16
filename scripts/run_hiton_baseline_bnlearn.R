# HITON-MB baseline, reproduced in R and checked against the MATLAB numbers that
# are already in the repository (results/HITONMB_4networks_summary.csv).
#
# This has to agree with MATLAB before the prior is injected, otherwise any
# comparison would be meaningless. The preprocessing mirrors the MATLAB driver
# (run_HITONMB_4networks.m):
#
#   * read the numeric-coded data file <NET>_n<N>_num.txt;
#   * ns = number of states actually observed in this sample;
#   * drop variables with fewer than two observed states;
#   * run HITON-MB on the surviving variables and map the result back to the
#     original variable indices before scoring.
#
# That last point matters: in ASIA n=100 the variable "tub" has a single
# observed state, so MATLAB simply removes it from the search.
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\run_hiton_baseline_bnlearn.R --networks=ASIA

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
source(sprintf("%s/R/hiton_mb.R", project_root))

networks <- strsplit(get_arg("networks", "ASIA,CHILD,INSURANCE,HAILFINDER"), ",")[[1]]
sizes <- as.integer(strsplit(get_arg("sizes", "100,500,1000,5000"), ",")[[1]])
alpha <- as.numeric(get_arg("alpha", "0.05"))
max_k <- as.integer(get_arg("max-k", "3"))
tag <- get_arg("tag", "hiton")

# ---------------------------------------------------------------------------
# Load and prepare exactly like the MATLAB driver
# ---------------------------------------------------------------------------

prepare <- function(network, n) {

  path <- sprintf("%s/data/%s/%s_n%d_num.txt", project_root, network, network, n)
  if (!file.exists(path)) return(NULL)

  raw <- as.matrix(read.table(path, header = FALSE))

  codes <- matrix(0L, nrow = nrow(raw), ncol = ncol(raw))
  n_states <- integer(ncol(raw))

  for (column in seq_len(ncol(raw))) {
    values <- sort(unique(raw[, column]))
    n_states[column] <- length(values)
    codes[, column] <- match(raw[, column], values)
  }

  valid <- n_states >= 2
  list(
    data = codes[, valid, drop = FALSE],
    ns = n_states[valid],
    original_index = which(valid),
    n_states_all = n_states
  )
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

rows <- data.frame()

for (network in networks) {

  truth <- as.matrix(read.table(sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )))

  for (n in sizes) {

    prepared <- prepare(network, n)
    if (is.null(prepared)) next

    data <- prepared$data
    ns <- prepared$ns
    original_index <- prepared$original_index
    variables <- seq_len(ncol(data))

    test <- make_g2(data, ns, alpha = alpha, scale = NULL)

    started <- Sys.time()
    precision <- c(); recall <- c(); f1 <- c(); sizes_found <- c()

    for (target in variables) {

      predicted <- tryCatch(
        hiton_mb(test, target, variables, max_k, alpha),
        error = function(error) integer(0)
      )

      predicted_original <- original_index[predicted]
      target_original <- original_index[target]
      true_set <- which(truth[target_original, ] > 0)

      tp <- length(intersect(predicted_original, true_set))
      p <- if (length(predicted_original)) tp / length(predicted_original) else 0
      r <- if (length(true_set)) tp / length(true_set) else 0
      precision <- c(precision, p)
      recall <- c(recall, r)
      f1 <- c(f1, if (p + r > 0) 2 * p * r / (p + r) else 0)
      sizes_found <- c(sizes_found, length(predicted_original))
    }

    elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))

    rows <- rbind(rows, data.frame(
      network = network,
      sample_size = n,
      method = "HITON-MB (R port)",
      precision = round(mean(precision), 4),
      recall = round(mean(recall), 4),
      f1 = round(mean(f1), 4),
      mean_mb_size = round(mean(sizes_found), 2),
      targets = length(variables),
      dropped_variables = ncol(truth) - length(variables),
      seconds = round(elapsed, 1)
    ))

    cat(sprintf(
      "%-11s n=%-5d HITON-MB R  F1=%.4f  (%.1fs, %d targets, %d dropped)\n",
      network, n, mean(f1), elapsed, length(variables),
      ncol(truth) - length(variables)
    ))
  }
}

write.csv(
  rows,
  sprintf("%s/results/hiton_baseline_%s.csv", project_root, tag),
  row.names = FALSE
)

# ---------------------------------------------------------------------------
# Compare with the MATLAB baseline already in the repository
# ---------------------------------------------------------------------------

# MATLAB 参考基线：先用汇总文件，缺哪个网络就去找该网络自己的文件。
#
# ALARM 一直没进对照表，就是因为 results/HITONMB_4networks_summary.csv 里
# 根本没有它（那份文件只有 ASIA / CHILD / INSURANCE / HAILFINDER），
# 而 ALARM 的结果在 data/ALARM/ALARM_HITONMB_summary.csv。
# 少了这一步，跨网络交叉验证最多只能覆盖 4 个网络。
load_matlab_reference <- function(networks) {

  pieces <- list()

  summary_path <- sprintf("%s/results/HITONMB_4networks_summary.csv", project_root)
  if (file.exists(summary_path)) {
    pieces[[length(pieces) + 1]] <- read.csv(summary_path, stringsAsFactors = FALSE)
  }

  for (network in networks) {
    per_network <- sprintf(
      "%s/data/%s/%s_HITONMB_summary.csv", project_root, network, network
    )
    if (file.exists(per_network)) {
      block <- read.csv(per_network, stringsAsFactors = FALSE)
      block$network <- network
      pieces[[length(pieces) + 1]] <- block
    }
  }

  if (length(pieces) == 0) return(NULL)

  columns <- c("network", "sample_size", "Precision", "Recall", "F1")
  reference <- do.call(rbind, lapply(pieces, function(piece) piece[, columns]))

  # 同一网络同时出现在两处时保留先出现的（汇总文件优先）
  reference[!duplicated(reference[, c("network", "sample_size")]), ]
}

matlab <- load_matlab_reference(networks)

if (!is.null(matlab)) {
  comparison <- merge(
    rows[, c("network", "sample_size", "f1", "precision", "recall")],
    matlab[, c("network", "sample_size", "F1", "Precision", "Recall")],
    by = c("network", "sample_size"), suffixes = c("_r", "_matlab")
  )
  comparison$delta_f1 <- round(comparison$f1 - comparison$F1, 4)
  comparison <- comparison[order(comparison$network, comparison$sample_size), ]
  print(comparison, row.names = FALSE)
  write.csv(
    comparison,
    sprintf("%s/results/hiton_baseline_vs_matlab_%s.csv", project_root, tag),
    row.names = FALSE
  )
  cat(sprintf(
    "\nmean |delta F1| = %.4f\n", mean(abs(comparison$delta_f1))
  ))
} else {
  cat("(no MATLAB baseline files found)\n")
}

cat(sprintf("\nwritten: results/hiton_baseline_%s.csv\n", tag))
