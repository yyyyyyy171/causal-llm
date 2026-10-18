# Fixing the two problems found in the threshold-adjustment design.
#
# Problem 1 - the mechanism. Shifting the p-value for every pair lets the prior
# simply inflate the Markov blanket, and the random-prior control showed that on
# ASIA n=100 that alone explains the "gain". Here the prior instead acts as a
# SEARCH CONSTRAINT, which bnlearn supports through whitelist / blacklist
# (they take node names, not edges):
#
#   prior-blacklist   never include variables the model says have no direct edge
#                     to the target - this can only shrink the blanket
#   prior-whitelist   always include the variables the model endorses
#
# Problem 2 - the metric. A single F1 at alpha = 0.05 rewards whoever selects
# more variables. So every variant is turned into a RANKING by running the
# search along an alpha path (strict to loose) and recording the alpha at which
# each variable first enters the blanket; the variant is then scored on its
# top-k variables, with k = the size of the true Markov blanket. Every variant
# therefore selects exactly the same number of variables.
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\run_constraint_bnlearn.R

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

networks <- strsplit(get_arg("networks", "ASIA,CHILD"), ",")[[1]]
size <- as.integer(get_arg("size", "100"))
repeats <- as.integer(get_arg("repeats", "20"))
mode <- get_arg("mode", "real")
# prior-task: which prompt produced the scores.
#   direction -> data/<NET>/<NET>_LLM_prior_<mode>_summary.csv  (direct edge, +/-)
#   mb        -> data/<NET>/<NET>_LLM_mb_<mode>_summary.csv     (blanket member, +/-)
prior_task <- get_arg("prior-task", "direction")
# 先验来自哪个模型：8B 的结果文件名带 _qwen3-8b-awq 后缀，留空读无后缀的旧结果
model_tag <- get_arg("model-tag", "")
suffix <- if (nzchar(model_tag)) paste0("_", model_tag) else ""
beta <- as.numeric(get_arg("beta", "0.5"))
tau <- as.numeric(get_arg("tau", "0.6"))
seed <- as.integer(get_arg("seed", "20261012"))
tag <- get_arg("tag", sprintf("n%d_%s", size, mode))

# strict to loose; the entry alpha becomes the ranking score
alphas <- as.numeric(strsplit(
  get_arg("alphas", "1e-06,1e-04,0.001,0.01,0.05,0.2"), ","
)[[1]])

# ---------------------------------------------------------------------------
# Data and prior
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

load_prior <- function(network, mode) {
  prefix <- switch(prior_task, mb = "LLM_mb", role = "LLM_role", "LLM_prior")
  path <- sprintf(
    "%s/data/%s/%s_%s_%s%s_summary.csv",
    project_root, network, network, prefix, mode, suffix
  )
  if (!file.exists(path)) return(NULL)
  table <- read.csv(path, stringsAsFactors = FALSE)
  scores <- list()
  for (i in seq_len(nrow(table))) {
    a <- table$variable_a[i]; b <- table$variable_b[i]
    confidence <- suppressWarnings(as.numeric(table$final_confidence[i]))
    if (is.na(confidence)) confidence <- 0
    confidence <- min(max(confidence, 0), 1)

    if (prior_task == "role") {
      role <- tolower(as.character(table$final_role[i]))
      score <- if (role %in% c("parent", "child", "spouse")) confidence
               else if (role == "none") -confidence
               else 0
    } else if (prior_task == "mb") {
      positive <- tolower(as.character(table$in_markov_blanket[i])) == "true"
      score <- if (positive) confidence else -confidence
    } else {
      relation <- as.character(table$final_relation[i])
      direct <- tolower(as.character(table$direct_edge[i])) == "true"
      score <- if (direct || relation %in% c("A_to_B", "B_to_A")) confidence
               else if (relation == "no_direct_edge") -confidence
               else 0
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

center_scores <- function(scores) {
  if (is.null(scores) || length(scores) == 0) return(scores)
  shift <- mean(unlist(scores, use.names = FALSE), na.rm = TRUE)
  lapply(scores, function(value) value - shift)
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

quiet <- function(expression) {
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
# Variants
# ---------------------------------------------------------------------------

scaled_test <- function(scores, beta) {
  function(x, y, z, data, args) {
    result <- if (length(z) > 0) {
      ci.test(x, y, z, data = data, test = "mi")
    } else {
      ci.test(x, y, data = data, test = "mi")
    }
    p <- as.numeric(result$p.value)
    statistic <- as.numeric(result$statistic)
    if (!is.finite(p)) p <- 1
    if (!is.finite(statistic)) statistic <- 0
    c(statistic, min(1, p * exp(-beta * lookup(scores, x, y))))
  }
}

# whitelist / blacklist are per target: they depend on the score of each pair
# with that target.
constrained_variables <- function(scores, target, variables, tau, kind) {
  others <- setdiff(variables, target)
  values <- vapply(others, function(v) lookup(scores, target, v), numeric(1))
  if (kind == "blacklist") others[values < -tau] else others[values > tau]
}

# ---------------------------------------------------------------------------
# Scoring
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

# 统计被 tryCatch 吞掉的调用失败次数：大于 0 就说明结果不可信，必须报出来
failed_calls <- 0
detail <- data.frame()

for (network in networks) {

  truth_matrix <- as.matrix(read.table(sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )))
  levels_map <- build_levels(network)
  full <- load_full(network, levels_map)
  variables <- names(full)

  prior <- load_prior(network, mode)
  prior_centered <- center_scores(prior)

  cat(sprintf("%s: %d variables, prior pairs = %d\n",
              network, length(variables),
              if (is.null(prior)) 0 else length(prior)))

  for (repeat_index in seq_len(repeats)) {

    set.seed(seed + repeat_index)
    rows <- if (size >= nrow(full)) seq_len(nrow(full))
            else sample(nrow(full), size, replace = FALSE)
    sample_frame <- full[rows, , drop = FALSE]

    shuffled <- shuffle_scores(prior, seed + 1000 + repeat_index)
    shuffled_centered <- center_scores(shuffled)

    variants <- list(
      baseline = list(test = NULL, kind = "none", scores = NULL),
      "prior-thresh" = list(
        test = scaled_test(prior_centered, beta), kind = "none", scores = NULL
      ),
      "prior-blacklist" = list(
        test = NULL, kind = "blacklist", scores = prior
      ),
      "random-blacklist" = list(
        test = NULL, kind = "blacklist", scores = shuffled
      ),
      "prior-whitelist" = list(
        test = NULL, kind = "whitelist", scores = prior
      )
    )

    for (variant in names(variants)) {

      specification <- variants[[variant]]
      started <- Sys.time()

      for (index in seq_along(variables)) {

        target <- variables[index]
        true_set <- variables[truth_matrix[index, ] == 1]
        k <- max(1, length(true_set))

        constrained <- if (specification$kind == "none") character(0) else {
          constrained_variables(
            specification$scores, target, variables, tau, specification$kind
          )
        }

        # alpha path -> entry alpha for every variable
        entry <- rep(Inf, length(variables))
        names(entry) <- variables
        membership_at_default <- character(0)

        for (alpha in alphas) {
          mb <- tryCatch(
            quiet(if (is.null(specification$test)) {
              if (specification$kind == "blacklist") {
                # 空黑名单必须走"不带约束"的分支。
                # bnlearn 的 learn.mb(blacklist = character(0)) 会直接报
                # "at least 1 node(s) needed"，而外面有 tryCatch 兜底，
                # 于是错误被吞成空毯子、F1 被静默打成 0 —— 这个坑真的踩到过：
                # role 先验下 ASIA 的 28 对全是正面答案，黑名单对每个目标都是空的，
                # 结果 prior-blacklist 与 random-blacklist 双双掉到 0.28，
                # 而它们本该与基线一样。
                if (length(constrained) == 0) {
                  learn.mb(sample_frame, node = target, method = "iamb",
                           test = "mi", alpha = alpha)
                } else {
                  learn.mb(sample_frame, node = target, method = "iamb",
                           test = "mi", alpha = alpha, blacklist = constrained)
                }
              } else if (specification$kind == "whitelist") {
                # 同理：白名单为空时不能把"其余全部"当黑名单传进去。
                if (length(constrained) == 0) {
                  learn.mb(sample_frame, node = target, method = "iamb",
                           test = "mi", alpha = alpha)
                } else {
                  learn.mb(sample_frame, node = target, method = "iamb",
                           test = "mi", alpha = alpha, whitelist = constrained,
                           blacklist = setdiff(variables, c(target, constrained)))
                }
              } else {
                learn.mb(sample_frame, node = target, method = "iamb",
                         test = "mi", alpha = alpha)
              }
            } else {
              learn.mb(sample_frame, node = target, method = "iamb",
                       test = "custom-test", fun = specification$test,
                       alpha = alpha)
            }),
            error = function(error) {
              # 不能只返回空集就走：那会让一次报错悄悄变成"这个目标没有毯子"，
              # 直接污染 F1。这里计数并在结尾报告。
              failed_calls <<- failed_calls + 1
              if (failed_calls <= 3) {
                cat(sprintf("  [警告] %s/%s/alpha=%g 调用失败：%s\n",
                            network, target, alpha, conditionMessage(error)))
              }
              character(0)
            }
          )
          if (is.null(mb)) mb <- character(0)
          for (name in mb) {
            if (is.infinite(entry[[name]])) entry[[name]] <- alpha
          }
          if (abs(alpha - 0.05) < 1e-12) membership_at_default <- mb
        }

        # ranking for the matched-size budget.
        # For the forced-inclusion variant the alpha path is degenerate (the
        # result does not depend on alpha), so it is ranked by the model's own
        # confidence instead - that variant doubles as the "prior only, no data"
        # reference point.
        if (specification$kind == "whitelist") {
          confidence <- vapply(
            variables,
            function(v) if (v == target) -Inf else lookup(specification$scores, target, v),
            numeric(1)
          )
          ranking <- variables[order(-confidence, variables)]
        } else {
          # earliest entry wins, ties broken alphabetically
          ranking <- variables[order(entry[variables], variables)]
        }
        top_k <- ranking[seq_len(min(k, length(ranking)))]

        matched <- score_set(top_k, true_set)
        unmatched <- score_set(membership_at_default, true_set)

        detail <- rbind(detail, data.frame(
          network = network,
          sample_size = size,
          repeat_index = repeat_index,
          variant = variant,
          target = target,
          budget = "matched",
          precision = matched[["precision"]],
          recall = matched[["recall"]],
          f1 = matched[["f1"]],
          n_selected = length(top_k)
        ))
        detail <- rbind(detail, data.frame(
          network = network,
          sample_size = size,
          repeat_index = repeat_index,
          variant = variant,
          target = target,
          budget = "alpha0.05",
          precision = unmatched[["precision"]],
          recall = unmatched[["recall"]],
          f1 = unmatched[["f1"]],
          n_selected = length(membership_at_default)
        ))
      }

      elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
      cat(sprintf("  repeat %2d %-16s %.1fs\n", repeat_index, variant, elapsed))
    }
  }
}

if (nrow(detail) == 0) {
  cat("no results\n")
  quit(status = 1)
}

write.csv(
  detail,
  sprintf("%s/results/bnlearn_constraint_detail_%s.csv", project_root, tag),
  row.names = FALSE
)

keys <- unique(detail[, c("network", "variant", "budget")])
summary_table <- data.frame()

for (i in seq_len(nrow(keys))) {
  key <- keys[i, ]
  rows <- detail[
    detail$network == key$network &
    detail$variant == key$variant &
    detail$budget == key$budget,
  ]
  per_repeat <- aggregate(f1 ~ repeat_index, rows, mean)
  summary_table <- rbind(summary_table, data.frame(
    network = key$network,
    variant = key$variant,
    budget = key$budget,
    precision = round(mean(rows$precision), 4),
    recall = round(mean(rows$recall), 4),
    f1 = round(mean(rows$f1), 4),
    f1_std = round(sd(per_repeat$f1), 4),
    mean_selected = round(mean(rows$n_selected), 2)
  ))
}

summary_table <- summary_table[order(
  summary_table$network, summary_table$budget, summary_table$variant
), ]

write.csv(
  summary_table,
  sprintf("%s/results/bnlearn_constraint_summary_%s.csv", project_root, tag),
  row.names = FALSE
)

cat("\n")
print(summary_table, row.names = FALSE)
cat(sprintf("\nwritten: results/bnlearn_constraint_summary_%s.csv\n", tag))
cat(sprintf("调用失败次数: %d\n", failed_calls))
if (failed_calls > 0) {
  cat("注意：上面这个数字大于 0，说明有调用报错并被吞成空毯子，本次结果不可信。\n")
}

# Paired test against the baseline, per repeat.
paired <- data.frame()
for (network in unique(detail$network)) {
  for (budget in unique(detail$budget)) {
    block <- detail[detail$network == network & detail$budget == budget, ]
    per_repeat <- aggregate(
      f1 ~ repeat_index + variant, block, mean
    )
    base <- per_repeat[per_repeat$variant == "baseline", ]
    for (variant in setdiff(unique(per_repeat$variant), "baseline")) {
      treated <- per_repeat[per_repeat$variant == variant, ]
      merged <- merge(
        treated[, c("repeat_index", "f1")],
        base[, c("repeat_index", "f1")],
        by = "repeat_index", suffixes = c("_variant", "_baseline")
      )
      delta <- merged$f1_variant - merged$f1_baseline
      p <- tryCatch(
        suppressWarnings(wilcox.test(delta)$p.value),
        error = function(e) NA
      )
      paired <- rbind(paired, data.frame(
        network = network,
        budget = budget,
        variant = variant,
        repeats = length(delta),
        baseline_f1 = round(mean(merged$f1_baseline), 4),
        variant_f1 = round(mean(merged$f1_variant), 4),
        mean_delta = round(mean(delta), 4),
        sd_delta = round(sd(delta), 4),
        wins = sprintf("%d/%d", sum(delta > 0), length(delta)),
        wilcox_p = round(p, 4)
      ))
    }
  }
}

write.csv(
  paired,
  sprintf("%s/results/bnlearn_constraint_paired_%s.csv", project_root, tag),
  row.names = FALSE
)

cat("\npaired vs baseline:\n")
print(paired, row.names = FALSE)
