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

# 仓库根目录：本机 Windows 与 Colab/Linux 都能用。
# 优先环境变量 CAUSAL_LLM_ROOT；否则本机的 E:/Causal-LLM；再否则用当前工作目录。
project_root <- Sys.getenv(
  "CAUSAL_LLM_ROOT",
  if (dir.exists("E:/Causal-LLM")) "E:/Causal-LLM" else normalizePath(".")
)
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
# 先验来自哪个模型：8B 的结果文件名带 _qwen3-8b 后缀，留空则是 4B 的旧结果。
model_tag <- get_arg("model-tag", "")
# 先验用哪种提问方式生成的：
#   direction = 问“有没有直接边”（文件名前缀 LLM_prior）
#   mb        = 直接问“是否在马尔可夫毯里”（文件名前缀 LLM_mb）
#   role      = 直接问“A 相对 B 是父/子/配偶”（文件名前缀 LLM_role）
# 三者语义不同：direction 要先翻译成邻接信号，mb 与 role 本身就是要用的对象，
# 其中 role 还会把配偶作为正面答案保留下来。
prior_task <- get_arg("prior-task", "direction")
tag <- get_arg("tag", sprintf("both_n%s", mode))

alphas <- as.numeric(strsplit(
  get_arg("alphas", "1e-06,1e-04,0.001,0.01,0.05,0.2"), ","
)[[1]])

# 目标变量子集。
#
# 代价随变量数爆炸：每个重复要对每个目标各跑一遍 IAMB 与 HITON-MB，
# HAILFINDER 有 56 个变量，全量跑是几十小时。
# 这里的子集由固定种子抽出而不是人工挑选：可复现，也不会挑好看的变量。
# 0（默认）表示全部变量，ASIA / CHILD / INSURANCE 都用全量。
target_limit <- as.integer(get_arg("max-targets", "0"))

# 每跑完几次重复落一次盘（默认 1 = 每次都落）。
# 大网络单次重复要几分钟，落盘本身是秒级以内，所以默认每轮都写最稳妥。
checkpoint_every <- as.integer(get_arg("checkpoint-every", "1"))

# 第四组对照：不用任何先验内容，只把显著性阈值整体收紧一档。
#
# 为什么需要它：先验进入算法的方式是把 p 值乘以 exp(-beta * score)，
# 也就是逐对改动有效阈值。如果观察到的提升其实只是"阈值动了一下"，
# 那么一个不带任何语义、只做全局阈值改动的对照组应当能复现同样的提升。
#
# 取值方式：在脚本已有的 alpha 路径上往前挪一格（0.05 -> 0.01），
# 这样它不是拍脑袋定的数，也和先验那种"收紧"方向一致；
# 需要别的值可以显式指定 --alpha-only=<数值>。
alpha_index <- which.min(abs(log(alphas) - log(alpha)))
alpha_only_default <- alphas[max(1, alpha_index - 1)]
alpha_only_value <- as.numeric(get_arg("alpha-only", alpha_only_default))

# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

prior_path <- function(network) {
  sprintf(
    "%s/data/%s/%s_%s_%s%s_summary.csv", project_root, network, network,
    switch(prior_task, mb = "LLM_mb", role = "LLM_role", "LLM_prior"), mode,
    if (nzchar(model_tag)) paste0("_", model_tag) else ""
  )
}

load_prior <- function(network) {
  path <- prior_path(network)
  if (!file.exists(path)) return(NULL)
  table <- read.csv(path, stringsAsFactors = FALSE)
  scores <- list()
  for (i in seq_len(nrow(table))) {
    a <- table$variable_a[i]; b <- table$variable_b[i]
    confidence <- suppressWarnings(as.numeric(table$final_confidence[i]))
    if (is.na(confidence)) confidence <- 0
    confidence <- min(max(confidence, 0), 1)
    if (prior_task == "role") {
      # 角色版：parent / child / spouse 三者都是"在毯内"，是正面信号；
      # "none" 才是负面；"uncertain" 记 0（不表态，不给分也不扣分）。
      # 这样配偶不会再因为"没有直接边"而被当成负分 —— 那正是旧问法的病根。
      role <- tolower(as.character(table$final_role[i]))
      if (role %in% c("parent", "child", "spouse")) {
        score <- confidence
      } else if (role == "none") {
        score <- -confidence
      } else {
        score <- 0
      }
    } else if (prior_task == "mb") {
      # mb 版先验直接回答毯成员资格，正是算法需要的东西，不用再翻译
      in_mb <- tolower(as.character(table$in_markov_blanket[i])) == "true"
      score <- if (in_mb) confidence else -confidence
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

# ---------------------------------------------------------------------------
# 落盘：把当前已经跑出来的结果写成三张表
# ---------------------------------------------------------------------------
#
# 为什么做成函数并在循环里调用：之前是所有网络、所有重复都跑完才写一次文件，
# 于是任何一次中断（掉电、手滑、要腾出机器、会话被回收）都意味着几小时的算力
# 全部作废 —— ALARM 与 HAILFINDER 单轮就是几十分钟量级。
# 现在每跑完一次重复就落盘一次，中断后 results/ 下至少是"跑到哪算哪"的表。
#
# 开销很小：明细表每行 8 个数，几千到几万行；汇总与配对检验是对已算好的
# 明细做分组，秒级以内。用 --checkpoint-every=N 可以调稀。

dir.create(sprintf("%s/results", project_root), showWarnings = FALSE)

write_outputs <- function(detail, print_tables = FALSE) {

  detail_path <- sprintf("%s/results/fusion_both_detail_%s.csv", project_root, tag)
  write.csv(detail, detail_path, row.names = FALSE)

  if (nrow(detail) == 0) return(invisible(NULL))

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
      f1_min = round(min(per_repeat$f1), 4),
      f1_max = round(max(per_repeat$f1), 4),
      recall_adj = round(mean(block$recall_adj, na.rm = TRUE), 4),
      recall_spouse = round(mean(block$recall_spouse, na.rm = TRUE), 4),
      mean_selected = round(mean(block$n_selected), 2)
    ))

    if (key$variant != "baseline") {

      baseline_block <- detail[
        detail$network == key$network &
        detail$sample_size == key$sample_size &
        detail$method == key$method &
        detail$variant == "baseline" &
        detail$budget == key$budget, ]

      base <- aggregate(
        f1 ~ repeat_index,
        baseline_block,
        mean
      )
      merged <- merge(per_repeat, base, by = "repeat_index",
                      suffixes = c("_v", "_b"))
      delta <- merged$f1_v - merged$f1_b
      p <- tryCatch(suppressWarnings(wilcox.test(delta)$p.value),
                    error = function(e) NA)

      # 父/子与配偶分开做配对检验。
      # 这是专门用来回答"先验会不会误伤配偶节点"的：只报总 F1 的话，
      # "帮了父/子"与"伤了配偶"两件事会互相抵消，看起来像"没效果"。
      paired_metric <- function(column) {
        variant_per_repeat <- aggregate(
          block[[column]], list(repeat_index = block$repeat_index),
          mean, na.rm = TRUE
        )
        baseline_per_repeat <- aggregate(
          baseline_block[[column]],
          list(repeat_index = baseline_block$repeat_index),
          mean, na.rm = TRUE
        )
        merged_metric <- merge(variant_per_repeat, baseline_per_repeat,
                               by = "repeat_index")
        difference <- merged_metric$x.x - merged_metric$x.y
        list(
          baseline = mean(merged_metric$x.y, na.rm = TRUE),
          variant = mean(merged_metric$x.x, na.rm = TRUE),
          delta = mean(difference, na.rm = TRUE),
          p = tryCatch(
            suppressWarnings(wilcox.test(difference)$p.value),
            error = function(e) NA
          )
        )
      }

      spouse <- paired_metric("recall_spouse")
      adjacent_metric <- paired_metric("recall_adj")

      paired <- rbind(paired, data.frame(
        network = key$network, sample_size = key$sample_size,
        method = key$method, variant = key$variant, budget = key$budget,
        baseline_f1 = round(mean(merged$f1_b), 4),
        variant_f1 = round(mean(merged$f1_v), 4),
        mean_delta = round(mean(delta), 4),
        delta_sd = round(sd(delta), 4),
        wins = sprintf("%d/%d", sum(delta > 0), length(delta)),
        wilcox_p = round(p, 4),
        adj_baseline = round(adjacent_metric$baseline, 4),
        adj_variant = round(adjacent_metric$variant, 4),
        adj_delta = round(adjacent_metric$delta, 4),
        spouse_baseline = round(spouse$baseline, 4),
        spouse_variant = round(spouse$variant, 4),
        spouse_delta = round(spouse$delta, 4),
        spouse_p = round(spouse$p, 4)
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

  if (print_tables) {
    cat("\n")
    print(rows, row.names = FALSE)
    cat("\npaired vs baseline:\n")
    print(paired, row.names = FALSE)
  }

  invisible(list(summary = rows, paired = paired))
}

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

  # DAG 邻接关系：用来把真实毯内成员拆成"父/子"与"配偶"两组。
  # 为什么需要：旧问法把配偶判成"没有直接边"，而融合算法把"没有直接边"
  # 当作负分，于是配偶会被系统性地推出毯子。只看总 F1 的话，
  # "帮了父/子、伤了配偶"会被平均掉，必须分开量。
  dag <- read.csv(
    sprintf("%s/data/%s/%s_dag_edges.csv", project_root, network, network),
    stringsAsFactors = FALSE
  )
  adjacent <- function(a, b) {
    any((dag$parent == a & dag$child == b) | (dag$parent == b & dag$child == a))
  }

  prior_real <- load_prior(network)

  # 没有先验文件时必须立刻停下来。
  # 之前这里是静默继续：prior 变体全部退化成 baseline，跑出来的表看起来像
  # “先验没有效果”，其实是文件根本没找到。这种降级比报错有害得多。
  if (is.null(prior_real)) {
    stop(sprintf(
      paste0("找不到 %s 的先验文件，拒绝用空先验跑融合。\n",
             "  期望路径: %s\n",
             "  当前 --mode=%s，--prior-task=%s，--model-tag='%s'\n",
             "  先跑 Colab（或本机）生成先验；文件名后缀要跟 --model-tag 完全一致。"),
      network, prior_path(network), mode, prior_task, model_tag
    ), call. = FALSE)
  }

  cat(sprintf(
    "%s: %d variables, prior(%s) pairs = %d\n", network, ncol(full_raw),
    prior_task,
    if (is.null(prior_real)) 0 else length(prior_real)
  ))

  if (target_limit > 0 && target_limit < length(labels)) {
    set.seed(seed)
    target_indices <- sort(sample(seq_along(labels), target_limit))
    cat(sprintf(
      "  只跑 %d/%d 个目标变量（种子 %d 固定抽取）：%s\n",
      length(target_indices), length(labels), seed,
      paste(labels[target_indices], collapse = ", ")
    ))
  } else {
    target_indices <- seq_along(labels)
  }

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

      # 四组对照，其余参数（beta / 重复数 / 抽样种子 / 目标集合）完全一致：
      #   baseline      纯数据，不注入任何先验
      #   prior-thresh  真实 LLM 先验，逐对缩放 p 值
      #   random-thresh 把先验分数在同网络内打乱，作为"任意扰动"的对照
      #   alpha-only    不含任何先验内容，只做等量的全局阈值改动
      variants <- list(
        baseline = list(scores = NULL, use = FALSE, alpha = alpha),
        "prior-thresh" = list(scores = prior_centered, use = TRUE, alpha = alpha),
        "random-thresh" = list(scores = shuffled_centered, use = TRUE, alpha = alpha),
        "alpha-only" = list(scores = NULL, use = FALSE, alpha = alpha_only_value)
      )

      for (variant in names(variants)) {

        scores <- variants[[variant]]$scores
        use_prior <- variants[[variant]]$use
        # 这一组自己的显著性水平（alpha-only 与其余三组不同）
        variant_alpha <- variants[[variant]]$alpha

        for (target_index in target_indices) {

          target_name <- labels[target_index]
          true_set <- labels[which(truth[target_index, ] > 0)]
          k <- max(1, length(true_set))

          # 把真实毯内成员拆成"父/子"（在 DAG 里相邻）与"配偶"（不相邻）
          adj_set <- true_set[
            vapply(true_set, function(m) adjacent(target_name, m), logical(1))
          ]
          spouse_set <- setdiff(true_set, adj_set)

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
              if (abs(alpha_value - variant_alpha) < 1e-12) raw_set <- found
            }

            ranking <- labels[order(entry[labels], labels)]
            top_k <- ranking[seq_len(min(k, length(ranking)))]

            raw_metrics <- score_set(raw_set, true_set)
            matched_metrics <- score_set(top_k, true_set)

            # 分开量父/子与配偶的召回率：总 F1 会把"帮了父/子、伤了配偶"平均掉
            recall_of <- function(found, members) {
              if (length(members) == 0) return(NA_real_)
              length(intersect(found, members)) / length(members)
            }

            detail <- rbind(detail,
              data.frame(
                network = network, sample_size = size,
                repeat_index = repeat_index, method = method,
                variant = variant, budget = "raw",
                precision = raw_metrics[["precision"]],
                recall = raw_metrics[["recall"]],
                f1 = raw_metrics[["f1"]],
                recall_adj = recall_of(raw_set, adj_set),
                recall_spouse = recall_of(raw_set, spouse_set),
                n_selected = length(raw_set)
              ),
              data.frame(
                network = network, sample_size = size,
                repeat_index = repeat_index, method = method,
                variant = variant, budget = "matched",
                precision = matched_metrics[["precision"]],
                recall = matched_metrics[["recall"]],
                f1 = matched_metrics[["f1"]],
                recall_adj = recall_of(top_k, adj_set),
                recall_spouse = recall_of(top_k, spouse_set),
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

      # 每跑完一次重复就落盘，中断了也不会全丢（见 write_outputs 的说明）
      if (repeat_index %% checkpoint_every == 0) {
        write_outputs(detail)
      }
    }
  }
}

# ---------------------------------------------------------------------------
# 全部跑完：再写一次（这次把表打印到控制台）
# ---------------------------------------------------------------------------

write_outputs(detail, print_tables = TRUE)
