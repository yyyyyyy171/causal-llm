# Why an adjacency prior is the wrong object for Markov blanket recovery.
#
# A Markov blanket contains parents, children AND spouses. Spouses are in the
# blanket but are NOT adjacent to the target, so when the language model is
# asked "is there a direct edge?" it answers "no edge" for every spouse pair.
# Using that answer as a blacklist therefore deletes true blanket members.
#
# This script quantifies the mismatch on every network: of the pairs that really
# belong to a target's blanket, how many are adjacent in the DAG (parents and
# children) and how many are spouses, and what the model answered for each group.
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\diagnose_prior_vs_spouses.R

# 仓库根目录：本机 Windows 与 Colab/Linux 都能用。
# 优先环境变量 CAUSAL_LLM_ROOT；否则本机的 E:/Causal-LLM；再否则用当前工作目录。
project_root <- Sys.getenv(
  "CAUSAL_LLM_ROOT",
  if (dir.exists("E:/Causal-LLM")) "E:/Causal-LLM" else normalizePath(".")
)
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

networks <- strsplit(
  get_arg("networks", "ASIA,CHILD,INSURANCE,ALARM,HAILFINDER"), ","
)[[1]]
mode <- get_arg("mode", "real")
# 先验来自哪个模型：8B 的结果文件名带后缀（如 _qwen3-8b-awq），
# 留空读的是旧的无后缀文件。不加这个开关就会拿 4B 的先验去解释 8B 的结论。
model_tag <- get_arg("model-tag", "")
suffix <- if (nzchar(model_tag)) paste0("_", model_tag) else ""
# 先验用哪种问法生成的：direction / mb / role。文件名前缀随之变化。
prior_task <- get_arg("prior-task", "direction")
out_name <- get_arg(
  "out",
  sprintf("prior_vs_spouses_%s%s%s.csv", mode, suffix,
          if (prior_task == "direction") "" else paste0("_", prior_task))
)

rows <- data.frame()

for (network in networks) {

  truth_path <- sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )
  dag_path <- sprintf(
    "%s/data/%s/%s_dag_edges.csv", project_root, network, network
  )
  prior_path <- sprintf(
    "%s/data/%s/%s_%s_%s%s_summary.csv",
    project_root, network, network,
    switch(prior_task, mb = "LLM_mb", role = "LLM_role", "LLM_prior"),
    mode, suffix
  )

  if (!file.exists(truth_path) || !file.exists(dag_path)) {
    cat(sprintf("%s: missing truth or DAG, skipped\n", network))
    next
  }

  truth <- as.matrix(read.table(truth_path))
  header <- read.table(
    sprintf("%s/data/%s/%s_n100.txt", project_root, network, network),
    header = TRUE, sep = "\t", nrows = 1, check.names = FALSE
  )
  variables <- colnames(header)

  dag <- read.csv(dag_path, stringsAsFactors = FALSE)
  adjacent <- function(a, b) {
    any((dag$parent == a & dag$child == b) | (dag$parent == b & dag$child == a))
  }

  if (!file.exists(prior_path)) {
    cat(sprintf("%s: no prior yet (mode=%s)\n", network, mode))
    next
  }
  prior <- read.csv(prior_path, stringsAsFactors = FALSE)

  # 把不同问法的回答统一成两个判断：是否相邻（edge）、是否在马尔可夫毯内（in_blanket）。
  #
  # role 版必须单独处理：它的"配偶"是**正面**答案——在毯内、但不相邻。
  # 只数"有没有边"（旧版的做法）会把配偶算成否定，恰好丢掉这一版最重要的信息。
  # mb 版不问方向、也不区分相邻与配偶，只能给"在不在毯内"，edge 记为 NA。
  answer <- function(a, b) {
    hit <- prior[
      (prior$variable_a == a & prior$variable_b == b) |
      (prior$variable_a == b & prior$variable_b == a), ]
    if (nrow(hit) == 0) return(NULL)

    if (prior_task == "role") {
      role <- tolower(as.character(hit$final_role[1]))
      return(list(edge = role %in% c("parent", "child"),
                  in_blanket = role %in% c("parent", "child", "spouse")))
    }

    if (prior_task == "mb") {
      inside <- tolower(as.character(hit$in_markov_blanket[1])) == "true"
      return(list(edge = NA, in_blanket = inside))
    }

    relation <- as.character(hit$final_relation[1])
    adjacent_answer <- relation %in% c("A_to_B", "B_to_A")
    # direction 版里"答有边"等价于"在毯内"；答"没有直接边"则无法区分配偶与无关
    list(edge = adjacent_answer, in_blanket = adjacent_answer)
  }

  n_child_parent <- 0; n_spouse <- 0
  said_edge_for_cp <- 0; said_noedge_for_cp <- 0
  said_edge_for_spouse <- 0; said_noedge_for_spouse <- 0
  said_inblanket_for_cp <- 0; said_inblanket_for_spouse <- 0
  predicted_edge_total <- 0; predicted_edge_in_blanket <- 0

  blanket_of <- function(name) {
    index <- match(name, variables)
    variables[truth[index, ] == 1]
  }

  for (i in seq_along(variables)) {
    members <- variables[truth[i, ] == 1]
    for (member in members) {
      ans <- answer(variables[i], member)
      if (is.null(ans)) next
      if (adjacent(variables[i], member)) {
        n_child_parent <- n_child_parent + 1
        if (isTRUE(ans$edge)) said_edge_for_cp <- said_edge_for_cp + 1
        if (identical(ans$edge, FALSE)) said_noedge_for_cp <- said_noedge_for_cp + 1
        if (isTRUE(ans$in_blanket)) said_inblanket_for_cp <- said_inblanket_for_cp + 1
      } else {
        n_spouse <- n_spouse + 1
        if (isTRUE(ans$edge)) said_edge_for_spouse <- said_edge_for_spouse + 1
        if (identical(ans$edge, FALSE)) said_noedge_for_spouse <- said_noedge_for_spouse + 1
        if (isTRUE(ans$in_blanket)) said_inblanket_for_spouse <- said_inblanket_for_spouse + 1
      }
    }
  }

  # Precision of the model's "direct edge" answer, scored against blanket
  # membership instead of against DAG adjacency.
  for (i in seq_len(nrow(prior))) {
    a <- prior$variable_a[i]; b <- prior$variable_b[i]
    if (a %in% variables && b %in% variables) {
      ans <- answer(a, b)
      if (!is.null(ans) && isTRUE(ans$edge)) {
        predicted_edge_total <- predicted_edge_total + 1
        if (b %in% blanket_of(a)) predicted_edge_in_blanket <- predicted_edge_in_blanket + 1
      }
    }
  }

  rows <- rbind(rows, data.frame(
    network = network,
    blanket_pairs = n_child_parent + n_spouse,
    child_or_parent = n_child_parent,
    spouse = n_spouse,
    said_edge_for_child_parent = said_edge_for_cp,
    said_noedge_for_child_parent = said_noedge_for_cp,
    said_edge_for_spouse = said_edge_for_spouse,
    said_noedge_for_spouse = said_noedge_for_spouse,
    said_inblanket_for_child_parent = said_inblanket_for_cp,
    said_inblanket_for_spouse = said_inblanket_for_spouse,
    predicted_edges = predicted_edge_total,
    predicted_edges_in_blanket = predicted_edge_in_blanket,
    precision_vs_blanket = round(predicted_edge_in_blanket / max(1, predicted_edge_total), 3)
  ))

  cat(sprintf(
    "%-11s blanket pairs=%3d (parents/children %3d, spouses %3d) | model said edge for %d/%d of parents-children, %d/%d of spouses\n",
    network, n_child_parent + n_spouse, n_child_parent, n_spouse,
    said_edge_for_cp, n_child_parent, said_edge_for_spouse, n_spouse
  ))
  cat(sprintf(
    "            said IN BLANKET for %d/%d of parents-children, %d/%d of spouses\n",
    said_inblanket_for_cp, n_child_parent, said_inblanket_for_spouse, n_spouse
  ))
  cat(sprintf(
    "            model claimed %d edges, %d of them are real blanket pairs -> precision vs blanket = %.3f\n",
    predicted_edge_total, predicted_edge_in_blanket,
    predicted_edge_in_blanket / max(1, predicted_edge_total)
  ))
}

if (nrow(rows) > 0) {
  write.csv(
    rows,
    sprintf("%s/results/%s", project_root, out_name),
    row.names = FALSE
  )
  cat(sprintf("\nwritten: results/%s\n", out_name))
}
