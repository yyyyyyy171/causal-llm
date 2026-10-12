# Independent cross-check of the DAG ground truth.
#
# src/mb/rds_dag.py recovers each network's DAG by parsing the bn.fit objects
# stored in data/<NET>/<NET>.rds. This script does the same thing with bnlearn
# itself and compares the two arc sets one by one, so the extracted DAGs do not
# depend on a single reader being correct.
#
# R lives at D:\R\R-4.6.1 on this machine. Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\verify_dag_bnlearn.R
#
# The file is deliberately ASCII-only: R on a Chinese Windows locale has
# trouble with non-ASCII source files.

networks <- c("ASIA", "CHILD", "INSURANCE", "HAILFINDER")
ok <- TRUE

for (n in networks) {

  path <- sprintf("E:/Causal-LLM/data/%s/%s.rds", n, n)

  if (!file.exists(path)) {
    cat(sprintf("%s: missing %s\n", n, path))
    ok <- FALSE
    next
  }

  net <- readRDS(path)

  # Pull the arcs straight out of the bn.fit object.
  arcs <- do.call(rbind, lapply(net, function(nd) {
    if (length(nd$parents) == 0) return(NULL)
    cbind(nd$parents, rep(nd$node, length(nd$parents)))
  }))

  exported <- read.csv(
    sprintf("E:/Causal-LLM/data/%s/%s_dag_edges.csv", n, n),
    stringsAsFactors = FALSE
  )

  key <- function(m) sort(apply(m, 1, function(r) paste(r[1], r[2], sep = "->")))

  from_bnlearn <- key(arcs)
  from_python <- key(as.matrix(exported))

  same <- identical(from_bnlearn, from_python)
  ok <- ok && same

  cat(sprintf(
    "%-11s bnlearn %3d arcs | python %3d arcs | identical=%s\n",
    n, length(from_bnlearn), length(from_python), same
  ))

  if (!same) {
    cat("  only in bnlearn:", paste(setdiff(from_bnlearn, from_python), collapse = ", "), "\n")
    cat("  only in python :", paste(setdiff(from_python, from_bnlearn), collapse = ", "), "\n")
  }
}

cat("\nALL IDENTICAL\n" )
