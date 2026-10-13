# How good is the prior as a Markov blanket predictor?
#
# The first prompt asked "is there a direct edge?"; the second asks "are the two
# variables in each other's Markov blanket?". This script scores both answers
# against the truth that actually matters for the downstream task - membership
# in the true Markov blanket - so the two questions can be compared directly.
#
# Run:
#   & 'D:\R\R-4.6.1\bin\Rscript.exe' --vanilla scripts\eval_mb_prior.R --networks=ASIA

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}

project_root <- "E:/Causal-LLM"
networks <- strsplit(get_arg("networks", "ASIA"), ",")[[1]]
modes <- strsplit(get_arg("modes", "real"), ",")[[1]]

rows <- data.frame()

for (network in networks) {

  truth <- as.matrix(read.table(sprintf(
    "%s/data/%s/%s_true_MB_matrix.txt", project_root, network, network
  )))
  header <- read.table(
    sprintf("%s/data/%s/%s_n100.txt", project_root, network, network),
    header = TRUE, sep = "\t", nrows = 1, check.names = FALSE
  )
  variables <- colnames(header)
  index <- setNames(seq_along(variables), variables)

  # every ordered pair inside a true blanket
  true_pairs <- c()
  for (i in seq_along(variables)) {
    for (j in seq_along(variables)) {
      if (i != j && truth[i, j] == 1) {
        true_pairs <- c(true_pairs, paste(sort(c(variables[i], variables[j])),
                                          collapse = "|"))
      }
    }
  }
  true_pairs <- unique(true_pairs)

  for (mode in modes) {
    for (task in c("direction", "mb")) {

      prefix <- if (task == "mb") "LLM_mb" else "LLM_prior"
      path <- sprintf(
        "%s/data/%s/%s_%s_%s_summary.csv", project_root, network, network,
        prefix, mode
      )
      if (!file.exists(path)) next

      table <- read.csv(path, stringsAsFactors = FALSE)

      predicted <- c()
      if (task == "mb") {
        flag <- tolower(as.character(table$in_markov_blanket)) == "true"
      } else {
        flag <- tolower(as.character(table$direct_edge)) == "true"
      }

      for (i in seq_len(nrow(table))) {
        if (!isTRUE(flag[i])) next
        key <- paste(sort(c(table$variable_a[i], table$variable_b[i])), collapse = "|")
        predicted <- c(predicted, key)
      }
      predicted <- unique(predicted)

      tp <- length(intersect(predicted, true_pairs))
      precision <- if (length(predicted)) tp / length(predicted) else 0
      recall <- if (length(true_pairs)) tp / length(true_pairs) else 0
      f1 <- if (precision + recall > 0) {
        2 * precision * recall / (precision + recall)
      } else 0

      rows <- rbind(rows, data.frame(
        network = network,
        mode = mode,
        question = task,
        claimed = length(predicted),
        truly_in_blanket = length(true_pairs),
        true_positives = tp,
        precision = round(precision, 3),
        recall = round(recall, 3),
        f1 = round(f1, 3)
      ))
    }
  }
}

if (nrow(rows) == 0) {
  cat("no priors found\n")
  quit(status = 1)
}

print(rows, row.names = FALSE)

write.csv(
  rows,
  sprintf("%s/results/mb_prior_quality.csv", project_root),
  row.names = FALSE
)
cat("\nwritten: results/mb_prior_quality.csv\n")
cat("\nquestion: direction = 'is there a direct edge?', mb = 'in the Markov blanket?'\n")
