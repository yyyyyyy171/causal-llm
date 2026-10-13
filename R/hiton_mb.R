# HITON-PC / HITON-MB, a faithful port of the reference implementation that
# produced the baselines already in this repository.
#
# Why a port at all: bnlearn 5.2.1 removed hiton.mb() and there is no Rtools
# here to build an older bnlearn from source, but the plan requires HITON-MB.
# The reference is the Causal-Learner MATLAB code used for
# results/HITONMB_4networks_summary.csv:
#
#   Code/alg_MB/HITONMB/HITONMB_G2.m
#   Code/alg_MB/HITONMB/HITONPC_G2.m
#   Code/common/my_g2_test.m
#
# Getting this wrong is easy, so the port is checked against the MATLAB numbers
# before any prior is injected. The details that matter and that a naive
# implementation gets wrong:
#
#   * the test is a G^2 (likelihood ratio) test, not mutual information;
#   * it is only attempted when the average number of samples per cell is at
#     least 5 (otherwise the p-value is NaN);
#   * the degrees of freedom are corrected for empty cells:
#       df = sum over conditioning strata of
#            max(nx - 1 - #empty_x, 0) * max(ny - 1 - #empty_y, 0)
#   * HITON-PC removes a candidate only against subsets that contain the
#     candidate being tested (the `X != Y` guard in the MATLAB loop);
#   * HITON-MB adds spouses through the separating sets returned by HITON-PC
#     and does NOT prune afterwards.
#
# The prior enters through `scale`: every p-value is multiplied by
# scale(x, y) before it is compared with alpha. scale = NULL is the plain
# data-driven algorithm.
#
# ASCII-only source on purpose (R on a Chinese Windows locale).

# ---------------------------------------------------------------------------
# G^2 conditional independence test (port of my_g2_test.m / citpvalue)
# ---------------------------------------------------------------------------

# data is a matrix of integer codes, one column per variable, coded 1..ns.
# Returns p-value and test statistic; p is NA when the test is not reliable.
g2_test <- function(data, ns, x, y, condition = integer(0), alpha = 0.05) {

  variables <- c(x, y, condition)
  if (!is.numeric(variables) || anyNA(match(variables, seq_along(ns)))) {
    stop("g2_test expects integer variable indices, not names")
  }
  levels = ns[variables]
  n_samples <- nrow(data)

  # reliability: at least 5 samples per cell on average
  if (n_samples / prod(levels) < 5) {
    return(list(p = NA_real_, statistic = NA_real_, reliable = FALSE))
  }

  sub <- data[, variables, drop = FALSE]

  # flatten the contingency table into a vector, then restore its dims
  strides <- cumprod(c(1, levels[-length(levels)]))
  flat <- 1 + as.vector((sub - 1) %*% strides)
  observed <- tabulate(flat, nbins = prod(levels))
  dim(observed) <- levels

  n_test <- length(variables)
  nx <- levels[1]
  ny <- levels[2]
  rest <- prod(levels[-(1:2)])

  # Reshape to (x, y, stratum). The flat index above already varies x fastest,
  # then y, then the conditioning variables, which is R's column-major order.
  dim(observed) <- c(nx, ny, rest)

  # Margin counts per stratum.
  # Note the empty-conditioning case (n_test == 2, rest == 1): apply() must not
  # be given a margin that does not exist, which is exactly the bug that made
  # every single test fail silently and every Markov blanket come out empty.
  if (n_test == 2) {
    sum_x <- matrix(rowSums(observed), nrow = nx)
    sum_y <- matrix(colSums(observed), nrow = ny)
    sum_s <- sum(observed)
  } else {
    sum_x <- matrix(apply(observed, c(1, 3), sum), nrow = nx)
    sum_y <- matrix(apply(observed, c(2, 3), sum), nrow = ny)
    sum_s <- as.vector(apply(observed, 3, sum))
  }

  expected <- array(0, dim = c(nx, ny, rest))
  for (stratum in seq_len(rest)) {
    expected[, , stratum] <- outer(sum_x[, stratum], sum_y[, stratum]) /
      sum_s[stratum]
  }

  # degrees of freedom, corrected for empty cells in each stratum
  dof <- 0
  for (stratum in seq_len(rest)) {
    empty_x <- sum(sum_x[, stratum] == 0)
    empty_y <- sum(sum_y[, stratum] == 0)
    dof <- dof + max(nx - 1 - empty_x, 0) * max(ny - 1 - empty_y, 0)
  }

  if (dof == 0) {
    return(list(p = 1, statistic = 0, reliable = TRUE))
  }

  cells <- as.vector(observed)
  expectations <- as.vector(expected)
  positive <- cells > 0 & expectations > 0
  statistic <- 2 * sum(cells[positive] * log(cells[positive] / expectations[positive]))

  list(
    p = stats::pchisq(statistic, df = dof, lower.tail = FALSE),
    statistic = statistic,
    reliable = TRUE
  )
}

# wrapper that applies the prior scaling to the p-value
make_g2 <- function(data, ns, alpha = 0.05, scale = NULL) {
  function(x, y, condition = integer(0)) {
    result <- g2_test(data, ns, x, y, condition, alpha)
    if (!is.na(result$p) && !is.null(scale)) {
      result$p <- min(1, result$p * scale(x, y))
    }
    result
  }
}

# ---------------------------------------------------------------------------
# HITON-PC (port of HITONPC_G2.m)
# ---------------------------------------------------------------------------

all_subsets <- function(pool, size) {
  if (size == 0) return(list(integer(0)))
  if (size > length(pool)) return(list())
  # combn() treats a single number n as seq_len(n), so a one-element pool has to
  # be handled by hand. Without this guard combn(6, 1) enumerates 1:6 and the
  # search conditions on variables that are not even in the candidate set -
  # including the target itself - which returns p = 1 every time and silently
  # deletes true parents and children from the Markov blanket.
  if (length(pool) == 1) {
    return(if (size == 1) list(pool) else list())
  }
  utils::combn(pool, size, simplify = FALSE)
}

hiton_pc <- function(test, target, variables, max_k = 3, alpha = 0.05) {

  others <- setdiff(variables, target)
  sepset <- vector("list", max(variables))

  # step 1: unconditional screen
  cpc <- integer(0)
  dep <- numeric(0)

  for (candidate in others) {
    result <- test(candidate, target, integer(0))
    # an unreliable test counts as independent here (MATLAB: CI = 1)
    independent <- is.na(result$p) || result$p > alpha
    if (independent) next
    cpc <- c(cpc, candidate)
    dep <- c(dep, abs(result$statistic))
  }

  if (length(cpc) == 0) return(list(pc = integer(0), sepset = sepset))

  # step 2: order by strength of dependence, strongest first
  order_index <- order(dep, decreasing = TRUE)
  cpc <- cpc[order_index]

  pc <- integer(0)

  for (position in seq_along(cpc)) {

    y <- cpc[position]
    pc <- c(pc, y)
    pc_tmp <- pc
    last_break <- FALSE

    for (j in rev(seq_along(pc))) {

      x <- pc[j]
      can_pc <- setdiff(pc_tmp, x)
      break_flag <- FALSE
      cut_size <- 0

      while (length(can_pc) >= cut_size && cut_size <= max_k) {

        subsets <- all_subsets(can_pc, cut_size)

        for (z in subsets) {

          # when x is not the newly added variable, the subset must contain it
          if (x != y && !(y %in% z)) next

          result <- test(x, target, z)
          # an unreliable test counts as dependent here (MATLAB: CI = 0)
          independent <- !is.na(result$p) && result$p > alpha

          if (independent) {
            pc_tmp <- can_pc
            sepset[[x]] <- z
            break_flag <- TRUE
            if (x == y) last_break <- TRUE
            break
          }
        }

        if (break_flag || last_break) break
        cut_size <- cut_size + 1
      }

      if (last_break) break
    }

    pc <- pc_tmp
  }

  list(pc = pc, sepset = sepset)
}

# ---------------------------------------------------------------------------
# HITON-MB (port of HITONMB_G2.m)
# ---------------------------------------------------------------------------

hiton_mb <- function(test, target, variables, max_k = 3, alpha = 0.05) {

  found <- hiton_pc(test, target, variables, max_k, alpha)
  pc <- found$pc
  sepset <- found$sepset

  mb <- pc
  spouses <- integer(0)

  for (x in pc) {

    pc_of_x <- hiton_pc(test, x, variables, max_k, alpha)$pc

    for (y in pc_of_x) {

      if (y %in% pc || y == target) next

      separating <- sepset[[y]]
      if (length(separating) > 0 && x %in% separating) next

      condition <- c(separating, x)
      result <- test(y, target, condition)
      dependent <- is.na(result$p) || result$p <= alpha

      if (dependent) spouses <- union(spouses, y)
    }
  }

  union(mb, spouses)
}
