## Run scalop::sigScores and scalop::permuteSigScores on the matrix/sigs
## prepared by validate_perm.py. Writes results back to the same IO dir.
##
## Invoked as:  Rscript validate_perm.R
## Reads:       $PYSCALOP_VALIDATE_DIR/{matrix.tsv, sigs.tsv}
## Writes:      $PYSCALOP_VALIDATE_DIR/{r_scores.tsv, r_perm.tsv, r_timing.tsv}

suppressPackageStartupMessages({
    library(scalop)
})

io <- Sys.getenv("PYSCALOP_VALIDATE_DIR")
stopifnot(io != "")

m <- as.matrix(read.table(file.path(io, "matrix.tsv"),
                          header = TRUE, row.names = 1,
                          sep = "\t", check.names = FALSE))

sigs_lines <- readLines(file.path(io, "sigs.tsv"))
sigs <- list()
for (line in sigs_lines) {
    parts <- strsplit(line, "\t", fixed = TRUE)[[1]]
    sigs[[parts[1]]] <- parts[-1]
}
cat(sprintf("R: read matrix %d x %d, %d signatures\n",
            nrow(m), ncol(m), length(sigs)))

expr_nbin <- as.integer(Sys.getenv("PYSCALOP_VALIDATE_NBIN", "30"))
expr_binsize <- as.integer(Sys.getenv("PYSCALOP_VALIDATE_BINSIZE", "100"))

set.seed(42)
t0 <- Sys.time()
scores <- scalop::sigScores(m = m, sigs = sigs,
                            expr.nbin = expr_nbin, expr.binsize = expr_binsize)
t_obs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(data.frame(cell = rownames(scores), scores, check.names = FALSE),
            file.path(io, "r_scores.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("R: sigScores took %.3fs\n", t_obs))

n_perm <- as.integer(Sys.getenv("PYSCALOP_VALIDATE_NPERM", "30"))
set.seed(42)
t0 <- Sys.time()
perm <- scalop::permuteSigScores(m = m, sigs = sigs, N = n_perm,
                                 alternative = "greater",
                                 expr.nbin = expr_nbin,
                                 expr.binsize = expr_binsize)
t_perm <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(perm, file.path(io, "r_perm.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("R: permuteSigScores (N=%d) took %.3fs\n", n_perm, t_perm))

write.table(data.frame(step = c("sigScores", "permuteSigScores"),
                       seconds = c(t_obs, t_perm)),
            file.path(io, "r_timing.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
