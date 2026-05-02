#!/usr/bin/env Rscript
# 06_convert_rds.R
# ----------------
# Converts GSE291147 RDS count matrices to Matrix Market (.mtx) format so that
# Python (scipy.io.mmread) can read them without requiring rpy2.
#
# Requires R packages: Matrix  (base R, usually available)
#
# Input files (data/raw/):
#   GSE291147_Dual_omics_RNA_gene_count_matrix.RDS   genes x cells sparse matrix
#   GSE291147_Dual_omics_sgRNA_count_matrix.RDS      guides x cells sparse matrix
#
# Output files (data/raw/):
#   GSE291147_RNA_matrix.mtx          Matrix Market sparse matrix
#   GSE291147_RNA_matrix_rownames.csv gene names (rows)
#   GSE291147_RNA_matrix_colnames.csv cell barcodes (cols)
#   GSE291147_sgRNA_matrix.mtx        (optional, same pattern)
#   GSE291147_sgRNA_matrix_rownames.csv
#   GSE291147_sgRNA_matrix_colnames.csv
#
# Usage:
#   Rscript scripts/06_convert_rds.R

suppressPackageStartupMessages(library(Matrix))

# ── Resolve project root ──────────────────────────────────────────────────────
args      <- commandArgs(trailingOnly = FALSE)
file_arg  <- grep("--file=", args, value = TRUE)
if (length(file_arg)) {
  script_path <- normalizePath(sub("--file=", "", file_arg))
} else {
  script_path <- normalizePath("scripts/06_convert_rds.R")
}
root_dir <- dirname(dirname(script_path))
data_dir <- file.path(root_dir, "data", "raw")
cat(sprintf("Project root : %s\n", root_dir))
cat(sprintf("Data dir     : %s\n", data_dir))


# ── Helper ────────────────────────────────────────────────────────────────────
convert_rds <- function(rds_path, out_prefix) {
  if (!file.exists(rds_path)) {
    cat(sprintf("[skip] %s not found\n", basename(rds_path)))
    return(invisible(NULL))
  }

  cat(sprintf("\nConverting %s ...\n", basename(rds_path)))
  obj <- readRDS(rds_path)

  # Coerce to dgCMatrix regardless of source type
  if (is(obj, "sparseMatrix")) {
    mat <- as(obj, "CsparseMatrix")
  } else if (is.matrix(obj) || is.data.frame(obj)) {
    mat <- as(as.matrix(obj), "sparseMatrix")
  } else {
    stop(sprintf(
      "Unsupported object class: %s\n",
      paste(class(obj), collapse = ", ")
    ))
  }

  # Write Matrix Market file
  mtx_path <- paste0(out_prefix, ".mtx")
  writeMM(mat, mtx_path)

  # Write row and column name tables
  write.csv(
    data.frame(name = rownames(mat)),
    paste0(out_prefix, "_rownames.csv"),
    row.names = FALSE, quote = FALSE
  )
  write.csv(
    data.frame(name = colnames(mat)),
    paste0(out_prefix, "_colnames.csv"),
    row.names = FALSE, quote = FALSE
  )

  cat(sprintf("  Dimensions : %d rows × %d cols\n", nrow(mat), ncol(mat)))
  cat(sprintf("  Written    : %s\n", basename(mtx_path)))
  cat(sprintf("             + _rownames.csv  _colnames.csv\n"))
}


# ── Convert RNA count matrix ──────────────────────────────────────────────────
convert_rds(
  file.path(data_dir, "GSE291147_Dual_omics_RNA_gene_count_matrix.RDS"),
  file.path(data_dir, "GSE291147_RNA_matrix")
)

# ── Convert sgRNA count matrix (optional but helps perturbation assignment) ───
convert_rds(
  file.path(data_dir, "GSE291147_Dual_omics_sgRNA_count_matrix.RDS"),
  file.path(data_dir, "GSE291147_sgRNA_matrix")
)

cat("\nDone. Next step:\n")
cat("  python scripts/06_preprocess_gse291147.py\n")
