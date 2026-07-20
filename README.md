# Paralog Upregulation

Analysis pipeline investigating paralog gene upregulation as a dosage compensation
mechanism in aneuploid cancer cell lines, using genome-scale CRISPRi perturbation
data (Replogle 2022 and KOLF2.1J 2026).

## Overview

When a gene is lost due to chromosomal aneuploidy in cancer, its paralog may be
upregulated to compensate. This pipeline:

1. Quantifies paralog upregulation (Δz) across 10+ cell lines using Perturb-seq data
2. Computes empirical p-values via permutation testing
3. Assesses cross-cell-line consistency
4. Performs pathway enrichment (Enrichr/GSEA)
5. Analyses copy number vs expression coupling
6. Examines temporal dynamics in CD4+ T cells
7. Extends analysis to iPSC cells (KOLF2.1J 2026)

---

## Repository Structure

```
scripts/          Analysis scripts (run in numbered order)
data/raw/         Raw data files (not tracked by git — see Data section)
results/          Per-cell-line CSV outputs
figures/          PDF figures per cell line and cross-cell-line
```

---

## Environment Setup

All analysis uses the `paralog-dep` conda environment (Python 3.11).

```bash
conda create -n paralog-dep python=3.11
conda activate paralog-dep
conda install -c conda-forge anndata pandas numpy scipy seaborn matplotlib tqdm requests openpyxl
```

---

## Data

All raw data files live in `data/raw/` (excluded from git due to file size).

### Perturb-seq h5ad files (Replogle 2022)

Source: [Figshare 20029387](https://figshare.com/articles/dataset/20029387)

Download automatically:
```bash
python scripts/00_download_data.py
```

This downloads the following h5ad files:
- `K562_gwps_normalized_bulk_01.h5ad` — K562 genome-wide screen
- `rpe1_normalized_bulk_01.h5ad` — RPE1

Additional cell lines require preprocessing from raw data (see below):
- `K562_essential_pseudobulk_normalized.h5ad`
- `HCT116_pseudobulk_normalized.h5ad`
- `HEK293T_pseudobulk_normalized.h5ad`
- `melanoma_pseudobulk_normalized.h5ad`
- `cd4t_rest/stim8hr/stim48hr_pseudobulk_normalized.h5ad`
- `neuron_pseudobulk_normalized.h5ad`

### iPSC dataset (KOLF2.1J, Nature Biotechnology 2026)

Source: [Figshare Plus doi:10.25452/figshare.plus.27261219](https://plus.figshare.com/articles/dataset/27261219)

The file `KOLF_Pan_Genome_Q...h5ad` (176 GB) requires a Figshare account and API token.
Due to AWS WAF restrictions, download must use the Figshare API endpoint.
Recommended to download directly to an HPC cluster (e.g. Stanford Sherlock):

```bash
# On Sherlock — replace YOUR_TOKEN with your Figshare personal token
curl -L \
     --header "Authorization: token YOUR_TOKEN" \
     -o $GROUP_SCRATCH/paralog_upregulation/data/raw/iPSC_KOLF2_raw.h5ad \
     "https://api.figshare.com/v2/file/download/64650261"
```

Then preprocess on the HPC (requires ≥512 GB RAM due to matrix size):
```bash
# Submit as Slurm job on Sherlock
python scripts/17_preprocess_ipsc.py \
    --input $GROUP_SCRATCH/paralog_upregulation/data/raw/iPSC_KOLF2_raw.h5ad \
    --output $GROUP_SCRATCH/paralog_upregulation/data/raw/iPSC_KOLF2_pseudobulk_normalized.h5ad
```

Transfer the preprocessed file (~few GB) back locally:
```bash
scp <sunetid>@dtn.sherlock.stanford.edu:/path/to/iPSC_KOLF2_pseudobulk_normalized.h5ad \
    data/raw/
```

### External datasets (copy number / expression)

- `OmicsAbsoluteCNGene.csv` — DepMap absolute copy number (~980 cell lines)
- `CCLE_expression.csv` — CCLE RNA expression log2(TPM+1)
- `ProCan_protein_matrix_8498_averaged.txt` — ProCan protein abundance (~949 cell lines)
- `Model.csv` — DepMap model metadata (used to map Sanger SIDM IDs to ACH IDs)

These files are stored outside the repo at:
```
/Users/jonathanzhao/Desktop/Sheltzer Lab/Paralog Difference/data/
/Users/jonathanzhao/Desktop/Sheltzer Lab/Chromosome Compensation/
```

### Paralog pair lists

- `sig_37_paralog.xlsx` — 37 significant aneuploidy vulnerability paralog pairs
- `non_sig_paralog.xlsx` — non-significant paralog pairs (background)

---

## Running the Pipeline

Activate the conda environment first:
```bash
conda activate paralog-dep
```

### Step 1 — Compute Δz for all cell lines

```bash
python scripts/02_compute_delta_z.py --cell-line K562
python scripts/02_compute_delta_z.py --all   # all registered cell lines
```

Outputs per cell line to `results/{cell_line}/`:
- `sig_results.csv` — Δz for the 37 significant pairs
- `nonsig_results.csv` — Δz for non-significant pairs

### Step 2 — Visualise per-cell-line results

```bash
python scripts/03_compare_visualize.py --cell-line K562
python scripts/03_compare_visualize.py --all
```

Outputs to `figures/{cell_line}/`:
- `01_sig_delta_z_barplot.pdf` — per-pair Δz sorted by magnitude
- `02_sig_vs_nonsig_violin.pdf` — violin comparison
- `03_ecdf_comparison.pdf` — empirical CDF
- `04_delta_z_vs_identity.pdf` — Δz vs sequence identity

### Step 3 — Empirical p-values (permutation test)

```bash
# Sig pairs only
python scripts/15_per_pair_pvalues.py --cell-line K562

# All pairs (adds p-values to all_pairs_ranked.csv)
python scripts/15_per_pair_pvalues.py --cell-line K562 --all-pairs

# All cell lines at once
python scripts/15_per_pair_pvalues.py --all --all-pairs
```

### Step 4 — Rank all paralog pairs

```bash
python scripts/10_rank_all_pairs.py --cell-line K562
python scripts/10_rank_all_pairs.py --all
```

Output: `results/{cell_line}/all_pairs_ranked.csv`

### Step 5 — Cross-cell-line consistency

```bash
python scripts/04_cross_cell_line.py
```

Output: `results/cross_cell_line_consistency.csv`, `figures/cross_cell_line/`

### Step 6 — CD4+ T cell temporal dynamics

```bash
python scripts/09_cd4t_temporal.py
```

Output: `figures/cd4t_temporal/01_temporal_dynamics.pdf`

### Step 7 — Pathway enrichment (Enrichr)

```bash
# Significant pairs
python scripts/11_gsea.py

# Top Δz across all pairs
python scripts/11_gsea.py --mode top --dz-threshold 2.0
```

Outputs to `results/gsea/` and `figures/gsea/`

### Step 8 — Copy number vs expression correlation

```bash
python scripts/16_cn_expression_correlation.py
```

Outputs to `results/cn_expression_correlation.csv` and `figures/cross_cell_line/`:
- `16_cn_rna_protein_correlation.pdf` — violin comparison of Pearson r
- `16_cn_rna_density.pdf` — density plot of CN-RNA coupling with sig paralogs marked
- `16_cn_vs_rna_scatter.pdf` — median CN vs RNA trend (sig vs non-sig)
- `16_cn_vs_protein_scatter.pdf` — median CN vs protein trend

---

## Key Concepts

**Δz (delta z-score)**
The mean z-score of the paralog gene across all knockdown cells minus the mean
z-score across non-targeting control cells. Δz > 0 means the paralog is upregulated
when the dependency gene is knocked down.

**Empirical p-value**
Fraction of all ~9,700–11,700 perturbations (depending on cell line) that produce
a Δz ≥ the observed value for that paralog. One-sided test for upregulation.
BH FDR correction applied across all pairs jointly.

**Cell lines included**

| Cell line | Type | Source |
|-----------|------|---------|
| K562 | Chronic myelogenous leukaemia | Replogle 2022 |
| K562_essential | K562 essential gene subset | Replogle 2022 |
| RPE1 | Retinal pigment epithelium | Replogle 2022 |
| HCT116 | Colorectal carcinoma | X-Atlas |
| HEK293T | Embryonic kidney | X-Atlas |
| Melanoma | Melanoma | GSE291147 |
| CD4T (rest/8hr/48hr) | CD4+ T cells, 3 activation states | X-Atlas |
| Neuron | iPSC-derived neurons | X-Atlas |
| iPSC | KOLF2.1J induced pluripotent stem cells | Nature Biotech 2026 |

---

## Citation

Replogle et al. (2022). Mapping information-rich genotype-phenotype landscapes with
genome-scale Perturb-seq. *Cell* 185, 2559–2575.

KOLF2.1J iPSC Perturbation Cell Atlas. *Nature Biotechnology* (2026).
doi:10.1038/s41587-026-03199-w
