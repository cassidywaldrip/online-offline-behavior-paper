# Paper assets

`make_paper_assets.py` turns the per-outcome grid outputs into the paper's figures and tables.
Run it from the repository root once every outcome's `grid.py` (and the robustness scripts, for the
tables they feed) has been run:

```bash
PYTHONPATH="$PWD" python paper_assets/make_paper_assets.py
```

The committed `figs/` (PDF+PNG) and `tables/` (LaTeX+CSV) are the exact outputs it produces.

## What it reads
- `<outcome>/figures/<outcome>_metrics_by_combination.csv` — the main combination grid (per outcome;
  `cov_vac` is `refuser_metrics_by_combination.csv`).
- `<outcome>/figures/<outcome>_fb1_neighbor_demo_grid.csv` — FB1 (Table 6).
- `<outcome>/figures/<outcome>_fb2_predict_demographics.csv` — FB2 (Table 5).
- `signal_cache/<outcome>_signal.csv`, `demo_cache/<outcome>_demo.csv` — per-person signal volume and
  the sample-characteristics demographics (the latter built by `build_demo_cache.py`).

## What it writes

Figures (`figs/`):
- `fig1_within_party_lift` — Demo → best-model dumbbell, by party, with 95% CIs (the within-party lift).
- `fig2_structure_beats_text` — single-block Overall ROC-AUC per outcome (network beats text among traces).
- `fig3_precision_at_topk` — precision @ top-10% flagged for the anchor outcome (operational value).
- `fig4_lift_decomposition` — Shapley decomposition of the best model's AUC into Individual vs Relational.

Tables (`tables/`):
- `table1_outcomes`, `table2_within_party`, `table3_covvac_grid`, `table4_baseline_caveat`,
  `table5_predict_demographics` (FB2), `table6_neighbor_demo` (FB1), `table7_signal_volume`,
  `table8_sample_characteristics`, `table9_pca_dims`, `table10_lift_decomposition` (Shapley),
  `appendixA_{age,gender}_subgroups`, `appendixB_keywords`,
  `appendixC_divergence_agg` (from FB3) + `appendixC_divergence_examples` (hand-written, de-identified).

> The manuscript's figure/table **numbers** differ from these filenames (the files are named by
> content and generation order, not by their position in the paper). `appendixC_divergence_examples.tex`
> contains a `\citet{joseph2021}` placeholder — replace with the real bib key.

## Helper
- `build_demo_cache.py` — builds `demo_cache/<outcome>_demo.csv` (the decoded demographics of each
  outcome's exact modeled population) for the sample-characteristics table. Run before `table8`.
