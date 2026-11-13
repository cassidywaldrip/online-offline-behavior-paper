# Robustness checks

Cross-cutting checks that probe whether the relational signal is what it claims to be. Each reads
an outcome's `config.py`, so run it with that outcome directory on `PYTHONPATH`, e.g.
`PYTHONPATH="$PWD/cand20" python robustness/<script>.py`.

### FB1 — `fb1_neighbor_demographics.py`  (→ Table 6)
Does the audience-lean (**Net**) add predictive value *over* neighbours' **demographics**? Builds a
relational block from external panelists' party/gender/age target-encoded onto followees
(**NbrDemo**), then scores `Demo`, `NbrDemo`, `Net` and their unions by party. Headline: the
marginal AUC of adding `Net` to `Demo+NbrDemo`.

> **Reconstructed and validated.** The original FB1 build script was not preserved; this
> re-implements the documented method and was checked by re-running it on candidate choice —
> it reproduces the committed `cand20_fb1_neighbor_demo_grid.csv` **exactly** (max |AUC diff|
> = 0.0000; neighbour-demographic features identical). It reuses each outcome's own `dataprep`
> helpers so NbrDemo is encoded from the same external-panelist universe as the audience-lean.

### FB2 — `fb2_predict_demographics.py`  (→ Table 5)
Are the traces just a demographic proxy? Predicts the behaviour **and** each demographic attribute
(party ×3, gender ×2, age ×4, one-vs-rest) from the trace blocks alone. Demographic legibility is
modest and flat across outcomes, while behaviour prediction swings much more widely — so the traces
carry outcome-specific signal beyond demographic inference.

### FB3 — `fb3_diverging_users.py`  (→ Appendix C)
Who are the respondents whose traces diverge from their survey answer? Reconstructs a trace-only
out-of-fold prediction and contrasts confidently-misclassified **high-signal** respondents with the
rest. The diverging cases are not low-signal noise — they have abundant, internally-consistent
online signal pointing the other way (cross-pressured / genuine mismatch). Emits only an
**aggregate, de-identified** table; no per-user export. (Built for `cand20`, the motivating example.)
