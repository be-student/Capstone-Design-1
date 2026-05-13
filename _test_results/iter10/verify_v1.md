# V1 — Verify Pages 00/01/02

Source PNGs (read with the Read tool as images, then cropped via PIL for
detailed legibility):

- `_test_results/dashboard_pages/00_overview.png`
- `_test_results/dashboard_pages/01_churn_analytics.png`
- `_test_results/dashboard_pages/02_model_performance.png`

Cross-reference baseline: `_test_results/iter9/iter9_audit_a1.md`.

---

## Issue-by-issue verdicts

| #  | Issue (from iter9_audit_a1)                                                                 | Verdict          | Evidence (number/quote from iter10 PNG) |
|----|---------------------------------------------------------------------------------------------|------------------|------|
| 1  | P02 headline P/R disagree with confusion-matrix P/R (ml 0.5331/0.7791 vs 0.7059/0.6000)    | **FIXED**        | Performance Comparison table (top of P02): `ml_model auc 0.8852 / precision 0.7059 / recall 0.6000 / f1 0.6486 / acc 0.7833`; `dl_model 0.8860 / 0.6471 / 0.5500 / 0.5946 / 0.7500`; `ensemble 0.8866 / 0.7647 / 0.6500 / 0.7027 / 0.8167`. Per-matrix captions match: `Acc: 78.33% / Prec: 70.59% / Rec: 60.00%`, `Acc: 75.00% / Prec: 64.71% / Rec: 55.00%`, `Acc: 81.67% / Prec: 76.47% / Rec: 65.00%`. Headline strip and matrices now agree exactly for all three models. |
| 2  | P02 confusion-matrix sample size (600) not disclosed; 3% of population, undisclosed split  | **FIXED**        | Caption directly under "Confusion Matrices": *"Test set size: 600 samples (3.0% of 20,000 customers). Headline Precision / Recall / F1 above are computed from these confusion matrices on the same test split."* |
| 3  | "Best Model: ensemble" asserted on Δ AUC = 0.0014 with no significance test, no CI         | **FIXED**        | "Best Model" header now carries a `(?)` help icon. Visible caption below KPI strip: *"AUC spread across the three models is 0.0014 (<0.005). No DeLong significance test was run; the 'Best Model' label is indicative only."* |
| 4  | P00 histogram leftmost bin (~4,000) ≠ P01 leftmost bin (~3,500) for the same 20k roster    | **NOT FIXED**    | P00 "Distribution of Churn Probabilities" first bin reads ≈ 4,000 (gridline at 4000 is touched). P01 "Distribution of Churn Risk Scores with Threshold Boundaries" first bin reads ≈ 3,000 (gridline at 3,000 with finer 0.025-wide bins). Same population, two different leftmost-bin counts; no footnote disclosing the binning difference. The gap in fact widened (≈4,000 vs ≈3,000). |
| 5  | P00 Total CLV ellipsis-truncated (`57,936,514,970 ...`) instead of compact `₩57.94B`       | **NOT FIXED**    | P00 KPI strip still renders Total CLV as `57,936,514,970 ...` (visible ellipsis, no `₩` prefix, no B/M/K compaction). F1 added `format_currency_krw()` and F5's log says Page 12's `Customers Retained` was switched, but the helper was not wired into the Page 00 Overview KPI strip. |
| 6  | Critical/High overlap implausibly high (3,596 / 5,717 = 62.9 % of High is Critical)         | **NOT FIXED**    | P01 Churn Risk Summary still shows `High Risk (>50%) 5,717` and `Critical (>75%) 3,596`. Risk Level Breakdown table: `low 11,574 / 57.9%`, `critical 3,596 / 18.0%`, `medium 2,709 / 13.5%` — same shape as iter9. Still 62.9% of the High bucket is Critical; no tapering remediation, no tooltip framing. |
| 7  | Synthetic-data banner persists on every page                                                 | NOT-IN-SCOPE-FOR-FIX-ITER | All three pages (P00/P01/P02) carry the banner *"Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."* Per task description, intentional disclosure for now. |

---

## Summary

- **FIXED:** 3 (issues 1, 2, 3 — all P02 critical defects)
- **PARTIAL:** 0
- **NOT FIXED:** 3 (issues 4, 5, 6)
- **REGRESSIONS:** 0
- **OUT-OF-SCOPE (banner, intentional):** 1 (issue 7)

The three P02-internal contradictions that were the iter9 "do-not-ship"
blockers are now resolved. The remaining three open items are cross-page
consistency (histogram), KPI formatting (Total CLV), and a
distribution-shape disclosure (Critical/High taper).

---

## Remaining work for next iteration

1. **P00/P01 histogram parity (issue 4).** Either share one binning helper
   (same bin edges, same count source) across `render_overview` and
   `render_churn_analytics`, or add a footnote on both panels disclosing
   the bin width and population so the leftmost-bin counts can be
   reconciled by the reader. Current iter10 difference (≈ 4,000 on P00
   vs ≈ 3,000 on P01) is *larger* than iter9's (≈ 4,000 vs ≈ 3,500), so
   the underlying binning split is still drifting.

2. **P00 Total CLV format (issue 5).** Apply `format_currency_krw()` (the
   B/M/K compactor that F1 already shipped) to the Total CLV KPI in the
   Page 00 Overview KPI strip. The helper exists; it is simply not yet
   called at the P00 site. Suggested change: in `app.py` `render_overview`,
   replace the raw `f"{total_clv:,.0f} KRW"` (or current equivalent) with
   `format_currency_krw(total_clv)` so the card reads `₩57.94B`.

3. **P01 Critical/High taper disclosure (issue 6).** Either (a) add a
   tooltip on the `Critical (>75%)` and `High Risk (>50%)` KPIs noting
   that 62.9% of the High bucket is Critical (i.e., the distribution is
   bimodal, not a healthy taper), or (b) split the High KPI into
   "High-only (50–75%): 2,121" and "Critical (>75%): 3,596" so the
   bimodality is visible without subtraction. This is a semantics fix,
   not a math fix — the underlying counts are internally consistent.

4. **Out of P00/01/02 scope but worth tracking:** the iter9 audit
   "Unreliable" / "Missing" lists for these three pages (no CI on
   Avg Churn Prob, no class balance, no model version stamp, no
   threshold tooltip on High Risk, no calibration curve on P02, no
   per-segment AUC, no McNemar / DeLong, identical 1.0 s training time
   floor) are unchanged in iter10. F5's log explicitly defers the
   confusion-matrix sample-size expansion (600 → 20,000) to the model
   layer; the matrix-vs-headline contradiction it set out to close is
   indeed closed (issue 1), but the underlying 3 % sample is still 3 %.
