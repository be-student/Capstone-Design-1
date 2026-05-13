# Verifier D — Slices #5 (ML Churn) & #6 (DL Churn)

| Slice | Verdict |
| --- | --- |
| #5 ML Churn Model | **PASS** |
| #6 DL Churn Model | **PASS** |

## Slice #5 — ML Churn: PASS

- LightGBM + XGBoost both implemented with separate CV/final paths in `src/models/churn_model.py` (`MLChurnModel.LGBM_PARAM_GRID`/`XGB_PARAM_GRID`, `_cv_score_lightgbm`/`_cv_score_xgboost`, lines 224-372); `tests/test_churn_model.py:301` asserts `model_type in ("xgboost","lightgbm")` and `cv_results` carries both `lightgbm_best_cv_auc` and `xgboost_best_cv_auc` (lines 315-318).
- 5-fold `StratifiedKFold` is hard-wired (line 246, 291-293, 344-346); `test_ml_model_has_cv_support` asserts `n_folds == 5` (line 278). Class imbalance handled via `_compute_scale_pos_weight` and `_class_imbalance_params`; `test_imbalance_weighting_applied_in_cv_paths` (lines 437-442) verifies both weighting strategies are wired into CV paths.
- SHAP global + local both implemented in `src/models/shap_explainer.py`: `global_feature_importance`, `get_top_features(k=10)`, `export_top_features`, `explain_individual`, `export_local_explanations` with row-oriented schema. `tests/test_shap_explainer.py` covers global (4 tests), top-k, individual (4 tests), high-risk schema export, and 4 plot save tests. The single Group-2 skip (`test_works_with_lightgbm_or_xgboost`) is intentional and documented in `docs/requirement_traceability.md` line 62.

Caveat: `analyze_threshold` (churn_model.py:713-818) lacks a dedicated Group-2 unit test but is wired into `src/main.py:1119-1122` and exercised through pipeline tests.

## Slice #6 — DL Churn: PASS

- `LSTMChurnNetwork` (churn_model.py:825-878) and `TransformerChurnNetwork` (lines 931-1007, with `nn.Linear` input projection, sinusoidal `PositionalEncoding`, GELU FC head) both built via `DLTrainer._build_network` (dl_trainer.py:206-222). Tests `test_train_single_architecture_lstm` and `test_train_single_architecture_transformer` (test_dl_trainer.py:233, 249) train both end-to-end.
- `EarlyStopping` class (dl_trainer.py:44-132) with `patience`, `min_delta`, `mode`, `restore_best_weights`, deep-copied `best_state_dict`, wired into the training loop (lines 397-404, 461-468). Seven dedicated tests in `TestEarlyStopping` (lines 96-192) plus `test_early_stopping_during_training` (line 326) verify trigger, best-epoch, and weight restore.
- Padding via `create_sequences` left-pads with zeros (sequence_utils.py:69-77); `scale_sequences` preserves padding rows. `test_short_sequences_are_padded` (test_sequence_dataset.py:165) asserts presence of zero-padded rows. CPU-only enforced (`torch.device("cpu")`, churn_model.py:1042, dl_trainer.py:180) — verified by `test_dl_model_cpu_only`. Versioned `.pt` sibling + `model_artifacts_manifest.json` (churn_model.py:1280-1293) verified by `test_dl_model_save_writes_versioned_artifact` (line 535). `EnsembleChurnModel` weighted ML/DL average tested by 7 tests in `TestEnsembleChurnModel`.

## Cross-cut decision

`test_best_run_above_threshold` ("Best AUC nan below threshold 0.78") in Group 5 is **NOT** a slice #5/#6 violation. It is a slice #11 (dashboard) environmental issue: no MLflow runs exist on disk in this verification env, so `DataLoader.load_mlflow_runs()` returns NaN. MERGED_REPORT root-cause #2 already classifies this cluster as "Empty data fixtures in Dashboard tests." The model-quality contract (AUC ≥ 0.78 achievable) is verified by Group 2 tests (`test_ml_model_auc`, `test_ensemble_auc_meets_threshold`).
