## Prepare of the Experiments

We wrote the two scripts : `preset/generate_eval_finesse_configs.py` and `preset/evaluate_finesse_automate.py`.
- `preset/generate_eval_finesse_configs.py` first generate the `srs.yaml` & `rss.yaml`.
- `preset/evaluate_finesse_automate.py` evaluate the model wrote on the `srs.yaml` & `rss.yaml`, half-automatically.

All task evaluated on Colab G4 instance. (NVIDIA RTX PRO 6000 Blackwell Server Edition). 

batch size unified size=32 for srs evaluation. rss evaluation is single-batch support only.

All task folllowed the methodology of the finesse evaluation thesis, which will be open-sourced and packaged later. We used psudo-import and used psudo-function like `from psudo_evaluator import run_benchmark_from_config` for the `preset/evaluate_finesse_automate.py`. You can run the actual code AFTER the package announcement.

We evaluated the models listed on the `model-selection\model-verify-lemb-log\investigation-verified.csv`. see the `model-selection\model-verify-lemb-log\log.md` and the `model-selection\readme.md` for detail.

## Result log of the Experiments

### 2026-03-29 : Experiment Finished

All model processed well on the experiments. Nothing special to record here. 

## Conclusion

-   **Final Model Count:** 34
-   Final FINESSE evaluation results are saved as,
    - `model_eval.zip`

- psudo-package `psudo_evaluator` will be announced later.


### 2026-04-04 : Additional Experiments Added

We evaluated additional method.

- `additional_model_eval/cultura-en-l16-srs-weak.zip` : We did SRS(L=16), which was SRS(L=8).
- `additional_model_eval/wikipedia-en-l15-rss.zip` : We changed dataset `CulturaX` to `Wikipedia` and evaluated only for RSS(L=15).
- `model_eval_baseline.zip` : additional baseline model, which is the "chunk-embed-then-average" context extending strategy applied for the `intfloat/multilingual-e5-base` model.

no model list changed, but evaluation method was varied for those experiements.

additional baseline model should NOT be treated as the single transformer model evaluation result.

### 2026-09-10 : Additional Experiments Added

We evaluated additional method.

- `additional_model_eval/cultura-en-l64-rss.zip` : We did RSS(L=4, 8, ..., 64), which was RSS(L=4, 5, ..., 16).

no model list changed, but evaluation method was varied for those experiements. also, the package `package/finesse-values` was created to proceed this experiment. Detailed changes will be introduced in `README.md`. 
