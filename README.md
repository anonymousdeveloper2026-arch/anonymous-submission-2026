## Installation Guide

This guide provides instructions for setting up the environment.

- First, please install torch==2.10.0. Ensuring your GPU environment is configured beforehand will greatly assist in smooth replication.
```bash
# OSX
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# ROCM 7.1 (Linux only)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1
# CUDA 12.6
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# CUDA 13.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
# CPU only
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```
- Next, please execute the following command: `pip install -e package/finesse`

- Afterward, you may simply run the command: `pip install -r requirements.txt`


## Reproduction Guide

### For Quick Verification of Tables and Statistics

1. First, the original (raw) evaluation result files must be unzipped and gathered in one place. To respect the valuable time of our reviewers, we have automated this process.
    - `python source/unzip.py`
    Executing the command above will automatically unzip all compressed files.
2. Subsequently, the tables and figures from the paper are reconstructed on the fly based on the original evaluation results.
    - `python source/reproduce.py`
3. The tables and figures will be generated in the `source/compiled` directory. Each file corresponds to the following items in the paper:
    - `source/compiled/tbl_rss/table.tex` => Table 1
    - `source/compiled/tbl_lemb/table.tex` => Table 2
    - `source/compiled/tbl_rsslemb_correl/table.tex` => Table 9
    - `source/compiled/tbl_srslemb_correl/table.tex` => Table 3
    - `source/compiled/tbl_srslemb_correl/pearson_tbl/table.tex` => Table 4
    - `source/compiled/tbl_srslemb_correl/spearman_tbl/table.tex` => Table 5
    - `source/compiled/tbl_twcp/table.tex` => Table 6, 7, 8
    - `source/compiled/tbl_tdbu/table.tex` => Table 10
    - `source/compiled/tbl_lemb_iqr/table.tex` => Table 11
    - `source/compiled/tbl_nomic/table.tex` => Table 12
    - `source/compiled/tbl_f2llm/table.tex` => Table 13
    - `source/compiled/tbl_rssfull/table.tex` => Table 14
    - `source/compiled/tbl_lembfull/table.tex` => Table 15
    - `source/compiled/tbl_ttest/table.tex` => Table 16
    - Figure 2, 3, 12-20 => `source/compiled/fig_srs_rss/srs`
    - Figure 21 => `source/compiled/fig_srs_rss/rss`
    - Figure 1 => `source/compiled/fig_lx_master`
    - Figure 4 => `source/compiled/fig_twcp_nemotron`, `source/compiled/fig_twcp_qwen`
    - Figure 5 => `source/compiled/fig_crc_heatmap`
    - Figure 6 => `source/compiled/fig_srs_16`
    - Figure 7 => `source/compiled/fig_cross_corpus`
    - Figure 8 => `source/compiled/fig_l15_lembtask`
    - Figure 9 => `source/compiled/fig_lemb_statistic`
    - Figure 10 => `source/compiled/fig_rss_jackknife`
    - Figure 11 => `source/compiled/fig_srs_jackknife`

### How to Add Your Own Data Points

Scripts such as `benchmarks/finesse/preset/evaluate_finesse_automate.py` within the `benchmarks` folder are archives of the scripts used by the author in a Google Colab environment. Direct execution may be difficult in your local GPU environment due to differences in paths, etc. Therefore, we have prepared two separate scripts that allow for identical reproduction with minor modifications.

- `python source/evaluate_finesse_local.py`
- `python source/evaluate_lemb_local.py`

Please open each script and edit the `TARGET_FOLDER` global variable. By default, they target `examples/evals/finesse` and `examples/evals/lemb`, and we have included an example of evaluating the `Alibaba-NLP_gte-modernbert-base` model in those folders for your reference.

Once the evaluation is complete, unify the folder names of the LEMB and FINESSE evaluation results to match the example format (e.g., `Alibaba-NLP_gte-modernbert-base`) and place them in the following paths respectively:

- `evals/typical/finesse/model_eval/finesse-main`
- `evals/typical/lemb/model_eval/lemb-main/included`

Afterward, running `python source/reproduce.py` will allow you to see the results with the new data points included. This completes the process.

## Validating the Research Process

### Reviewing Experiment Logs

Even without running the scripts yourself, you can verify the transparency of our research process through our detailed logs. First, please refer to:

- `model-selection/readme.md`

...to understand the model selection process. Following that, you can review the experiment logs for each evaluation at:

- `benchmarks/finesse/log.md`
- `benchmarks/lemb/log.md`
- `benchmarks/sffd/log.md`

### Supplementary Materials (Not in Main Paper)

For reviewers who wish to conduct a deeper review or for future research, we have left the following additional materials, which are not included in the main body of the paper.

- `benchmarks/environment/report.txt`: A report verifying that the corpora used in this study (`CulturaX`, `Wikipedia`) contain a balanced distribution of semantically diverse text.
- `benchmarks/sffd`: Materials related to a new evaluation paradigm called SFfD (Summary Finding from Documents). While not directly within the scope of this study's application, it was explored during the research process. Should you be interested, we invite you to review these materials.


## Notice

### Minor Bug Notation

We wish to inform you of a minor bug in the `package/finesse` package used for evaluation in this study.

Please check the following code within `package/finesse/src/finesse/cli.py`:

```python
# ... (code snippet above) ...
        length_scores[target_length] = {
            'rss_scores': scaled_rss,
            'total_latency_scores': scaled_total,  # ms, cold start
            'synthesis_latency_scores': scaled_synth,  # ms, warm start
            'raw_td': td_scores,
            'raw_bu': bu_scores
        }
# ... (rest of the function) ...
```

The code above should be corrected as follows:

```python
        length_scores[target_length] = {
            'rss_scores': scaled_rss,
            'total_latency_scores': scaled_total,  # ms, cold start
            'synthesis_latency_scores': scaled_synth,  # ms, warm start
            'raw_td': sample_tds,
            'raw_bu': sample_bus
        }
```

However, this bug does not affect the final RSS calculation. Specifically, the evaluation proceeded with only the `raw_td` and `raw_bu` items being output in an incorrect format in the final results:

```json
...
  "length_scores": {
    "4": {
      "rss_scores": [...],
      "total_latency_scores": [...],
      "synthesis_latency_scores": [...],
      "raw_td": {
        "contextual_coherence": wrong_value
      },
      "raw_bu": {
        "bottom_up_coherence": wrong_value
      }
    }
  },
```

Therefore, in `source/reproduce.py`, the TD and BU scores were calculated by directly opening the index file (the `.pt` file generated by `finesse generate --config`). For a detailed implementation, please refer to `source/scripts/tbl10_td_bu_rss_correl.py`.

### Version Differ Notation

`package/finesse-values` package is almost identical from the original `package/finesse`. However, this package is specially further developed to iterate through specific length value which was used for experiment `additional_model_eval/cultura-en-l64-rss.zip`.

Major difference is just the iteration method, which was originally
```python
  length_results = {}  
  for target_length in tqdm(range(min_length, max_length + 1), desc="Native RSS: Probing Lengths"):
```
to
```python
  length_results = {}
  target_lengths = self.config.probe_config.sequence_length.get_lengths()
  for target_length in tqdm(target_lengths, desc="Native RSS: Probing Lengths"):
```

You can compare the modified `package/finesse-values/src/finesse/config.py` and `package/finesse-values/src/finesse/evaluator.py` with the original `package/finesse-values/src/...` source code.

## Conclusion

We sincerely thank you for dedicating your valuable time to our work. We hope this material has been helpful in verifying the reproducibility and transparency of our research.