import shutil
from pathlib import Path
import os

UNZIP_CHART = {
    "rss-64" : "benchmarks/finesse/additional_model_eval/cultura-en-l64-rss.zip",
    "srs-16" : "benchmarks/finesse/additional_model_eval/cultura-en-l16-srs-weak.zip",
    "rss-wikipedia" : "benchmarks/finesse/additional_model_eval/wikipedia-en-l15-rss.zip",
    "finesse-baseline" : "benchmarks/finesse/model_eval_baseline.zip",
    "finesse-main" : "benchmarks/finesse/model_eval.zip",
    "finesse-pt-0" : "benchmarks/finesse/model_eval_pt/model_eval_pt_0.zip",
    "finesse-pt-1" : "benchmarks/finesse/model_eval_pt/model_eval_pt_1.zip",
    "finesse-pt-2" : "benchmarks/finesse/model_eval_pt/model_eval_pt_2.zip",
    "finesse-pt-3" : "benchmarks/finesse/model_eval_pt/model_eval_pt_3.zip",
    "lemb-main" : "benchmarks/lemb/model_eval_2.zip",
    "lemb-baseline": "benchmarks/lemb/model_eval_baseline.zip"
}

MV_CHART = {
    "evals/special/rss-64" : "benchmarks/finesse/additional_model_eval/cultura-en-l64-rss",
    "evals/special/srs-16" : "benchmarks/finesse/additional_model_eval/cultura-en-l16-srs-weak",
    "evals/special/rss-wikipedia" : "benchmarks/finesse/additional_model_eval/wikipedia-en-l15-rss",
    "evals/typical/finesse/model_eval/finesse-baseline" : "benchmarks/finesse/model_eval_baseline",
    "evals/typical/finesse/model_eval/finesse-main" : "benchmarks/finesse/model_eval",
    "evals/typical/finesse/model_eval_pt/finesse-pt-0" : "benchmarks/finesse/model_eval_pt/model_eval_pt_0",
    "evals/typical/finesse/model_eval_pt/finesse-pt-1" : "benchmarks/finesse/model_eval_pt/model_eval_pt_1",
    "evals/typical/finesse/model_eval_pt/finesse-pt-2" : "benchmarks/finesse/model_eval_pt/model_eval_pt_2",
    "evals/typical/finesse/model_eval_pt/finesse-pt-3" : "benchmarks/finesse/model_eval_pt/model_eval_pt_3",
    "evals/typical/lemb/model_eval/lemb-main" : "benchmarks/lemb/model_eval_2",
    "evals/typical/lemb/model_eval/lemb-baseline": "benchmarks/lemb/model_eval_baseline"
}

def unzipper():
    for i, (instruct, path) in enumerate(UNZIP_CHART.items()):
        print(f"Unzipping {instruct} : {i + 1} / {len(UNZIP_CHART)} ...")
        path_unzipped = str(path)[:-4]
        path = Path(path)
        mv_chart_swapped = {v: k for k, v in MV_CHART.items()}
        mv_dir = mv_chart_swapped.get(str(path_unzipped))
        if not (os.path.exists(path_unzipped) or os.path.exists(mv_dir)):
            shutil.unpack_archive(path, path.parent)
            print(f"Unzip Succeed! {instruct} : {i + 1} / {len(UNZIP_CHART)} ...")
        else:
            print(f"Unzip Skipped! {instruct} : {i + 1} / {len(UNZIP_CHART)} ...")

def mv_operator():
    for i, (to, fr) in enumerate(MV_CHART.items()):
        print(f"Moving {fr} -> {to} : {i + 1} / {len(MV_CHART)} ...")
        fr_path = Path(fr)
        to_path = Path(to)
        if not os.path.exists(to_path):
            shutil.move(fr_path, to_path)
            print(f"Mov Succeed! {fr} -> {to} : {i + 1} / {len(MV_CHART)} ...")
        else:
            print(f"Mov Skipped! {fr} -> {to} : {i + 1} / {len(MV_CHART)} ...")


if __name__ == "__main__":
    unzipper()
    mv_operator()