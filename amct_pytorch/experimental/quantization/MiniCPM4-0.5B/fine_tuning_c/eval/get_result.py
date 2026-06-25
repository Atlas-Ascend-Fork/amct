import json
from os import walk
from os.path import join, dirname, basename
from csv import DictReader
import numpy as np

path = "logs/MiniCPM4-qa-tune"

target_dataset_and_metrics = {
    "gsm8k": "accuracy",
    "openai_humaneval": "humaneval_pass@1",
    "math": "accuracy",
    "sanitized_mbpp": "score",
    "mmlu": "accuracy",
    "IFEval": "Inst-level-strict-accuracy",
    "lambada": "accuracy",
    "openbookqa": "accuracy",
}

for root, _, files in walk(path):
    for file in files:
        if file.endswith(".csv"):
            all_norm = []
            print(f"================{root}================")
            with open(join(root, file)) as f:
                csv_reader = DictReader(f)
                for row in csv_reader:
                    if row["dataset"] in target_dataset_and_metrics:
                        if row["metric"] == target_dataset_and_metrics[row["dataset"]]:
                            print("{}: {} = {}".format(row["dataset"], row["metric"], row["model-vllm"]))
                            all_norm.append(float(row["model-vllm"]))
            print("average: {:.4f}".format(np.asarray(all_norm).mean()))

for root, _, files in walk(path):
    for file in files:
        if "result" in file and file.endswith(".json"):
            print(f"================{basename(root)}================")
            avg_metrics = []
            with open(join(root, file)) as f:
                result = json.load(f)["results"]
                for dataset, logs in result.items():
                    if logs.get("alias", "alias").strip().startswith("-"):
                        continue
                    if "acc_norm,none" in logs:
                        metric = logs["acc_norm,none"]
                        metric_name = "acc_norm,none"
                    else:
                        metric = logs["acc,none"]
                        metric_name = "acc,none"
                    print("{}: {} = {:.2f}".format(dataset, metric_name, 100*metric))
                    avg_metrics.append(100*metric)
                print("[average]: {:.2f}".format(np.asarray(avg_metrics).mean()))