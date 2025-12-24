from utils import load_jsonl
import numpy as np
import os
import matplotlib.pyplot as plt
from clean_and_find_equ import extract_equations
from tqdm import tqdm
from collections import Counter
from tqdm.contrib.concurrent import process_map
from sklearn.linear_model import LinearRegression
import sys
import pandas as pd

# dataset_list = "gsm8k,math500,minerva_math,olympiadbench,college_math"
dataset_list = "gsm8k,math500,minerva_math,olympiadbench,aime24,amc23"
dataset_list = "gsm8k,math500,olympiadbench,college_math"
# dataset_list = "gsm8k,math500"

dataset_list = dataset_list.split(",")
model_name_list = [
    # ("Qwen/Qwen2.5-MATH-7B-Instruct", "qwen25-math-cot"),
    ("Qwen/Qwen2.5-MATH-7B", "qwen25-math-cot"),
    ("PRIME-RL/Eurus-2-7B-PRIME", "qwen25-math-cot"),
    ("hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo", "qwen25-math-cot"),
    ("QMATH-7B_gsmk_div0/checkpoint-240", "longer_prompt"),
    ("QMATH-7B_gsmk_div001/checkpoint-240", "longer_prompt"),
    # ("QMATH-7B_gsmk_div001_both/checkpoint-200", "longer_prompt"),
    ]

# model_name_list = []

model_name_list += [
#     # ("Qwen/Qwen2.5-MATH-7B-Instruct", "qwen25-math-cot"),
#     # ("QMATH-1.5B_gsmk_div001/checkpoint-200", "longer_prompt"),
#     # ("QMATH-1.5B_gsmk_div0/checkpoint-240", "longer_prompt"),
#     # ("QMATH-1.5B_gsmk_div001/checkpoint-280", "longer_prompt"),
#     # ("QMATH-1.5B_gsmk_div0005/checkpoint-309", "longer_prompt"),
#     # ("QMATH-1.5B_gsmk_div0002/checkpoint-309", "longer_prompt"),
    # ("Qwen/Qwen2.5-MATH-1.5B", "longer_prompt"),
    # ("QMATH-1.5B_gsmk_div0/checkpoint-309", "longer_prompt"),
    # ("QMATH-1.5B_gsmk_div001/checkpoint-309", "longer_prompt"),
    ]

recal = True
ignore_true = False
num_k = 8
start_idx = int(sys.argv[1])


def calc_diversity(n_samples):

    k_responses = n_samples["code"][start_idx:start_idx+num_k]
    equation_list, remain_code_list = extract_equations(k_responses)

    equation_list = [set(val) for val in equation_list]

    equation_total = []
    for equations in equation_list:
        equation_total.extend(list(equations))
    equation_total_count = Counter(equation_total)
    equation_total_count_N = len(equation_total_count.values())

    unique_num = [0 for _ in range(len(equation_list))]
    if equation_total_count_N == 0:
        return -1

    for i, equations in enumerate(equation_list):
        if len(equations) == 0:
            continue
        for equ in equations:
            unique_num[i] += int(equation_total_count[equ] == 1) 
        # unique_num[i] /= len(equations)
    
    return np.sum(unique_num) / sum(equation_total_count.values())


def calc_potiential(args):

    n_samples, greedy_sample = args
    k_scores = [int(val) for val in n_samples["score"][start_idx:start_idx+num_k]]
    greedy_score = [int(val) for val in greedy_sample["score"]][0]

    preds = n_samples['pred'][start_idx:start_idx+num_k]
    most_common_pred = Counter(preds).most_common(1)[0][0]
    most_common_pred_idx = preds.index(most_common_pred)

    print("K", len(preds))

    # return [sum(k_scores), int(sum(k_scores)>=4), sum(k_scores)/num_k, greedy_score==1]
    return [np.std(k_scores), k_scores[most_common_pred_idx], max(k_scores), sum(k_scores)/num_k, greedy_score==1]


def sub_tasks(args):

    greedy_sample, n_sample = args
    assert greedy_sample["idx"] == n_sample["idx"]
    assert greedy_sample["question"] == n_sample["question"]
    
    pot = calc_potiential((n_sample, greedy_sample))
    # checked
    if ignore_true:
        if pot[-1]:
            div = 0
        else:
            div = calc_diversity(n_sample)
    else:
        div = calc_diversity(n_sample)
    tmp_data = [div]
    tmp_data.extend(pot)
    
    return tmp_data


df_csvfile = "score_analysis.csv"
if not os.path.exists(df_csvfile) or recal:
    df_data = []

    for dataset in dataset_list:
        for (model_name, prompt_type) in model_name_list:
            pre_path = f"score_pass_k/{model_name}/{dataset}"
            pattern = f"{prompt_type}_-1_seed0_t0.5_s0_e-1.jsonl"
            n_samples_filepath = [val for val in os.listdir(pre_path) if val.endswith(pattern)]
            assert len(n_samples_filepath) == 1, f"{model_name}: {pre_path} {pattern}"
            n_samples_filepath = n_samples_filepath[0]
            n_samples_filepath = os.path.join(pre_path, n_samples_filepath)
            n_samples_list = sorted(list(load_jsonl(n_samples_filepath)), key=lambda x: x["idx"])

            pre_path = f"score_pass_1/{model_name}/{dataset}"
            if model_name != "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo":
                pattern = f"{prompt_type}_-1_seed0_t0.0_s0_e-1.jsonl"
            else:
                pattern = f"{prompt_type}_-1_seed0_t1.0_s0_e-1.jsonl"
            greedy_sample_filepath = [val for val in os.listdir(pre_path) if val.endswith(pattern)]
            assert len(greedy_sample_filepath) == 1, f"{model_name}: {greedy_sample_filepath}"
            greedy_sample_filepath = greedy_sample_filepath[0]
            greedy_sample_filepath = os.path.join(pre_path, greedy_sample_filepath)
            greedy_sample_list = sorted(list(load_jsonl(greedy_sample_filepath)), key=lambda x: x["idx"])  
            
            data_list = []
            for greedy_sample, n_sample in (zip(greedy_sample_list, n_samples_list)):
                tmp_data = sub_tasks((greedy_sample, n_sample))
                data_list.append(tmp_data)
            data_list = np.array(data_list)
            
            maj_k_acc = data_list[:, -4].mean()
            pass_k_acc = data_list[:, -3].mean()
            avg_k_acc = data_list[:, -2].mean()
            pass_1_acc = data_list[:, -1].mean()
            avg_k_acc_std = data_list[:, -1].mean() / np.sqrt(len(data_list) * num_k)

            div = data_list[data_list[:, 0]!=-1][:, 0].mean()

            tmp_data = [dataset, model_name, pass_1_acc, pass_k_acc, avg_k_acc, maj_k_acc, div, avg_k_acc_std]
            
            # data_list = data_list[data_list[:, -1]==0] # pass@1 fail
            # data_list = data_list[:, :-1]
            # data_list = data_list[data_list[:, 0]!=-1] # can not find any equations
            # div = data_list[:, 0].mean()

            # _, sum_k_score, max_k_score, mean_k_score =  list(np.mean(data_list, axis=0))
            
            df_data.append(tmp_data)

    df_data = pd.DataFrame(df_data, columns=["dataset", "model_name", "pass_1_acc", "pass_k_acc", "avg_k_acc", "maj_k_acc", "div", "avg_k_acc_std"])
    df_data.to_csv("score_analysis.csv", index=False)

else:
    df_data = pd.read_csv(df_csvfile)

# df_data.loc[df_data["dataset"].isin(["aime24"]), "pass_1_acc"] = df_data[df_data["dataset"].isin(["aime24"])]["avg_k_acc"]

df_data[["pass_1_acc", "pass_k_acc", "div", "avg_k_acc", "maj_k_acc", "avg_k_acc_std"]] *= 100
toshow_col = ["pass_1_acc", "pass_k_acc", "avg_k_acc", "div", "maj_k_acc"]
toshow_col = ["pass_1_acc", "avg_k_acc", "avg_k_acc_std"]
for col in toshow_col:
    pivot_table = df_data.pivot(index="model_name", columns="dataset", values=col)
    pivot_table = pivot_table.round(2)  # Round all data to 2 decimal places
    pivot_table["avg"] = pivot_table.mean(axis=1).round(2)  # Add an average column and round to 2 decimal places
    print("\n", "-"*20, f"{col}", "-"*20)
    pivot_table = pivot_table[dataset_list + ["avg"]]  # Reorder columns based on dataset_list and add avg
    print(pivot_table)
    # print(pivot_table.to_csv(sep="&", index=True))
