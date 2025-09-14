""""calculate score"""
import os
import json
import argparse
import matplotlib.pyplot as plt

def get_average(score_list: list[float], percentage: bool = False) -> float:
    """Get average score from a list of scores."""
    if score_list == []:
        return
    score = sum(score_list) / len(score_list)
    return score * 100 if percentage else score


def calc_score(results_dir):
    results = {}
    # load results
    file_paths = []
    for root, dirs, files in os.walk(results_dir):
        for name in files:
            file_path = os.path.join(root, name)
            file_paths.append(file_path)

    # load scores
    ele_acc, act_f1, step_sr, sr = [], [], [], []
    for fp in file_paths:
        res = json.load(open(fp, 'r'))[-1]
        ele_acc.append(get_average(res["element_acc"]))
        act_f1.append(get_average(res["action_f1"]))
        step_sr.append(get_average(res["step_success"]))
        sr.append(get_average(res["success"]))
    
    # print scores

    results["Element Acc"] = format(get_average(ele_acc, True), '.1f')
    results["Action F1"] = format(get_average(act_f1, True), '.1f')
    results["Step SR"] = format(get_average(step_sr, True), '.1f')
    results["SR"] = format(get_average(sr, True), '.1f')

    return results


def write_results(benchmark, flag, upper_results_dir):
    website_dist_path = os.path.join("data", f"{benchmark}_websites_dist.json")
    websites = json.load(open(website_dist_path, "r"))

    all_results = {}
    results_dir = os.path.join(upper_results_dir, f"{benchmark}{flag}")
    files = os.listdir(results_dir)
    assert len(files) == len(websites)

    all_results["all"] = calc_score(results_dir)
    print(all_results["all"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results-rebuttal")
    args = parser.parse_args()

    benchmark = "test_website"
    flag = "_awm-random-3"

    # calc_score(args.results_dir)
    write_results(benchmark, flag, args.results_dir)
