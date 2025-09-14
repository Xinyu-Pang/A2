import os
import json
import pickle
from pathlib import Path
import argparse

def map_domains(args):
    """Find the corresponding domain and subdomain for the website."""
    maps = json.load(open(args.map_path, "r"))
    try:
        args.domain = maps[args.website]["domain"]
        args.subdomain = maps[args.website]["subdomain"]
        return args
    except KeyError:
        print("Website not found in the map!")
        return 


def set_paths(args):
    """Set paths."""
    flag_suffix = f"_{args.flag}" if args.flag is not None else ""
    # extracted test data (of the same website) path
    args.tmp_examples_path = os.path.join(args.upper_results_dir, "extracted_data", f"{args.benchmark}", "{}_tmp_examples.json".format(args.website))
    os.makedirs(os.path.dirname(args.tmp_examples_path), exist_ok=True)
    
    # workflow path
    args.workflow_path = Path(f"{args.abstraction_dir}/{args.benchmark}/{args.flag}/{args.website}{flag_suffix}.txt")
    args.semantic_workflow_path = Path(f"{args.abstraction_dir}/{args.benchmark}/{args.flag}/{args.website}{flag_suffix}_semantic.txt")
    os.makedirs(os.path.dirname(args.workflow_path), exist_ok=True)

    # existed abstraction to log history
    args.existed_workflow_path = Path(f"{args.abstraction_dir}/{args.benchmark}/log/{args.website}{flag_suffix}_all.json")
    os.makedirs(os.path.dirname(args.existed_workflow_path), exist_ok=True)

    # prediction results directory (for specific website)
    args.results_dir = Path(f"{args.upper_results_dir}/{args.benchmark}{flag_suffix}/{args.website}")
    os.makedirs(args.results_dir, exist_ok=True)

    # log path
    # args.log_path = Path(f"logs/{args.benchmark}/log_{args.flag}.txt")
    # os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    return args

def load_data(args):
    """Load and extract mind2web test data based on website."""
    
    if os.path.exists(args.tmp_examples_path):
        samples = json.load(open(args.tmp_examples_path, "r", encoding="utf-8"))
        print(f"Filtering down to #{len(samples)} examples on website [{args.website}]")
    else:
        samples = load_json(args.data_dir, args.benchmark)
        print(f"Loaded #{len(samples)} test examples")
        if args.website is not None:
            samples = [s for s in samples if s["website"] == args.website]
            print(f"Filtering down to #{len(samples)} examples on website [{args.website}]")
            # save extracted data
            os.makedirs(os.path.dirname(args.tmp_examples_path), exist_ok=True)
            json.dump(samples, open(args.tmp_examples_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    return samples

def load_existed_results(args, num_samples):
    """Load existed results and return pids to solve."""
    if args.load_results:
        solved_files = os.listdir(args.results_dir)
        solved_pids = [int(f.split("-")[0]) for f in solved_files if f.endswith(".json")]
    else:
        solved_pids = []

    pids = list(range(num_samples))
    pids = [p for p in pids if p not in solved_pids]
    return pids

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



# %% load data
def load_json(data_dir, folder_name):
    folder_path = os.path.join(data_dir, folder_name)
    print(f"Data path: {folder_path}")
    data_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".json")
    ]
    data_paths = sorted(data_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Construct trajectory dataset
    samples = []
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            samples.extend(json.load(f))
    print("# of samples:", len(samples))

    return samples


def add_scores(
    examples: list[dict], candidate_results: dict = None,
    score_path: str = "data/scores_all_data.pkl"
):
    """Add prediction scores and ranks to candidate elements."""
    if candidate_results is None:
        with open(score_path, "rb") as f:
            candidate_results = pickle.load(f)

    for sample in examples:
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_results["scores"][sample_id][candidate_id]
                    candidate["rank"] = candidate_results["ranks"][sample_id][candidate_id]
    
    return examples


# %% abstraction induction

def format_examples_abs(examples: list[dict], prefix: str = None, suffix: str = None) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(f"## Query {i+1}: {ex['confirmed_task']}")
        lines.append("    Actions:")
        lines.extend(ex["action_reprs"])
        lines.append("")
    prompt = '\n'.join(lines)
    if prefix is not None:
        prompt = prefix + '\n' + prompt
    if suffix is not None:
        prompt += '\n\n' + suffix
    return prompt 