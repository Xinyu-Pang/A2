"""Online Induction and Workflow Utilization Pipeline."""
import os
import json
import time
import copy
import argparse
import subprocess
from pathlib import Path
from utils.data import load_data, load_existed_results, set_paths, map_domains, str2bool
from utils.llm import Caller
from run_mind2web import run_prediction
from abs_induction import abstraction_induction
from utils.calc_score import calc_score
from concurrent.futures import ThreadPoolExecutor, as_completed


# python pipeline.py --benchmark test_website --flag abs_task --load_scores 1

class DotDict(dict):
    """
    A dictionary subclass that allows dot notation access to dictionary keys.
    
    This class extends the built-in dict class to provide convenient dot notation
    access to dictionary values, making the code more readable and similar to
    object attribute access.
    """
    
    def __getattr__(self, item):
        """Get dictionary value using dot notation."""
        return self.get(item)

    def __setattr__(self, key, value):
        """Set dictionary value using dot notation."""
        self[key] = value

    def __delattr__(self, item):
        """Delete dictionary item using dot notation."""
        del self[item]

def parse_args():
    """
    Parse command line arguments for the Mind2Web pipeline.
    """
    parser = argparse.ArgumentParser()

    # Data configuration
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing input data files")
    parser.add_argument("--memory_path", type=str, default="data/memory",
                       help="Path to memory storage directory")
    parser.add_argument("--benchmark", type=str, default="test_website",
                       choices=["test_task", "test_website", "test_domain", "train"],
                       help="Type of benchmark to run")
    parser.add_argument("--map_path", type=str, default="data/website_domain_pairs.json",
                       help="Path to website-domain mapping file")

    # Results and workflows configuration
    parser.add_argument("--upper_results_dir", type=str, default="results",
                       help="Directory for storing results")
    parser.add_argument("--abstraction_dir", type=str, default="abstraction",
                       help="Directory for storing generated abstractions")

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="LLM model to use")
    parser.add_argument("--temperature", type=str, default=0.0,
                       help="Temperature setting for model generation")

    # Environment context configuration
    parser.add_argument("--previous_top_k_elements", type=int, default=3,
                       help="Number of top elements to select for negative candidates")
    parser.add_argument("--top_k_elements", type=int, default=5,
                       help="Number of top elements to retain for positive candidates")
    parser.add_argument("--retrieve_top_k", type=int, default=0,
                       help="Number of few-shot examples to retrieve")

    # Ablation study parameters
    parser.add_argument("--mode", type=str, default="memory", 
                       choices=["memory", "action"],
                       help="Prediction mode for the system")
    parser.add_argument("--start_idx", type=int, default=0, 
                       help="Starting index for example selection")
    parser.add_argument("--end_idx", type=int, default=None, 
                       help="Ending index for example selection")

    # Prompt configuration
    parser.add_argument("--abs_sys_prompt_path", type=str, default="prompt/instruction_abs_basic.txt",
                       help="Path to system prompt for generating semantic abstractions")
    parser.add_argument("--abs_exemplar_prompt_path", type=str, default="prompt/one_shot_abs_basic.txt",
                       help="Path to exemplar prompt for semantic abstractions")
    parser.add_argument("--episodic_abs_sys_prompt_path", type=str, default="prompt/instruction_abs_episodic.txt",
                       help="Path to system prompt for generating episodic abstractions")
    parser.add_argument("--episodic_abs_exemplar_prompt_path", type=str, default="prompt/one_shot_abs_episodic.txt",
                       help="Path to exemplar prompt for episodic abstractions")
    parser.add_argument("--prefix_abs", type=str, default="<Input>",
                       help="Prefix for abstraction prompts")
    parser.add_argument("--suffix_abs", type=str, default="<Extracted Abstractions>",
                       help="Suffix for abstraction prompts")
    parser.add_argument("--prefix", type=str, default=None,
                       help="General prefix for prompts")
    parser.add_argument("--suffix", type=str, default="# Summary Workflows",
                       help="General suffix for prompts")

    # Induction frequency configuration
    parser.add_argument("--induce_steps", type=int, default=1,
                       help="Number of steps between workflow inductions")
    parser.add_argument("--if_induce", type=str2bool, default=True,
                       help="Whether to perform induction")
    parser.add_argument("--if_workflow", type=str2bool, default=True,
                       help="Whether to generate abstractions")

    # Setup configuration
    parser.add_argument("--setup", type=str, default="online",
                       help="Setup mode for the pipeline")
    parser.add_argument("--load_results", type=str2bool, default=False,
                       help="Whether to load previous results (may include partially inferred websites)")
    parser.add_argument("--load_scores", type=str2bool, default=False,
                       help="Whether to load previously computed website scores")
    parser.add_argument("--flag", type=str, default="abs_task",
                       help="Induction strategy flag")

    # Analysis configuration
    parser.add_argument("--analysis_task", type=str, default=None,
                       help="Task analysis configuration for experimental studies")

    args = parser.parse_args()
    return args

def online(args):
    """
    Execute online learning pipeline with workflow induction and prediction.
    """
    print("===== Running for website: {} =====".format(args.website))
    
    # Step 1: Initialize LLM caller
    caller = Caller(args.model)

    # Step 2: Load and process test data
    samples = load_data(args)  # Load original test data for the website
    
    # Step 3: Setup processing parameters
    num_samples = len(samples)
    pids = list(range(len(samples)))  # Process all samples
    
    # Initialize log file if not loading previous scores
    if not args.load_scores:
        with open(args.log_path, "w", encoding="utf-8") as fl:
            fl.write("")

    # Clear previous workflows if not loading results and workflow generation is enabled
    if not args.load_results and args.if_workflow:
        with open(args.workflow_path, 'w', encoding='utf-8') as fw:
            fw.write("")

    # Step 4: Run inference and learning loop
    if len(pids) == 0:
        raise ValueError("No samples to process")
        return None
    
    for i in pids:
        j = min(num_samples, i + args.induce_steps)
        print(f"Running inference on {i}-{j} th example..")

        # Step 4.1: Run prediction on current batch
        if args.if_induce:
            run_prediction(args, samples, caller, i, j)
            print(f"    Finished prediction on {i}-{j} th example!")

        # Step 4.2: Generate new workflows based on current results
        if j < num_samples and args.if_workflow:
            if "abs" in args.flag:
                # Abstraction-based induction
                abstraction_induction(args, samples, caller, i)
                print(f"    Finished abstraction induction with 0-{i} th examples!\n")
            else:
                raise ValueError(f"Invalid flag: {args.flag}")
    
    # Step 5: Calculate final metrics
    print(f"Results for {args.benchmark} / {args.website}:")
    results = calc_score(args.results_dir)

    # Compute and log API costs and token usage
    my_cost = caller.compute_cost()
    tokens = {
        "total": caller.total_tokens, 
        "input": caller.prompt_tokens, 
        "output": caller.completion_tokens
    }
    print("Total cost for {} / {}: {}".format(args.benchmark, args.website, my_cost))
    
    # Write results to log file
    with open(args.log_path, "a") as fl:
        fl.write(f"Results for {args.benchmark} / {args.website}:\n")
        for k, v in results.items():
            fl.write(f"    {k}: {v}\n")
        fl.write(f"    total cost: {my_cost}\n\n")
    
    return results, float(my_cost), tokens

if __name__ == "__main__":
    """
    Main execution block for the Mind2Web pipeline.
    """

    args = parse_args()

    # Initialize results and token tracking paths
    args.all_results_path = Path(f"{args.upper_results_dir}/results_{args.benchmark}_{args.flag}.json")
    if os.path.exists(args.all_results_path) and args.load_scores:
        # Load existing results if available and loading is enabled
        all_results = json.load(open(args.all_results_path, "r"))
    else:
        all_results = {}
    
    # Setup logging directory and file
    args.log_path = Path(f"logs/{args.benchmark}/log_{args.flag}.txt")
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    if not args.load_scores:
        with open(args.log_path, "a") as fl:
            fl.write("")

    # Load and filter websites for processing
    websites = json.load(open(f"data/{args.benchmark}_websites_dist.json", "r")).keys()
    if args.load_scores:
        # Filter out websites that have already been processed
        existed_websites = all_results.keys()
        websites = list(set(websites) - set(existed_websites))

    if args.setup == "online":
        # Validate required directories for online mode
        assert (args.upper_results_dir is not None) and (args.abstraction_dir is not None)

        # Execute online learning pipeline with parallel processing
        # # Adjust max_workers as needed
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {}
            
            # Submit tasks for each website
            for website in websites:
                # Create a copy of args for each website to avoid conflicts
                args_copy = DotDict(vars(args).copy())
                args_copy.website = website
                args_copy = map_domains(args_copy)  # Map website to domain information
                args_copy = set_paths(args_copy)    # Set website-specific paths

                # Submit online learning task for this website
                future = executor.submit(online, args_copy)
                futures[future] = website

            # Collect results from completed tasks
            for future in as_completed(futures):
                website = futures[future]
                try:
                    results, cost, tokens = future.result()
                    if results:
                        # Store results for this website
                        all_results[website] = results
                        json.dump(all_results, open(args.all_results_path, "w", encoding="utf-8"), indent=4)
                except Exception as e:
                    print(f"Error: {e}")
                
                # Brief pause between website processing
                time.sleep(1)
        
        # Calculate overall results for the benchmark
        overall_results = calc_score(Path(f"{args.upper_results_dir}/{args.benchmark}_{args.flag}"))
        with open(args.log_path, "a") as fl:
            fl.write(f"Overall Results for {args.benchmark}:\n")
            for k, v in overall_results.items():
                fl.write(f"    {k}: {v}\n")
