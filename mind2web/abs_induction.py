"""Induce Abstraction from Past Agent Experiences (for A2)."""

import os
import json
from typing import List, Dict, Any
from utils.data import load_json, format_examples_abs
from utils.env import is_io_dict


def get_trajectory(path: str) -> List[Dict[str, str]]:
    """
    Extract trajectory from result file.
    
    Args:
        path: Path to the result JSON file
        
    Returns:
        List of trajectory steps with environment and action information
    """
    trajectory = []
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for item in result:
        if not is_io_dict(item):
            continue
        step = {
            "env": "# " + item["input"][-1]["content"],
            "action": item["output"],
        }
        trajectory.append(step)
    return trajectory

def abstraction_induction(args, samples, caller, i):
    """Abstraction Induction."""
    abs_intro = "Here are the extracted abstractions that represent common sub-routines:\n{element_type} represents the corresponding type of the chosen element.\n"
    
    # load model predictions and format examples
    sys_prompt = open(args.abs_sys_prompt_path, 'r').read()    # semantic
    example_prompt = open(args.abs_exemplar_prompt_path, 'r').read() 
    episodic_sys_prompt = open(args.episodic_abs_sys_prompt_path, 'r').read()    # episodic
    episodic_example_prompt = open(args.episodic_abs_exemplar_prompt_path, 'r').read()

    if args.flag == "abs_task":
        # Task-based abstraction induction: q_i + a_{i-1} = a_i, a_i + q_{i+1} = a_i_hat
        
        # Step 1: Semantic memory induction
        semantic_path = args.semantic_workflow_path
        if os.path.exists(semantic_path):
            last_abs = open(semantic_path, 'r', encoding='utf-8').read()
            last_abs = last_abs.replace(abs_intro, "").strip("\n")
        else:
            last_abs = ""
            open(semantic_path, "w").close()

        last_task_path = os.path.join(args.results_dir, f"{i}-{args.model}.json")
        examples = [
            {
                "confirmed_task": samples[i]["confirmed_task"],
                "action_reprs": ["        "+step["action"] for step in get_trajectory(last_task_path)],
            }
        ]
        prompt = format_examples_abs(examples, None, None)

        split_prompt = example_prompt.split("<Split token>")
        if len(split_prompt) <= 2:
            raise ValueError("Invalid example prompt format: missing split tokens")
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"<Lastest solved task>\n{split_prompt[0]}"},
            {"role": "user", "content": f"<Existing abstractions>\n{split_prompt[1]}"},
            {"role": "assistant", "content": f"<Extracted abstractions>\n{split_prompt[2]}"},
            {"role": "user", "content": f"<Lastest solved task>\n{prompt}"},
            {"role": "user", "content": f"<Existing abstractions>\n{last_abs}"}
        ]
        
        semantic_abs = caller.call(messages=messages)
        semantic_abs = semantic_abs.replace("<Extracted abstractions>", "")
        semantic_abs = abs_intro + semantic_abs
        with open(semantic_path, 'w', encoding='utf-8') as fw:
            fw.write(semantic_abs)

        # (2) episodic abs
        if int(i+1) >= len(samples):
            return
        else:
            episodic_split_prompt = episodic_example_prompt.split("<Split token>")
            next_task = "## Query 2: "+samples[i+1]["confirmed_task"]
            messages = [
                {"role": "system", "content": episodic_sys_prompt},
                {"role": "user", "content": f"<Next task>\n{episodic_split_prompt[0]}"},
                {"role": "user", "content": f"<Existing abstractions>\n{episodic_split_prompt[1]}"},
                {"role": "assistant", "content": f"<Extracted abstractions>\n{episodic_split_prompt[2]}"},
                {"role": "user", "content": f"<Next task>\n{next_task}"},
                {"role": "user", "content": f"<Existing abstractions>\n{semantic_abs}"}
            ]
            # with open("prompt/demos/abs_task/abs_task_prompt_episodic.json", "w", encoding='utf-8') as fp:
            #     json.dump(messages, fp, indent=4, ensure_ascii=False)
    else:
        raise ValueError(f"Invalid flag: {args.flag}")

    # Generate abstraction response
    response = caller.call(messages=messages)
    response = response.replace("<Extracted abstractions>", "")
    response = abs_intro + response
    
    # Save abstraction to file
    with open(args.workflow_path, 'w', encoding='utf-8') as fw:
        fw.write(response)
