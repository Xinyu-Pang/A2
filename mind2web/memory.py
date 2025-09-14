import os, json, random
import numpy as np
from pathlib import Path
from openai import BadRequestError
from utils.env import *
import utils

import logging
logger = logging.getLogger(__name__)


def get_exemplars(args, task, caller) -> list:
    """Get exemplar workflows in the prompt."""
    # (a) load workflow memory
    memory = []
    workflow_text = open(args.workflow_path, 'r', encoding='utf-8').read().strip()
    if len(workflow_text):
        if "abs" in args.flag:
            memory = [[
                {"role": "user", "content": f"Existed Workflow: {workflow_text}"}
            ]]
        else:
            raise ValueError("invalid flag!")
    # (b) load concrete examples
    with open(os.path.join(args.memory_path, "exemplars.json"), "r", encoding="utf-8") as f:
        concrete_examples = json.load(f)
    if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples 
            if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain, args.website]])
        ]
    elif any([args.subdomain in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples 
            if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain]])
        ]

    memory += random.sample(concrete_examples, 
        min(args.retrieve_top_k, len(concrete_examples)))

    memory = [[{k:v for k,v in m.items() if k!="specifier"} for m in e] for e in memory]
    return memory

def eval_sample(task_id, args, sample, caller, start_idx):
    """Predict for the sample. (Initial version)"""    
    # (a) initialize metrics
    element_acc, action_f1, step_success, success = [], [], [], []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    exemplars = get_exemplars(args, sample["confirmed_task"], caller)
    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]

    prev_actions, prev_obs = [], []
    # (b) prediction
    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)
        
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]

        # get query, obs, act
        target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)
        
        # Continue next loop if the ground truth element is not in the cleaned html
        if len(pos_candidates) == 0:
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
            conversation.append("The ground truth element is not in cleaned html")
            continue

        # construct query
        query = []
        for o, a in zip(prev_obs, prev_actions):
            if len(query) == 0:
                query.append({
                    "role": "user",
                    "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n" + o,
                })
            else:
                query.append({"role": "user", "content": o})
            query.append({"role": "assistant", "content": a})
        
        obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
        
        if len(query) == 0:
            query.append({
                "role": "user",
                "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                + "Observation: `" + obs + "`",
            })
        else:
            query.append({"role": "user", "content": "Observation: `" + obs + "`"})
        
        prev_obs.append("Observation: `" + target_obs + "`")
        prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
        
        # token limit
        total_num_tokens = caller.num_tokens_from_messages(sys_message + query, args.model)
        if total_num_tokens > caller.MAX_TOKENS[args.model]:
            logger.info(
                f"Too many tokens in acting ({total_num_tokens} / {caller.MAX_TOKENS[args.model]}), skipping..."
            )
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            conversation.append(
                {
                    "input": sys_message + query,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            continue

        # message token control
        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = caller.num_tokens_from_messages(
                sys_message + demo_message + e + query, args.model
            )
            if total_num_tokens > caller.MAX_TOKENS[args.model]:
                logger.info(
                    f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                )
                break
            else:
                demo_message.extend(e)
        message = sys_message + demo_message + query

        try:
            response = caller.call(
                messages=message,
                stop_tokens=["Task:", "obs:"]
            )
        except BadRequestError:
            response = ""
            info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        # conversation.append({"input": message, "output": response, "token_stats": info})
        conversation.append({"input": message, "output": response})
        # for k, v in info.items():
        #     token_stats[k] += v

        pred_act = caller.extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    # save results
    with open(os.path.join(args.results_dir, f"{task_id}-{args.model}.json"), "w") as f:
        json.dump(conversation, f, indent=4)
