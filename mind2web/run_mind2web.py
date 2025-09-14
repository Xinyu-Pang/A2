"""Predict the next action based on the current state and memory."""
import os
from tqdm import tqdm
from memory import eval_sample
from utils.data import add_scores

import logging
logger = logging.getLogger("atm")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def run_prediction(args, examples, caller, start_idx, end_idx=None):
    """Run prediction for mind2web workflow.
    
    Args:
        args: Configuration arguments
        examples: List of examples to process
        caller: Caller object for evaluation
        start_idx: Starting index for processing
        end_idx: Ending index for processing (optional)
    """
    # Create workflow file if it doesn't exist
    if not os.path.exists(args.workflow_path): 
        open(args.workflow_path, 'w').close()
    
    # Add prediction scores and ranks to examples
    examples = add_scores(examples)
    
    # Process examples in the specified range
    if end_idx is None:
        end_idx = len(examples)
    
    for i in range(start_idx, end_idx):
        if args.mode == "memory":
            eval_sample(i, args, examples[i], caller, start_idx)
        elif args.mode == "action":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported workflow format: {args.mode}")
