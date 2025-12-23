#!/usr/bin/env python3
"""
Utility functions for vLLM inference scripts.
"""

import json
from pathlib import Path

def load_prompts():
    """Load prompts from data/prompts.txt"""
    prompts_file = Path(__file__).parent.parent / "data" / "prompts.txt"
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def save_results(results):
    """Save results to results/inference_outputs.jsonl"""
    results_file = Path(__file__).parent.parent / "results" / "inference_outputs.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')