#!/usr/bin/env python3
"""
Script to create custom datasets for kvcached vLLM benchmarking.
Allows flexible configuration of prompt length, number of samples, and content.
"""

import argparse
import json
import random
import string
from pathlib import Path
from typing import Optional


def generate_random_text(length: int, seed: Optional[int] = None) -> str:
    """Generate random text of specified length."""
    if seed is not None:
        random.seed(seed)
    
    # Generate random text using letters and spaces
    chars = string.ascii_letters + " "
    text = ''.join(random.choice(chars) for _ in range(length))
    return text


def generate_completion(prompt: str, completion_length: int, seed: Optional[int] = None) -> str:
    """Generate a completion text of specified length."""
    if seed is not None:
        random.seed(seed + 1)  # Different seed for completion
    
    # Generate random completion text using letters and spaces
    chars = string.ascii_letters + " "
    completion = ''.join(random.choice(chars) for _ in range(completion_length))
    return completion


def create_dataset(
    output_path: str,
    num_samples: int,
    prompt_length: int,
    completion_length: int = 128,
    seed: Optional[int] = None
) -> None:
    """Create a custom dataset with specified parameters."""
    
    if seed is not None:
        random.seed(seed)
    
    dataset = []
    
    for i in range(num_samples):
        # Generate prompt with specified length
        prompt = generate_random_text(prompt_length, seed=seed + i if seed else None)
        
        # Generate completion
        completion = generate_completion(prompt, completion_length, seed=seed + i if seed else None)
        
        dataset.append({
            "prompt": prompt,
            "completion": completion
        })
    
    # Write to JSONL file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created dataset with {num_samples} samples")
    print(f"Prompt length: {prompt_length} characters")
    print(f"Completion length: {completion_length} characters")
    print(f"Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create custom datasets for kvcached vLLM benchmarking"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="custom_dataset.jsonl",
        help="Output file path (default: custom_dataset.jsonl)"
    )
    
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)"
    )
    
    parser.add_argument(
        "--prompt-length", "-p",
        type=int,
        required=True,
        help="Length of each prompt in characters (required)"
    )
    
    parser.add_argument(
        "--completion-length", "-c",
        type=int,
        default=128,
        help="Length of each completion in characters (default: 128)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    args = parser.parse_args()
    
    create_dataset(
        output_path=args.output,
        num_samples=args.num_samples,
        prompt_length=args.prompt_length,
        completion_length=args.completion_length,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
