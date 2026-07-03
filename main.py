#!/usr/bin/env python3
"""
Main entry point for HLV Toolkits.

Provides a unified CLI interface for training, prediction, evaluation, and preprocessing.
"""

import sys
from pathlib import Path

# Add the repository root to the import path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from hlv_toolkits.scripts import evaluate, predict, preprocess, train


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("\nCommands:")
        print("  train      - Train a model")
        print("  predict    - Generate predictions with a trained model")
        print("  evaluate   - Evaluate predictions against ground truth")
        print("  preprocess - Normalize raw datasets into canonical JSONL")
        print("\nFor help on a specific command, run:")
        print("  python main.py <command> --help")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Remove command from argv so subcommands can parse their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "train":
        train.main()
    elif command == "predict":
        predict.main()
    elif command == "evaluate":
        evaluate.main()
    elif command == "preprocess":
        preprocess.main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict, evaluate, preprocess")
        sys.exit(1)


if __name__ == "__main__":
    main()
