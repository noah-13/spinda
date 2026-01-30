#!/usr/bin/env python3
"""
Main entry point for NLI toolkits.

Provides unified CLI interface for training, prediction, and evaluation.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from nli_toolkits.scripts import evaluate, predict, train


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("\nCommands:")
        print("  train      - Train a model on SNLI")
        print("  predict    - Generate predictions with a trained model")
        print("  evaluate   - Evaluate predictions against ground truth")
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
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict, evaluate")
        sys.exit(1)


if __name__ == "__main__":
    main()
