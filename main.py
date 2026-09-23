#!/usr/bin/env python3
"""Command-line entry point for SPINDA.

SPINDA's commands intentionally stay small and composable: dataset-specific
preparation commands create the canonical JSON data, while training,
prediction, and evaluation can also be used independently.
"""

import sys
from pathlib import Path

# Add the repository root to the import path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("\nCommands:")
        print("  train      - Train a model")
        print("  predict    - Generate predictions with a trained model")
        print("  evaluate   - Evaluate predictions against ground truth")
        print("  download   - Download a supported raw dataset")
        print("\nDataset preparation commands are available as modules; see")
        print("docs/reproducing_paper.md for the supported dataset launchers.")
        print("\nFor help on a specific command, run:")
        print("  python main.py <command> --help")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Remove command from argv so subcommands can parse their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    commands = {
        "train": "spinda.scripts.train",
        "predict": "spinda.scripts.predict",
        "evaluate": "spinda.scripts.evaluate",
        "download": "spinda.scripts.download_data",
        "analyze": "spinda.scripts.analyze",
    }
    module_name = commands.get(command)
    if module_name is None:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict, evaluate, download, analyze")
        sys.exit(1)

    # Delay imports so ``python main.py`` remains informative even on machines
    # without a CUDA runtime or optional ML dependencies installed.
    module = __import__(module_name, fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
