"""Installed command-line entry point for SPInDa."""

import sys
from importlib import import_module


COMMANDS = {
    "train": "spinda.scripts.train",
    "predict": "spinda.scripts.predict",
    "evaluate": "spinda.scripts.evaluate",
    "download": "spinda.scripts.download_data",
        "analyze": "spinda.scripts.analyze",
}


def main() -> None:
    """Dispatch a SPInDa command without importing ML dependencies eagerly."""
    if len(sys.argv) < 2:
        print("Usage: spinda <command> [options]")
        print("\nCommands: train, predict, evaluate, download, analyze")
        print("Dataset preparation is documented in docs/reproducing_paper.md.")
        raise SystemExit(1)

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict, evaluate, download, analyze")
        raise SystemExit(1)

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    import_module(module_name).main()
