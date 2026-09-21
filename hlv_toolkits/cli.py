"""Installed command-line entry point for SPInDa."""

import sys
from importlib import import_module


COMMANDS = {
    "train": "hlv_toolkits.scripts.train",
    "predict": "hlv_toolkits.scripts.predict",
    "evaluate": "hlv_toolkits.scripts.evaluate",
    "download": "hlv_toolkits.scripts.download_data",
        "analyze": "hlv_toolkits.scripts.analyze",
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
