"""CLI entry point for running experiments.

This module serves as the package entry point that delegates to the main
script in scripts/run_experiment.py.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Main entry point for running experiments."""
    # Import and run the main CLI from scripts
    from scripts.run_experiment import main as run_main
    run_main()


if __name__ == "__main__":
    main()
