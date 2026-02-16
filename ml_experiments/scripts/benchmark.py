"""CLI entry point for benchmarking models.

This module serves as the package entry point that delegates to the main
script in scripts/benchmark.py.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Main entry point for model benchmarking."""
    # Import and run the main CLI from scripts
    from scripts.benchmark import main as bench_main
    bench_main()


if __name__ == "__main__":
    main()
