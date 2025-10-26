import argparse
import os
import subprocess
from pathlib import Path


# run python format_repo.py .
def format_repository(repo_path: str, line_length: int = 88):
    repo_path = os.path.abspath(repo_path)
    print(f"Formatting repository at: {repo_path}")

    python_files = []
    exclude_dirs = {"venv", ".env", "env", "__pycache__", ".git", ".ruff_cache"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    if not python_files:
        print("No Python files found in the repository.")
        return

    print(f"Found {len(python_files)} Python files to format.")

    print("\nChecking and formatting with Ruff...")
    ruff_config = [
        "ruff",
        "check",
        "--fix",
        "--select",
        "I",
        "--line-length",
        str(line_length),
        *[str(f) for f in python_files],
    ]
    subprocess.run(ruff_config, check=True)

    print("\nRunning Ruff formatting...")
    ruff_format_config = [
        "ruff",
        "format",
        "--line-length",
        str(line_length),
        *[str(f) for f in python_files],
    ]
    subprocess.run(ruff_format_config, check=True)

    print("\nFormatting complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format Python files in a repository using Ruff"
    )
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument(
        "--line-length", type=int, default=88, help="Maximum line length (default: 88)"
    )

    args = parser.parse_args()
    format_repository(args.repo_path, args.line_length)
