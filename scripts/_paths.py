"""Shared locations. Set LRVF_DATA_DIR/LRVF_OUTPUT_DIR to override defaults."""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("LRVF_DATA_DIR", REPO_DIR / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("LRVF_OUTPUT_DIR", REPO_DIR / "outputs")).expanduser().resolve()
