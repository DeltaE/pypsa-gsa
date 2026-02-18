"""Removes all template data from the data directory."""

from pathlib import Path

root = Path(__file__).parent.parent

data_dir = Path(root, "dashboard", "data", "state")

for f in data_dir.glob("**/*"):
    if f.suffix in [".csv", ".json"] and f.is_file():
        f.unlink()
