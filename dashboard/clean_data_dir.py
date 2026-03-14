"""Removes all template data from the data directory."""

from pathlib import Path

root = Path(__file__).parent.parent

state_dir = Path(root, "dashboard", "data", "state")
system_dir = Path(root, "dashboard", "data", "system")

for data_dir in [state_dir, system_dir]:
    for f in data_dir.glob("**/*"):
        if f.suffix in [".csv", ".json", ".parquet"] and f.is_file():
            f.unlink()
