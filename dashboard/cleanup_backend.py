"""Cleanup script to remove cached backend files."""

import os
import shutil


def cleanup(cache_dir: str) -> None:
    """Remove all files and subdirectories inside the cache directory."""
    if not os.path.isdir(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    entries = os.listdir(cache_dir)
    if not entries:
        print("Cache directory is already empty.")
        return

    for entry in entries:
        entry_path = os.path.join(cache_dir, entry)
        if entry == ".gitkeep":
            continue
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)

    print("Cache cleanup complete.")


if __name__ == "__main__":
    cache_dir = os.path.join(os.path.dirname(__file__), "file_system_backend")
    cleanup(cache_dir)
