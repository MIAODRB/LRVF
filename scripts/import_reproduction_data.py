#!/usr/bin/env python3
"""Install the checksum-verified companion data bundle into the local data folder."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Companion ZIP, reproduction_data folder, or its data subfolder")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    source, target = args.bundle.expanduser().resolve(), args.data_dir.expanduser().resolve()
    entries = json.loads((ROOT / "docs/data_manifest.json").read_text(encoding="utf-8"))["files"]
    archive = zipfile.ZipFile(source) if source.is_file() else None
    base = source / "data" if (source / "data").is_dir() else source
    def open_input(item):
        rel = item["path"]
        return archive.open("data/" + rel) if archive else (base / rel).open("rb")
    try:
        # Check the complete bundle and existing destinations before any writes.
        for item in entries:
            with open_input(item) as stream:
                if digest(stream) != item["sha256"]:
                    raise ValueError("Companion data checksum mismatch: " + item["path"])
            dest = (target / item["path"]).resolve()
            if target not in dest.parents:
                raise ValueError("Unsafe destination in manifest")
            if dest.is_file():
                with dest.open("rb") as stream:
                    if digest(stream) != item["sha256"]:
                        raise ValueError("Existing data differs; preserve or move it before import: " + str(dest))
        for item in entries:
            dest = target / item["path"]
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".importing")
            try:
                with open_input(item) as stream, temp.open("wb") as out:
                    shutil.copyfileobj(stream, out, 1024 * 1024)
                temp.replace(dest)
            finally:
                temp.unlink(missing_ok=True)
        print("Imported " + str(len(entries)) + " verified inputs into " + str(target))
    finally:
        if archive:
            archive.close()
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
