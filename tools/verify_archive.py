"""Verify migration hashes, Git byte preservation, links and exclusions."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
FORBIDDEN_PARTS = {"cache", "checkpoints", "__pycache__"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".safetensors", ".bin", ".npy", ".npz", ".pyc"}


def main():
    checked = set()
    for entry in MANIFEST["entries"]:
        for item in entry["files"]:
            path = ROOT / item["path"]
            assert path.is_file(), item["path"]
            assert not (FORBIDDEN_PARTS & set(path.parts)), item["path"]
            assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, item["path"]
            content = path.read_bytes()
            assert len(content) == item["bytes"], item["path"]
            assert hashlib.sha256(content).hexdigest() == item["sha256"], item["path"]
            checked.add(item["path"])
    actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / "experiments").rglob("*")
              if path.is_file()}
    assert checked == actual, f"Manifest file set mismatch: {len(checked)} != {len(actual)}"

    document = (ROOT / "EXPERIMENT_RESULTS.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)]+)\)", document):
        assert (ROOT / link).is_file(), link

    staged = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    count = 0
    for record in staged.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        object_id = metadata.split()[1].decode("ascii")
        path = ROOT / raw_path.decode("utf-8")
        data = path.read_bytes()
        blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        assert hashlib.sha1(blob).hexdigest() == object_id, str(path)
        count += 1
    print(f"Verified {len(checked)} archived files, all links, and {count} staged Git blobs")


if __name__ == "__main__":
    main()
