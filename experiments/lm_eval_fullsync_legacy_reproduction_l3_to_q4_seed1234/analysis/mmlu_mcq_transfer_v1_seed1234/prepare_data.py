"""Download only official MMLU test/dev parquet bytes at a fixed revision."""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def main():
    folder = HERE / "datasets" / "mmlu"
    folder.mkdir(parents=True, exist_ok=True)
    sources = []
    for split in ("dev", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        url = (f"https://huggingface.co/datasets/{CONFIG['dataset_repo']}/resolve/"
               f"{CONFIG['dataset_revision']}/all/{filename}")
        path = folder / filename
        if not path.exists():
            print(f"Downloading {split} from pinned official revision", flush=True)
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            if not (content.startswith(b"PAR1") and content.endswith(b"PAR1")):
                raise RuntimeError(f"Not a parquet file: {split}")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        sources.append({"split": split, "file": filename, "url": url,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance = {"repo": CONFIG["dataset_repo"],
                  "revision": CONFIG["dataset_revision"], "files": sources,
                  "routing": "Filter the official all split by subject, preserving row order."}
    (folder / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
