"""CPU-only protocol audit before loading model weights."""

from transformers import AutoTokenizer

import evaluate_crossbench as E


def main():
    tok = {name: AutoTokenizer.from_pretrained(path, local_files_only=True)
           for name, path in E.CFG["models"].items()}
    for family in ("llama", "gemma"):
        for dataset in ("openbookqa", "arc_challenge", "mmlu_pro", "hellaswag"):
            rows, _ = E.crossbench_rows(dataset, {family: tok[family], "qwen": tok["qwen"]})
            assert len(rows) == 128, (family, dataset, len(rows))
            for row in rows[:2]:
                target, source, _ = E.fullsync_map(row, tok[family], tok["qwen"], family)
                assert len(target) == len(source) == len(row["encoded"]["qwen"]["body"]) - 1
                assert row["encoded"]["qwen"]["answer_only"]
            print(f"OK {family}/{dataset}: {len(rows)} fixed test rows", flush=True)


if __name__ == "__main__":
    main()
