"""Re-evaluate all 192 complete long-document prompts with the finished M64 bank."""
import importlib.util
from time import perf_counter

from run_m64 import ARM, GPU_FRACTION, ROOT, expand, load_final

REFERENCE = ROOT.parent / "dst_rank64_route_output_stageb_seed1234/evaluate_stageb.py"


def main():
    spec = importlib.util.spec_from_file_location("m64_cohort_replay", REFERENCE)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay.ROOT = ROOT
    replay.ARM = ARM
    original_report = replay.write_report
    memory = None

    def checkpoint(C, torch, engine):
        nonlocal memory
        from evaluate import B

        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
        initial, _ = B.R.load_memory(engine)
        memory = expand(initial, torch)
        engine.mode = "test"
        return load_final(C, torch, memory)

    def predict(longer, engine, row, task, gold):
        B, torch = longer.B, longer.torch
        B.SENDER_CHUNK = longer.PREFILL
        ids, source = B.capture(engine, row)
        assert ids == row["sender_ids"]
        align = B.fullsync(row["prompt"], engine.sender_tok, engine.tok, row["content_ids"])
        assert list(align.source_ids) == ids
        indices = torch.tensor(align.source_indices, device="cpu")
        body = longer.transferred(engine, len(row["content_ids"]), source, indices, None, longer.CHUNK)
        del source
        torch.cuda.synchronize()
        started = perf_counter()
        bank = B.construct(memory, body, 1)
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        result = longer.answer(engine, task, row, body, gold, memory, bank)
        return dict(result, memory_construction_seconds=elapsed)

    def report(core, split, pin, records, metrics, _baseline):
        return original_report(core, split, pin, records, metrics, "gated_dynamic")

    replay.checkpoint = checkpoint
    replay.long_prediction = predict
    replay.write_report = report
    replay.long()


if __name__ == "__main__":
    main()
