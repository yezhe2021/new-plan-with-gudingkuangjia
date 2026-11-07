# Reused sources

- Original Full-Sync rank/mapping: https://github.com/yezhe2021/new-plan-with-gudingkuangjia/blob/main/experiments/fulltoken_sync_copydrop_v0_l3_to_q4_seed1234/fullsync_v0.py
- Original shared-unit builder: https://github.com/yezhe2021/new-plan-with-gudingkuangjia/blob/main/experiments/sync_boundary_nativekv_full28_diagonal_translator_llama3_2_3b_to_qwen3_4b_seed1234/units.py
- Original span definition: https://github.com/yezhe2021/new-plan-with-gudingkuangjia/blob/main/experiments/sync_boundary_nativekv_full28_diagonal_translator_llama3_2_3b_to_qwen3_4b_seed1234/offsets.py

Fetched through the connected GitHub plugin on2026-11-06. The original files are retained under `reference/` with normalized line endings. Their SHA256 identities are included in the new experiment manifest. The old pilot entrypoint is a reference artifact, not executed here; it imports old experiments that are not needed by DST. The production alignment wrapper uses only the original span/unit definitions and exactly the same ordered Copy/Drop rule, extending the selected set from Options to all RAW body tokens.

Data, prefix and scoring reference: `rnkv_keep_both_all36_arc_hotpot_gsm_seed1234`, manifest `ca045f531bf9596c883f1f22dd9f02232f1fbdaa5e0d553f9934a546561e262c`. Its historical Recolor+C predictions are reused with provenance for system comparison; its translator or Memory weights are not loaded by DST.
