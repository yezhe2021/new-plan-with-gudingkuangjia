"""Load the exact same frozen translator as the matched Static Memory run."""

import experiment_common as run

# The verified Static engine reads the Rank64 checkpoint pin from the run manifest.
Engine = run.Z.scoped_module(
    "rank64_dynamic_frozen_engine",
    run.STATIC_ROOT / "engine_frozen.py",
    run,
).Engine
