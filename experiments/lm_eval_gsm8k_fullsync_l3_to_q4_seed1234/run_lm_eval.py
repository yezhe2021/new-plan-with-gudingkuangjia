"""Import the custom backend, then invoke the stock lm-eval CLI."""

import fullsync_lm  # noqa: F401
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
