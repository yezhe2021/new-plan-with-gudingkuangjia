# Stage-A64 to512, then fresh Stage-B

Server B only; preserve original experiment and paused HeteroFold.
Reuse byte-identical prepared256-paper630-query data, fixed Dev8/25 and heldout148.
Copy Stage-A last64 including optimizer/scaler, best64, selection history and steps.
Do not copy completion markers or any Stage-B weights. Resume epoch3 through16.
Keep all losses, learning rates, batch sizes, Full-Sync mapping and generation unchanged.
Validate representation loss each epoch; generation diagnostics at128/256/512 steps.
Select lowest Dev representation loss over all epochs (could retain the earlier best);
record selected epoch/step explicitly. Freeze it and train a new zero-residual rank64
adapter for4 epochs/316 updates. Existing functional validation selection unchanged.
Final evaluation records all condition predictions and per-question raw official F1.

Startup: CPU tests, exact data/checkpoint SHA audit, optimizer/scaler checks,
previous unchanged full protocol-gate provenance, fresh GPU longest-paper forward/backward
for Stage-A and Stage-B functional adapter. No smoke optimizer update.
Verify resumed step65/66 is finite, then stop monitoring and provide pipeline.log command.
No Test tuning. Original data filtering remains512-4096 complete evidence tokens.
Budget: checkpoint copies/snapshots under15GiB; do not delete other experiment files.
