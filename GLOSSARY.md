# ML course homework

Training and comparing image classifiers for the course homework, starting with the Class10 dog-breed task.

## Language

**Experiment**:
One training run of one model on a fixed training set with fixed settings, identified by its name and settings. It produces exactly one Checkpoint.
_Avoid_: run, cfg, trial

**Experiment key**:
Everything that changes an Experiment's trained weights. Two Experiments with the same key produce the same Checkpoint, so the second one is never trained.
_Avoid_: cfg, hash, cache key

**Checkpoint**:
The saved weights of an Experiment's best epoch, judged by validation accuracy.
_Avoid_: ckpt, state file

**Parent**:
The Experiment whose Checkpoint another Experiment starts training from.
_Avoid_: init, base model
