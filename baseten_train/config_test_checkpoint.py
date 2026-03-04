"""Quick test: verify BT_CHECKPOINT_DIR is set and writable."""

from truss_train import TrainingProject, TrainingJob, Image, Compute, Runtime, definitions
from truss.base.truss_config import AcceleratorSpec, Accelerator

training_job = TrainingJob(
    image=Image(
        base_image="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
    ),
    compute=Compute(
        # 1 GPU is enough for a quick test
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=1),
    ),
    runtime=Runtime(
        start_commands=[
            "echo '=== ENV CHECK ===' && "
            "echo \"BT_CHECKPOINT_DIR=$BT_CHECKPOINT_DIR\" && "
            "echo \"Contents:\" && ls -la \"$BT_CHECKPOINT_DIR\" 2>/dev/null || echo '(dir does not exist)' && "
            "echo '=== WRITE TEST ===' && "
            "mkdir -p \"$BT_CHECKPOINT_DIR/test\" && "
            "echo 'hello' > \"$BT_CHECKPOINT_DIR/test/probe.txt\" && "
            "echo \"Write OK. Sleeping 5min so you can inspect logs...\" && "
            "sleep 300"
        ],
        checkpointing_config=definitions.CheckpointingConfig(enabled=True),
    ),
)

training_project = TrainingProject(name="ttdr-checkpoint-test", job=training_job)
