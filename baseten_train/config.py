"""Baseten training config: full pipeline (precompute + train) on 4×H100."""

from truss_train import TrainingProject, TrainingJob, Image, Compute, Runtime, definitions
from truss.base.truss_config import AcceleratorSpec, Accelerator

training_job = TrainingJob(
    image=Image(
        base_image="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
    ),
    compute=Compute(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=4),
    ),
    runtime=Runtime(
        start_commands=["chmod +x ./run.sh && ./run.sh"],
        environment_variables={"hf_token": definitions.SecretReference(name="hf_token")},
        checkpointing_config=definitions.CheckpointingConfig(enabled=True),
    ),
)

training_project = TrainingProject(name="ttdr-precompute", job=training_job)
