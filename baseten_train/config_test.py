"""Baseten config: quick end-to-end pipeline test on 1×H100."""

from truss_train import TrainingProject, TrainingJob, Image, Compute, Runtime, definitions
from truss.base.truss_config import AcceleratorSpec, Accelerator

training_job = TrainingJob(
    image=Image(
        base_image="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
    ),
    compute=Compute(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=1),
    ),
    runtime=Runtime(
        start_commands=["chmod +x ./run_test.sh && ./run_test.sh"],
        environment_variables={"hf_token": definitions.SecretReference(name="hf_token")},
        checkpointing_config=definitions.CheckpointingConfig(enabled=True, volume_size_gib=500),
    ),
)

training_project = TrainingProject(name="ttdr-test", job=training_job)
