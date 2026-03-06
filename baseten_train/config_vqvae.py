"""Baseten training config: VQ-VAE tokenizer + dynamics transformer on 8×H200.

Uses the ttdr-precompute project — Bridge V2 tfrecords pulled from S3.

Phase 1: VQ-VAE tokenizer (100K steps)
Phase 2: Dynamics transformer + value head (100K steps)
"""

from truss_train import TrainingProject, TrainingJob, Image, Compute, Runtime, definitions
from truss.base.truss_config import AcceleratorSpec, Accelerator

training_job = TrainingJob(
    image=Image(
        base_image="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
    ),
    compute=Compute(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H200, count=8),
    ),
    runtime=Runtime(
        start_commands=["chmod +x ./run_vqvae.sh && ./run_vqvae.sh"],
        environment_variables={"hf_token": definitions.SecretReference(name="hf_token")},
        checkpointing_config=definitions.CheckpointingConfig(enabled=True),
    ),
)

training_project = TrainingProject(name="ttdr-precompute", job=training_job)
