"""Baseten config: run WM diagnostics on real data (1×H100, quick job)."""

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
        start_commands=["chmod +x ./run_diagnostics.sh && ./run_diagnostics.sh"],
        environment_variables={},
    ),
)

training_project = TrainingProject(name="ttdr-diagnostics", job=training_job)
