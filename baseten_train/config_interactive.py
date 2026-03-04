"""Baseten config: interactive session. Keeps container alive for 24h.
Connect via rSSH in VS Code/Cursor, run commands manually."""

from truss_train import TrainingProject, TrainingJob, Image, Compute, Runtime
from truss.base.truss_config import AcceleratorSpec, Accelerator

training_job = TrainingJob(
    image=Image(
        base_image="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
    ),
    compute=Compute(
        accelerator=AcceleratorSpec(accelerator=Accelerator.H100, count=4),
    ),
    runtime=Runtime(
        start_commands=[
            "echo 'Container ready. Connect via rSSH in VS Code/Cursor.' && sleep 86400"
        ],
    ),
)

training_project = TrainingProject(name="ttdr-interactive", job=training_job)
