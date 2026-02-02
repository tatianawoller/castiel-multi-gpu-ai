# 3A: Accelerate FSDP Fine-Tuning

This directory contains code for fine-tuning large language models using Accelerate with FSDP (Fully Sharded Data Parallel) on multiple GPUs.

## Code

```{note}
See {repo}`it/accelerate_fsdp`
```

- {download}`config_FSDP.yaml`: Configuration file for Accelerate FSDP setup
- {download}`finetune.py`: Main training script using Accelerate and peft to perform SFTTrainer with low-rank adapter (LoRA)
- {download}`job.sh`: SLURM job script for Leonardo Booster HPC cluster
- {download}`requirements.txt`: Python dependencies
