# Tools

Tools required for this training.  The environment can be created using conda if required.
HPC systems such as Leonardo provide a module that contains all the necessary tools.


## What is it?

1. {download}`environment_full_specs.yml`: conda environment file for installing the necessary
   dependencies, all dependencies with explicit versions (less portable).
1. {download}`environment_from_history.yml`: conda environment file for installing the necessary
   dependencies, only high-level packages (more portable, less reproducible); see below:
1. {download}`requirements.txt`: requirements file for installing the necessary dependencies.

```{literalinclude} environment_from_history.yml
:language: yaml
```
