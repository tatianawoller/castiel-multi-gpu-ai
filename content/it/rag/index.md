# 5M: Retrieval Augmented Generation (RAG)

## Code

```{note}
{repo}`it/rag`
```

This directory contains materials for implementing Retrieval Augmented Generation (RAG) pipelines using large language models.

## Setup and Configuration

- {download}`0_env_config.sh`: Environment configuration script
- {download}`1_start_jupyter.sh`: SLURM job script for starting Jupyter with RAG environment
- {download}`requirements_rag.txt`: Python dependencies for RAG implementation

The `data/` directory contains sample input and output files for RAG pipelines.

```{toctree}
:maxdepth: 1
:caption: Data Files

data/input/index
```

```{toctree}
:maxdepth: 1
:caption: Notebooks

notebooks/index
```

## Getting Started

To run the RAG environment on Leonardo:

1. Submit the job script:
   ```bash
   sbatch 1_start_jupyter.sh
   ```

2. Follow the SSH tunnel instructions to access the Jupyter server

3. The environment includes:
   - vLLM endpoint for model serving
   - Pre-configured Hugging Face cache
   - All necessary RAG dependencies

## Key Features

- Multi-GPU support with tensor parallelism
- vLLM for efficient LLM serving
- ChromaDB for vector storage
- LangChain integration

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)