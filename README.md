# PrivFusion: LLM-Powered Heterogeneous Data Consolidation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Official Implementation** of "PrivFusion: A Privacy-preserving Multi-Agent Framework for Harmonizing Distributed Datasets"

📄 [Paper](https://arxiv.org/abs/2605.24249)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Running Experiments](#running-experiments)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

**PrivFusion** is a novel framework that leverages Large Language Models (LLMs) to automatically align and consolidate heterogeneous tabular datasets with overlapping but differently structured features. This repository contains the official implementation of our paper.

### Problem Statement

Organizations often need to consolidate multiple datasets from different sources that describe similar entities but use different schemas, naming conventions, and data representations. Traditional approaches require extensive manual effort and domain expertise.

### Our Solution

PrivFusion uses LLMs to:
1. **Semantically cluster** similar features across datasets
2. **Normalize** feature representations to a unified schema
3. **Generate** transformation code to align data values
4. **Validate** transformations with comprehensive quality metrics

### Key Contributions

- 🎯 **Novel LLM-based approach** for automated dataset consolidation
- 🔄 **End-to-end pipeline** from feature clustering to code generation
- 📊 **Comprehensive evaluation** with fidelity, privacy, and statistical metrics
- 🔌 **Flexible architecture** supporting multiple LLM backends
- 🎓 **Semantic type detection** using DBpedia ontology

## Key Features

- 🤖 **LLM-Powered Analysis**: Leverages state-of-the-art language models for semantic understanding
- 🔄 **Automated Transformations**: Generates Python code to transform data between schemas
- 📊 **Multiple Metrics**: Includes fidelity, privacy, and statistical metrics for data quality assessment
- 🔌 **Flexible LLM Backends**: Supports WatsonX, Ollama, and custom LLM endpoints
- 🎯 **Semantic Type Detection**: Automatically identifies and maps semantic types using DBpedia URIs
- 📝 **Experiment Tracking**: YAML-based configuration for reproducible experiments

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/IBM/privfusion.git
cd privfusion

# Using uv (recommended - fastest)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --extra dev  # Installs all dependencies including dev extras
pre-commit install

# Or using uv pip
uv pip install -e .[dev]
pre-commit install

# Or using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
pre-commit install
```

### Environment Configuration

Create a `.env` file in the project root (see `.env.example`):

```bash
# For WatsonX
watsonx_apikey=your_watsonx_api_key
watsonx_project_id=your_project_id

# For RITS endpoint (optional)
rits_api_key=your_rits_api_key
```

## Quick Start

### Basic Usage

```python
from privfusion.consolidater import Consolidator
from privfusion.agents.llms import OllamaLLM
from privfusion.dataset_analyzer import DatasetAnalyzer
import pandas as pd

# Initialize LLM
llm = OllamaLLM(model_name="llama3.2", temperature=0)

# Analyze datasets
analyzer = DatasetAnalyzer(llm)
dataset1_info = analyzer.analyze(pd.read_csv("data/dataset1.csv"))
dataset2_info = analyzer.analyze(pd.read_csv("data/dataset2.csv"))

# Prepare datasets
datasets = {
    "dataset1": {
        "data": pd.read_csv("data/dataset1.csv"),
        "info": dataset1_info
    },
    "dataset2": {
        "data": pd.read_csv("data/dataset2.csv"),
        "info": dataset2_info
    }
}

# Run consolidation
consolidator = Consolidator()
result = consolidator.consolidate(datasets, llm)

# View results
print(result[['dataset', 'feature_name', 'cluster_id', 'norm_feature_name']])
```

### Interactive Notebooks

Explore the framework through our Jupyter notebooks:

- **`01-getting-started.ipynb`** - Introduction and basic usage
- **`02-generate-data.ipynb`** - Synthetic data generation
- **`03-consolidate.ipynb`** - Full consolidation workflow
- **`04-run-experiments.ipynb`** - Running configured experiments
- **`05-show-experiments.ipynb`** - Analyzing and visualizing experiments

## Configuration

Experiments are configured using YAML files. Example structure:

```yaml
datasets:
  - name: dataset1
    path: data/dataset1.csv
  - name: dataset2
    path: data/dataset2.csv

cluster:
  system_prompt: >
    Analyze and cluster semantically similar features...
  llm: privfusion.agents.llms.OllamaLLM
  args:
    model_name: llama3.2
  kwargs:
    temperature: 0
    max_tokens: 5000

normalize:
  system_prompt: >
    Normalize clustered features to unified schema...
  llm: privfusion.agents.llms.OllamaLLM
  number_samples: 5

transform:
  system_prompt: >
    Generate transformation code...
  llm: privfusion.agents.llms.OllamaLLM
  number_samples: 5

experiment:
  max_iter: 3
```

See `configs/README.md` for detailed configuration options.

## Architecture

### Core Components

```
PrivFusion Pipeline
├── DatasetAnalyzer      # Extract semantic & structural information
├── AgentCluster         # Cluster similar features across datasets
├── AgentNorm            # Normalize to unified schema
├── AgentCode            # Generate transformation code
└── Consolidator         # Orchestrate the pipeline
```

### Component Details

- **DatasetAnalyzer**: Extracts semantic types using DBpedia, analyzes data distributions
- **AgentCluster**: Uses LLMs to identify semantically similar features across datasets
- **AgentNorm**: Normalizes feature names, types, and value structures
- **AgentCode**: Generates Python transformation code with validation
- **Consolidator**: Manages the end-to-end consolidation workflow

### LLM Backends

- **WatsonXLLM**: IBM WatsonX AI integration
- **OllamaLLM**: Local Ollama models (Llama3.2, Mistral, etc.)
- **RITSLLM**: Custom RITS endpoint support

### Metrics

The framework includes comprehensive evaluation metrics:

- **Fidelity Metrics**: Measure preservation of data patterns and relationships
- **Privacy Metrics**: Assess privacy preservation during consolidation
- **Statistical Metrics**: Compare statistical properties between original and consolidated data

## Running Experiments

### Available Datasets

The framework includes several datasets for experimentation (available in `data/`):

1. **COVID-19 Datasets**
   - `covid19-dataset.csv` - Global COVID-19 statistics
   - `covid19-indonesia.csv` - Indonesia-specific COVID-19 data
   - `covid_19_indonesia_time_series_all.csv` - Time series data
   - `covid19_italy_province.csv` - Italy provincial data

2. **Adult Income Dataset**
   - `adult_dataset/adult.csv` - UCI Adult dataset

### Running Experiments

```bash
# Run a specific experiment configuration
python -m notebooks.04-run-experiments --config configs/experiment_1.yaml

# Run all experiments
for config in configs/experiment_*.yaml; do
    python -m notebooks.04-run-experiments --config $config
done

# Analyze experiments
python -m notebooks.05-show-experiments
```

### Pre-configured Experiments

Pre-configured experiments are available in `configs/`:

- `experiment_1.yaml` - COVID-19 global consolidation
- `experiment_2.yaml` - COVID-19 regional analysis
- `experiment_3.yaml` - Multi-source COVID-19 integration
- `experiment_4.yaml` - Adult dataset experiments
- `experiment_5-7.yaml` - Ablation studies

### Evaluation Methodology

The framework evaluates consolidation quality through:

1. **Clustering Quality**
   - Precision, recall, F1-score
   - Semantic similarity scores

2. **Normalization Accuracy**
   - Schema alignment correctness
   - Type mapping accuracy

3. **Transformation Quality**
   - Code execution success rate
   - Data fidelity preservation
   - Statistical distribution similarity

## Project Structure

```
privfusion/
├── configs/              # Experiment configurations
├── data/                 # Datasets
├── notebooks/            # Jupyter notebooks
├── src/
│   ├── metrics/          # Evaluation metrics
│   ├── privfusion/       # Core framework
│   │   ├── agents/       # LLM agents
│   │   │   ├── agent_cluster.py
│   │   │   ├── agent_norm.py
│   │   │   ├── agent_code.py
│   │   │   └── llms.py
│   │   ├── consolidater.py
│   │   ├── dataset_analyzer.py
│   │   ├── data_models.py
│   │   └── utils/
│   └── tabular_data/     # Synthetic data generation
├── tests/                # Unit tests
└── requirements.txt      # Dependencies
```

## Citation

If you use PrivFusion in your research, please cite our paper:

```bibtex
@article{privfusion2026,
  title={{PrivFusion}: A Privacy-preserving Multi-Agent Framework for Harmonizing Distributed Datasets},
  author={Anisa Halimi and Liubov Nedoshivina and Kieran Fraser and Stefano Braghin},
  journal={arXiv preprint arXiv:2605.24249},
  year={2026},
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests: `pytest`
5. Run linters: `ruff check src/ && ruff format --check src/ && mypy src/`
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_mapping.py -v
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgment

- This project is partly supported by the Innovative Health Initiative Joint Undertaking (IHI JU) under grant agreement No. 101172997 – SEARCH.
- Built with [LangChain](https://github.com/langchain-ai/langchain) for LLM orchestration
- Uses [READI](https://github.com/IBM/READI) for semantic type detection
- Powered by IBM WatsonX, Ollama, and other LLM providers
- COVID-19 datasets from [Our World in Data](https://ourworldindata.org/coronavirus)
- Adult dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

## Contact

For questions about the paper or code:

- **Issues**: [GitHub Issues](https://github.com/IBM/privfusion/issues)


---

**Note**: This is research code. For production use, additional testing and optimization may be required.

--

<div align="center">

**Built with ❤️ by IBM Research**

[Documentation](https://github.com/IBM/READI) • [Examples](./notebooks) • [Contributing](CONTRIBUTING.md) • [License](LICENSE)

</div>
