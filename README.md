# Feedback-Optimized SLM with RLHF

A reusable RAG pipeline built on **Microsoft Phi-4-mini** (3.8B), fine-tuned with **DPO** (Direct Preference Optimization) from human feedback, benchmarked with **RAGAS**, and hosted on **Microsoft Azure**.

## Architecture

```
PubMedQA ──> Azure Blob ──> Azure AI Foundry (embed) ──> Azure AI Search
                                                          │
                                      Phi-4-mini ◄───── RAG Pipeline ──> RAGAS Benchmarks
                                          │                   │
                                     DPO Training         Gradio UI ──> Cosmos DB (feedback)
                                     (Azure ML)               │
                                          │              Human Feedback
                                   Phi-4-mini (tuned) ──> RAGAS Benchmarks (post)
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Azure subscription with `az login` completed
- Azure resources deployed (see Infrastructure below)

### Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd feedback-optimized-slm-with-rlhf

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv sync --all-extras

# Configure Azure credentials
cp .env.example .env
# Edit .env with your Azure resource endpoints and keys
```

### Phase 1: Data Download and Preprocessing

```bash
# Download PubMedQA dataset
uv run download-data

# Preprocess into retrieval-ready documents
uv run preprocess

# Run tests
uv run pytest tests/test_data.py -v
```

### Infrastructure (Azure)

Deploy all Azure resources with Bicep (the template creates the resource group automatically):

```bash
az deployment sub create \
  --location eastus2 \
  --template-file infrastructure/bicep/main.bicep \
  --parameters infrastructure/bicep/main.parameters.json
```

Or pass parameters inline:

```bash
az deployment sub create \
  --location eastus2 \
  --template-file infrastructure/bicep/main.bicep \
  --parameters baseName=ragrlhf location=eastus2
```

This provisions: a resource group, Azure AI Search, Azure AI Foundry (with gpt-5.4 and text-embedding-3-large), Azure Cosmos DB, Azure Blob Storage, and Azure ML Workspace.

## Project Structure

```
├── config/settings.yaml         # Model, search, training hyperparameters
├── infrastructure/bicep/        # Azure resource templates
├── src/
│   ├── config.py                # Pydantic settings from YAML + .env
│   ├── data/                    # Download, preprocess, index
│   ├── rag/                     # Retriever, generator, pipeline
│   ├── evaluation/              # RAGAS benchmarks, comparison reports
│   ├── feedback/                # Gradio UI, Cosmos DB storage
│   ├── training/                # DPO data prep, QLoRA trainer, adapter merge
│   └── pipeline/                # CLI orchestrator, Azure ML pipeline
├── tests/                       # Phase-by-phase test suites
├── pyproject.toml               # Dependencies + uv script entrypoints
└── .env.example                 # Azure credential template
```

## CLI Commands (via `uv run`)

| Command | Description | Phase |
|---------|-------------|-------|
| `download-data` | Fetch PubMedQA from Hugging Face | 1 |
| `preprocess` | Process into retrieval-ready docs | 1 |
| `index-documents` | Embed and index into Azure AI Search | 2 |
| `rag-demo` | Run 5 sample queries | 2 |
| `rag-query` | Interactive single query | 2 |
| `benchmark` | Run RAGAS + accuracy evaluation | 3 |
| `serve-feedback` | Launch Gradio feedback UI | 4 |
| `feedback-stats` | Print feedback summary | 4 |
| `prepare-dpo` | Convert feedback to DPO pairs | 5 |
| `train-dpo` | Run DPO fine-tuning | 5 |
| `merge-adapter` | Merge LoRA into base model | 5 |
| `compare` | Generate before/after report | 6 |
| `orchestrate` | Run full pipeline end-to-end | 7 |

## License

MIT
