# Feedback-Optimized SLM with RLAIF

A reusable pattern for iteratively improving a Small Language Model using **RLAIF** (Reinforcement Learning from AI Feedback). Built on **Microsoft Phi-4-mini** (3.8B), fine-tuned with **DPO** via an AI judge (gpt-5.4), benchmarked with **Azure AI Evaluation SDK**, with RAG powered by **Azure AI Agent Service** — all hosted on **Microsoft Azure**.

## The RLAIF Pattern

This project implements an **iterative, on-policy DPO loop** where a strong AI model (gpt-5.4) acts as the judge to generate preference signals for a smaller model (Phi-4-mini). Each iteration generates fresh responses from the current policy, judges them against gold-standard answers, trains with DPO, and benchmarks the result — creating a measurable improvement curve.

```
┌─────────────────────────────────────────────────────────────────┐
│  For each iteration i = 1..N:                                   │
│                                                                 │
│  1. Phi-4-mini (policy_i) generates RAG responses               │
│     for 800 training questions via Azure AI Agent Service       │
│                                                                 │
│  2. gpt-5.4 (AI judge) evaluates responses against gold        │
│     answers → preference pairs (chosen / rejected)              │
│                                                                 │
│  3. DPO + QLoRA trains policy_i → policy_{i+1}                 │
│                                                                 │
│  4. Azure AI Evaluation benchmarks policy_{i+1}                │
│     on 200 held-out questions                                   │
│                                                                 │
│  5. If improved and not converged → next iteration              │
└─────────────────────────────────────────────────────────────────┘
```

### Why RLAIF + DPO?

| Aspect | This pattern |
|--------|-------------|
| **Feedback source** | AI judge (gpt-5.4) — scalable, consistent, no human bottleneck |
| **Training algorithm** | DPO — no reward model needed, directly optimizes from preference pairs |
| **Efficiency** | QLoRA (4-bit quantization + LoRA rank 16) — trains on a single GPU |
| **On-policy data** | Each iteration generates responses from the *current* model, ensuring the training signal stays relevant |
| **Gold anchoring** | PubMedQA expert answers anchor the judge, so preferences reflect medical accuracy, not just fluency |

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **SLM (policy model)** | Microsoft Phi-4-mini-instruct (3.8B) |
| **AI Judge** | gpt-5.4 via native Azure OpenAI SDK |
| **RAG** | Azure AI Agent Service (`azure-ai-projects`) with Azure AI Search grounding |
| **Evaluation** | Azure AI Evaluation SDK (`azure-ai-evaluation`) |
| **Training** | TRL `DPOTrainer` + PEFT QLoRA + bitsandbytes 4-bit |
| **Data** | PubMedQA (1,000 expert-annotated medical QA pairs) |
| **Observability** | Azure AI Foundry portal + MLflow experiment tracking |
| **Infrastructure** | Bicep IaC (AI Search, Foundry v2 with Project, Cosmos DB, Blob, ML Workspace) |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Azure subscription with `az login` completed
- Azure resources deployed (see Infrastructure below)

### Setup

```bash
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

### Infrastructure (Azure)

Deploy all Azure resources with Bicep:

```bash
az deployment sub create \
  --location eastus2 \
  --template-file infrastructure/bicep/main.bicep \
  --parameters infrastructure/bicep/main.parameters.json
```

This provisions: a resource group, Azure AI Search, Azure AI Foundry (with gpt-5.4, text-embedding-3-large, and an RLAIF project), Azure Cosmos DB, Azure Blob Storage, and Azure ML Workspace.

Preview changes before deploying:

```bash
az deployment sub what-if \
  --location eastus2 \
  --template-file infrastructure/bicep/main.bicep \
  --parameters baseName=ragrlhf location=eastus2
```

## Phases

### Phase 1: Data Preparation

Download PubMedQA (1,000 expert-annotated medical QA pairs) and preprocess into retrieval-ready JSONL documents.

```bash
uv run download-data          # Fetch from Hugging Face
uv run preprocess              # Transform to flat JSONL
uv run pytest tests/test_data.py -v   # Verify (14 tests)
```

**Data split:** 800 questions for training (used each RL iteration) + 200 held-out for evaluation.

### Phase 2: RAG Pipeline

Index documents into Azure AI Search and build the RAG pipeline using Azure AI Agent Service with hybrid search grounding.

```bash
uv run index-documents         # Embed + index into Azure AI Search
uv run rag-demo                # Run 5 sample queries
uv run rag-query               # Interactive single query
```

### Phase 3: Baseline Evaluation

Benchmark the base Phi-4-mini RAG pipeline on the held-out evaluation set using Azure AI Evaluation SDK.

```bash
uv run benchmark --tag baseline --samples 200
```

Metrics: Groundedness, Relevance, Response Completeness (via gpt-5.4 evaluator), plus PubMedQA classification accuracy.

### Phase 4: AI Judge

Run the AI judge to generate preference pairs. For each training question, Phi-4-mini generates two candidate responses; gpt-5.4 judges which is better using the gold answer as reference.

```bash
uv run judge --iteration 0     # Generate preference pairs for base model
```

### Phase 5: DPO Training

Train Phi-4-mini with DPO using the AI judge's preference pairs and QLoRA for efficiency.

```bash
uv run prepare-dpo --iteration 0       # Format pairs for DPOTrainer
uv run train-dpo --iteration 0         # QLoRA DPO fine-tuning
uv run merge-adapter --iteration 0     # Merge LoRA into base model
```

### Phase 6: RL Loop

Orchestrate the full iterative RLAIF pipeline: generate -> judge -> train -> benchmark -> repeat for N iterations.

```bash
uv run rl-loop --iterations 3 --questions 800 --eval-samples 200
uv run compare                 # Cross-iteration comparison report
```

### Phase 7: Model Deployment

Register the best iteration's model in Azure ML and deploy to a managed online endpoint.

```bash
uv run register-model          # Register in Azure ML Model Registry
uv run deploy-model            # Deploy to managed endpoint
uv run test-endpoint           # Send test query to endpoint
```

## Project Structure

```
├── config/settings.yaml         # Model, search, judge, RL loop, training config
├── infrastructure/bicep/        # Azure resource templates (Foundry v2 with Project)
├── src/
│   ├── config.py                # Pydantic settings from YAML + .env
│   ├── data/                    # Download, preprocess, index
│   ├── rag/                     # Azure AI Agent Service retriever, generator, pipeline
│   ├── evaluation/              # Azure AI Evaluation benchmarks, comparison reports
│   ├── judge/                   # AI judge (gpt-5.4) for RLAIF preference pairs
│   ├── training/                # DPO data prep, QLoRA trainer, adapter merge
│   ├── deployment/              # Azure ML model registration + endpoint deployment
│   └── pipeline/                # RL loop orchestrator, Azure ML pipeline
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
| `benchmark` | Run Azure AI Evaluation benchmarks | 3 |
| `judge` | AI judge generates preference pairs | 4 |
| `prepare-dpo` | Format preference pairs for DPO | 5 |
| `train-dpo` | Run QLoRA DPO fine-tuning | 5 |
| `merge-adapter` | Merge LoRA into base model | 5 |
| `compare` | Cross-iteration comparison report | 6 |
| `rl-loop` | Run full iterative RLAIF pipeline | 6 |
| `register-model` | Register model in Azure ML | 7 |
| `deploy-model` | Deploy to managed endpoint | 7 |
| `test-endpoint` | Test deployed endpoint | 7 |

## Key Configuration (`config/settings.yaml`)

| Section | Key settings |
|---------|-------------|
| **model** | Phi-4-mini-instruct, 512 max tokens, temperature 0.3 |
| **embedding** | text-embedding-3-large, 3072 dimensions |
| **search** | pubmedqa-index, top_k 5, semantic search |
| **judge** | gpt-54 deployment, rubric: medical accuracy, faithfulness, completeness, clarity |
| **rl_loop** | 3 iterations, 800 questions/iteration, on-policy, convergence threshold 0.01 |
| **training** | QLoRA rank 16, alpha 32, 4-bit quantization, DPO beta 0.1, lr 5e-5, 3 epochs |
| **evaluation** | Groundedness, Relevance, Response Completeness (Azure AI Evaluation SDK) |

## License

MIT
