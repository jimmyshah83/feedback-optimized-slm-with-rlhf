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
# Edit .env — only endpoints are needed (managed identity, no API keys)
```

### Authentication

This project uses **managed identity** throughout — no API keys are stored in `.env`.

- **Azure AI Foundry** has local auth (API keys) disabled. All access goes through `DefaultAzureCredential`.
- **Azure AI Search** uses RBAC. The Foundry's system-assigned managed identity is granted Search Index Data Reader.
- Your user identity (via `az login`) needs these roles on the relevant resources:

| Role | Resource | Purpose |
|------|----------|---------|
| **Cognitive Services OpenAI User** | AI Foundry account | Use embeddings + chat models |
| **Search Index Data Contributor** | AI Search service | Upload documents to the index |
| **Search Service Contributor** | AI Search service | Create/manage indexes |

Assign roles via:

```bash
# Replace <principal-id> with your user object ID from `az ad signed-in-user show --query id -o tsv`
PRINCIPAL_ID=$(az ad signed-in-user show --query id -o tsv)
RG="ragrlhf-rg"

az role assignment create --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/<sub>/resourceGroups/$RG/providers/Microsoft.CognitiveServices/accounts/ragrlhf-ai

az role assignment create --assignee $PRINCIPAL_ID \
  --role "Search Index Data Contributor" \
  --scope /subscriptions/<sub>/resourceGroups/$RG/providers/Microsoft.Search/searchServices/ragrlhf-search

az role assignment create --assignee $PRINCIPAL_ID \
  --role "Search Service Contributor" \
  --scope /subscriptions/<sub>/resourceGroups/$RG/providers/Microsoft.Search/searchServices/ragrlhf-search
```

### Infrastructure (Azure)

Deploy all Azure resources with Bicep:

```bash
az deployment sub create \
  --location eastus2 \
  --template-file infrastructure/bicep/main.bicep \
  --parameters infrastructure/bicep/main.parameters.json
```

This provisions:
- **Resource group** (`ragrlhf-rg`)
- **Azure AI Foundry** (with system-assigned managed identity, `disableLocalAuth: true`)
  - gpt-5.4 (GlobalStandard) and text-embedding-3-large model deployments
  - An RLAIF project (`rlaif-project`) for agent management and observability
  - AI Search connection (AAD auth) for agent grounding
- **Azure AI Search** (standard SKU with semantic search)
- **Azure Cosmos DB** (for judge results, partitioned by `/question`)
- **Azure Blob Storage** (for model checkpoints + LoRA adapters)
- **Azure ML Workspace** (for experiment tracking + training)
- **RBAC role assignments** (Foundry MI gets Search Index Data Reader + Search Service Contributor)

After deployment, populate your `.env`:

```bash
# Get the project endpoint from deployment output
az deployment sub show --name main --query properties.outputs.foundryProjectEndpoint.value -o tsv
az deployment sub show --name main --query properties.outputs.searchEndpoint.value -o tsv
```

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

Index documents into Azure AI Search, then create a persistent RAG agent in Azure AI Foundry with hybrid vector+semantic search grounding.

**Step 1 — Index documents:**

```bash
uv run index-documents         # Embed with text-embedding-3-large + upload to Azure AI Search
uv run index-documents --rebuild   # Recreate the index from scratch
```

This uses `AIProjectClient.get_openai_client()` for embeddings (managed identity, no API key) and `SearchIndexClient` with `DefaultAzureCredential` for index management. The index includes a vector field (3072d HNSW) and semantic search configuration.

**Step 2 — Create agent and run queries:**

```bash
uv run rag-demo                # Create agent (if needed) + run 5 sample queries
uv run rag-query               # Interactive single query
```

The agent is created as a **persistent named resource** in Azure AI Foundry (`pubmedqa-rag`). It remains available in the Foundry portal across runs — it is never deleted programmatically. The agent uses:
- **Model:** gpt-5.4 via the Foundry project
- **Tool:** Azure AI Search with `vector_semantic_hybrid` query type
- **Instructions:** Medical QA prompt with structured answer format

On subsequent runs, the existing agent is reused (no duplicate creation). You can view and test the agent directly in the Foundry portal under Agents.

**How it works:**
1. `AIProjectClient.agents.get("pubmedqa-rag")` checks for an existing agent
2. If not found, `agents.create_version()` creates a new `PromptAgentDefinition` with `AzureAISearchTool`
3. Queries go through the OpenAI-compatible Assistants API via `project_client.get_openai_client()`
4. The agent searches the index, retrieves relevant PubMedQA documents, and generates grounded answers

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
├── config/settings.yaml            # Model, search, agent, judge, RL loop, training config
├── infrastructure/bicep/
│   ├── main.bicep                  # Subscription-scope deployment orchestrator
│   ├── main.parameters.json        # Default parameters (baseName, location)
│   └── modules/
│       ├── ai_search.bicep         # Azure AI Search (standard, semantic)
│       ├── cosmos.bicep            # Cosmos DB (judge results)
│       ├── foundry.bicep           # AI Foundry v2 (system MI, project, connection, models)
│       ├── ml_workspace.bicep      # Azure ML Workspace
│       ├── role_assignments.bicep  # RBAC (Foundry MI → Search)
│       └── storage.bicep           # Blob Storage (model artifacts)
├── src/
│   ├── config.py                   # Pydantic settings from YAML + .env
│   ├── data/                       # Download, preprocess, index
│   ├── rag/                        # Azure AI Agent Service RAG pipeline
│   ├── evaluation/                 # Azure AI Evaluation benchmarks, comparison
│   ├── judge/                      # AI judge (gpt-5.4) for RLAIF preference pairs
│   ├── training/                   # DPO data prep, QLoRA trainer, adapter merge
│   ├── deployment/                 # Azure ML model registration + endpoint
│   └── pipeline/                   # RL loop orchestrator, Azure ML pipeline
├── tests/                          # Phase-by-phase test suites
├── pyproject.toml                  # Dependencies + uv script entrypoints
└── .env.example                    # Azure endpoint template (no API keys)
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
| **search** | pubmedqa-index, top_k 5, vector_semantic_hybrid query type |
| **agent** | `pubmedqa-rag` named agent, gpt-54 model, medical QA instructions |
| **judge** | gpt-54 deployment, rubric: medical accuracy, faithfulness, completeness, clarity |
| **rl_loop** | 3 iterations, 800 questions/iteration, on-policy, convergence threshold 0.01 |
| **training** | QLoRA rank 16, alpha 32, 4-bit quantization, DPO beta 0.1, lr 5e-5, 3 epochs |
| **evaluation** | Groundedness, Relevance, Response Completeness (Azure AI Evaluation SDK) |

## License

MIT
