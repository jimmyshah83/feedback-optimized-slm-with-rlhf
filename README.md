# Feedback-Optimized SLM with RLAIF

A reusable pattern for iteratively improving a Small Language Model using **RLAIF** (Reinforcement Learning from AI Feedback). Built on **Microsoft Phi-4-mini** (3.8B) running locally via **Ollama**, fine-tuned with **DPO** via an AI judge (gpt-5.4), benchmarked with **Azure AI Evaluation SDK**, with retrieval powered by **Azure AI Search** — a hybrid local+cloud architecture on **Microsoft Azure**.

## The RLAIF Pattern

This project implements an **iterative, on-policy DPO loop** where a strong AI model (gpt-5.4) acts as the judge to generate preference signals for a smaller model (Phi-4-mini). Each iteration generates fresh responses from the current policy, judges them against gold-standard answers, trains with DPO, and benchmarks the result — creating a measurable improvement curve.

```
┌─────────────────────────────────────────────────────────────────┐
│  For each iteration i = 1..N:                                   │
│                                                                 │
│  1. Phi-4-mini (policy_i) generates RAG responses               │
│     for 800 training questions via Ollama (local)               │
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
| **SLM (policy model)** | Microsoft Phi-4-mini (3.8B) via Ollama (local) — swappable after each DPO iteration |
| **AI Judge** | gpt-5.4 via native Azure OpenAI SDK |
| **RAG retrieval** | Azure AI Search (vector + semantic hybrid) with AzureOpenAI embeddings (text-embedding-3-large) |
| **RAG generation** | Phi-4-mini via Ollama (OpenAI-compatible API at `localhost:11434`) |
| **Evaluation** | Azure AI Evaluation SDK (`azure-ai-evaluation`) |
| **Training** | TRL `DPOTrainer` + PEFT QLoRA + bitsandbytes 4-bit |
| **Data** | PubMedQA (1,000 expert-annotated medical QA pairs) |
| **Observability** | Azure AI Foundry portal + MLflow experiment tracking |
| **Infrastructure** | Bicep IaC (AI Search, Foundry v2 with Project, Cosmos DB, Blob, ML Workspace) |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) for local SLM inference
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

# Pull the SLM model for local inference
ollama pull phi4-mini
```

### Authentication

This project uses **managed identity** throughout — no API keys are stored in `.env`.

- **Azure AI Foundry** has local auth (API keys) disabled. All access goes through `DefaultAzureCredential`.
- **Azure AI Search** has RBAC data plane auth enabled (`aadOrApiKey`). The Foundry's system-assigned managed identity is granted Search Index Data Reader.
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

Index documents into Azure AI Search, then run the RAG pipeline using **phi-4-mini (local via Ollama)** for generation and **Azure AI Search** for retrieval.

**Step 1 — Index documents:**

```bash
uv run index-documents         # Embed with text-embedding-3-large + upload to Azure AI Search
uv run index-documents --rebuild   # Recreate the index from scratch
```

This uses `AzureOpenAI` with `DefaultAzureCredential` for embeddings (managed identity, no API key) and `SearchIndexClient` with `DefaultAzureCredential` for index management. The index includes a vector field (3072d HNSW) and semantic search configuration.

**Step 2 — Run queries (requires Ollama running):**

```bash
ollama serve                   # Start Ollama (if not running as a service)
uv run rag-demo                # Create agent def + run 5 sample queries via phi-4-mini
uv run rag-query               # Interactive single query
```

**Architecture — split local + cloud:**

| Component | Where | What |
|-----------|-------|------|
| **Generation** | Local (Ollama) | Phi-4-mini answers questions grounded in evidence |
| **Embeddings** | Azure (Foundry) | text-embedding-3-large encodes queries for search |
| **Retrieval** | Azure (AI Search) | Vector + semantic hybrid search over pubmedqa-index |
| **Agent definition** | Azure (Foundry) | `pubmedqa-rag` agent visible in Foundry portal |

The agent definition is stored as a **persistent named resource** in Azure AI Foundry, visible in the portal. At runtime, queries use a **search-then-generate** pattern:

1. `AzureOpenAI` embeds the question (text-embedding-3-large)
2. `SearchClient` runs vector + semantic hybrid search against the index
3. **Phi-4-mini via Ollama** generates a grounded answer from the retrieved evidence

> **Why local?** Each DPO iteration produces new model weights. Running phi-4-mini locally lets us swap in fine-tuned models instantly — no redeployment needed. The same Ollama OpenAI-compatible API (`localhost:11434/v1`) works for base and fine-tuned models.

### Phase 3: Baseline Evaluation

Benchmark the base Phi-4-mini RAG pipeline on the held-out evaluation set (200 questions) using **Azure AI Evaluation SDK**. Results are scored by gpt-5.4 and logged to Azure AI Foundry for portal visibility.

```bash
ollama serve                                    # Ensure Ollama is running
uv run benchmark --tag baseline --samples 200   # Full eval set (200 questions)
uv run benchmark --tag smoke --samples 5        # Quick smoke test
uv run pytest tests/test_evaluation.py -v       # Unit tests
```

**What happens:**

1. **Generate responses** — runs phi-4-mini RAG (Ollama + Azure AI Search) on each eval question
2. **PubMedQA classification accuracy** — extracts yes/no/maybe from model output and compares to gold labels
3. **Azure AI Evaluation** — gpt-5.4 scores each response for Groundedness, Relevance, and Response Completeness (1–5 scale)
4. **Upload to Foundry** — results are logged to your Azure AI Foundry project and visible in the portal under "Evaluations"

**Metrics produced:**

| Metric | Source | Scale |
|--------|--------|-------|
| `groundedness` | Azure AI Evaluation (gpt-5.4 judge) | 1–5 |
| `relevance` | Azure AI Evaluation (gpt-5.4 judge) | 1–5 |
| `response_completeness` | Azure AI Evaluation (gpt-5.4 judge) | 1–5 |
| `pubmedqa_accuracy` | Custom (yes/no/maybe classification) | 0–100% |

Results are saved to `data/evaluations/` as both per-row JSONL and a JSON summary. The Foundry portal URL is printed at the end of the run.

### Phase 4: AI Judge

Generate **DPO preference pairs** using gpt-5.4 as the AI judge. For each training question, phi-4-mini produces two candidate responses at different temperatures (0.3 and 0.9), then gpt-5.4 evaluates both against the PubMedQA gold answer on four rubric dimensions.

```bash
ollama serve                                     # Ensure Ollama is running
uv run judge --iteration 0 --samples 800         # Full training set
uv run judge --iteration 0 --samples 10          # Quick smoke test
uv run pytest tests/test_judge.py -v             # Unit tests (15 tests)
```

**What happens:**

1. **Load training set** — first 800 questions from `pubmedqa_processed.jsonl`
2. **Generate two candidates** — phi-4-mini via Ollama answers each question twice (low temp for precision, high temp for diversity)
3. **AI judge evaluation** — gpt-5.4 scores both candidates on medical accuracy, faithfulness, completeness, and clarity (1–5 each)
4. **Preference pair** — the higher-scored response becomes `chosen`, the lower becomes `rejected`

**Output:** `data/judge/preferences_iter{N}.jsonl` — one JSON object per line with `chosen`, `rejected`, rubric scores, and the full judge verdict. These pairs feed directly into Phase 5 (DPO training).

**Rubric dimensions:**

| Dimension | What the judge evaluates |
|-----------|------------------------|
| `medical_accuracy` | Factual correctness relative to the gold answer |
| `faithfulness` | How well grounded in the retrieved evidence |
| `completeness` | Coverage of key points from the gold answer |
| `clarity` | Structure, readability, and final decision |

> **Rate limiting:** gpt-5.4 calls include automatic retry with exponential backoff (60–120s). For the S0 tier, 800 questions takes approximately 1–2 hours depending on quota.

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
│   ├── rag/                        # RAG pipeline (search + generate) with Foundry agent def
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
| **slm** | Ollama base URL `localhost:11434/v1`, model `phi4-mini` |
| **embedding** | text-embedding-3-large, 3072 dimensions |
| **search** | pubmedqa-index, top_k 5, vector_semantic_hybrid query type |
| **agent** | `pubmedqa-rag` named agent, phi4-mini model, medical QA instructions |
| **judge** | gpt-54 deployment, rubric: medical accuracy, faithfulness, completeness, clarity |
| **rl_loop** | 3 iterations, 800 questions/iteration, on-policy, convergence threshold 0.01 |
| **training** | QLoRA rank 16, alpha 32, 4-bit quantization, DPO beta 0.1, lr 5e-5, 3 epochs |
| **evaluation** | Groundedness, Relevance, Response Completeness (Azure AI Evaluation SDK) |

## License

MIT
