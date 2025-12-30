# 🧠 MM-RAG: Multimodal Clinical Decision Support System

A **Multimodal Retrieval-Augmented Generation (RAG)** system for clinical decision support that combines medical imaging analysis with textual radiology reports using a multi-agent architecture powered by LangGraph.

---

## Overview

MM-RAG is a clinical decision support system that leverages multimodal data—medical images and radiology text reports—to provide evidence-based clinical reasoning. The system uses specialized agents orchestrated via LangGraph to:

1. **Route** clinical queries to appropriate imaging modalities (X-Ray, CT, MRI)
2. **Retrieve** relevant patient records using dual-vector similarity search
3. **Analyze** medical images using vision-language models
4. **Generate** grounded clinical responses with citations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
│              "Is there any pulmonary abnormality?"              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Modality Router Agent                         │
│    Analyzes query → Routes to: XRAY / CT / MRI / All            │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  X-Ray   │        │    CT    │        │   MRI    │
    │  Agent   │        │  Agent   │        │  Agent   │
    │          │        │          │        │          │
    │ Qdrant   │        │ Qdrant   │        │ Qdrant   │
    │ Retrieval│        │ Retrieval│        │ Retrieval│
    └──────────┘        └──────────┘        └──────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Evidence Aggregation Agent                       │
│         Merges & deduplicates multimodal evidence               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Image Insight Agent                            │
│              LLaVA 7B via Ollama (VLM Analysis)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Clinical Reasoning Agent                         │
│         DeepSeek-R1:7B → Grounded Diagnosis + Citations         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Output                                │
│     Diagnosis | Supporting Evidence | Recommendations           │
│                  + Evaluation Metrics                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Architecture** | Specialized agents for X-Ray, CT, MRI retrieval |
| **Dual-Vector Retrieval** | Searches both image and text embeddings via Qdrant |
| **Vision-Language Analysis** | LLaVA extracts visual findings from medical images |
| **Grounded Responses** | Every claim includes `[Rx]` citation to source evidence |
| **LangGraph Orchestration** | Declarative workflow with parallel modality execution |
| **Built-in Evaluation** | Precision@K, Recall@K, MRR, Groundedness, Completeness |
| **Multiple VLM Backends** | LLaVA, PaliGemma, BLIP-2, LLaVA-Med, GPT-4V |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph + LangChain |
| **Vector Database** | Qdrant (local persistent storage) |
| **Text Embeddings** | CLIP ViT-B/32 (via FastEmbed) |
| **Image Embeddings** | CLIP ViT-B/32 (via Sentence-Transformers) |
| **Vision-Language Model** | LLaVA 7B (via Ollama) |
| **Clinical Reasoning LLM** | DeepSeek-R1:7B (via Ollama) |
| **Python Version** | 3.10 |

---

## Project Structure

```
mmrag-clinical/
│
├── agents/                              # Multi-agent system
│   ├── langgraph_flow/
│   │   └── mmrag_graph.py              # LangGraph workflow definition
│   ├── modality_router_agent.py        # Query → Modality routing
│   ├── xray_agent.py                   # X-Ray retrieval
│   ├── ct_agent.py                     # CT retrieval
│   ├── mri_agent.py                    # MRI retrieval
│   ├── retrieval_utils.py              # Qdrant dual-vector search
│   ├── evidence_aggregation_agent.py   # Merge multimodal results
│   ├── clinical_reasoning_agent.py     # DeepSeek reasoning + eval
│   ├── image_insight_agent_ollama.py   # LLaVA via Ollama (default)
│
├── embeddings/                          # Embedding modules
│   ├── text_embeddings.py              # CLIP text encoder (512-dim)
│   └── image_embeddings.py             # CLIP image encoder (512-dim)
│
├── evaluation/                          # Metrics
│   └── diagnosis_evaluator.py          # P@K, R@K, MRR, Groundedness
│
├── ingestion/                           # Data pipeline
│   ├── preprocess_dataset.py           # CSV → structured format
│   └── ingest_to_qdrant.py             # Embed + upload to Qdrant
│
├── vectorstore/                         # Vector DB
│   └── qdrant_setup.py                 # Qdrant client + collection
│
├── scripts/
│   └── start_llava_med.py              # LLaVA-Med server launcher
│
├── tests/
│   ├── test_qdrant_basic.py            # Collection verification
│   ├── test_patient_retrieval.py       # Retrieval test
│   └── list_gemini_models.py           # Gemini API explorer
│
├── utils/
│   └── logger.py                       # Logging utility
│
├── data/
│   ├── raw/                            # Source CSV + images
│   │   └── final_multimodal_dataset.csv
│   └── qdrant/                         # Persistent vector store
│
├── app.py                              # Main entry point
├── environment.yml                     # Conda environment
├── requirements.txt                    # Pip dependencies
├── start_ollama.bat                    # Windows Ollama launcher
└── steps.txt                           # Quick start reference
```

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- [Ollama](https://ollama.ai/) installed
- CUDA GPU (recommended for VLMs)

### Step 1: Create Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate
conda activate mmrag-clinical

# Install additional pip dependencies
pip install -r requirements.txt
```

### Step 2: Install Ollama Models

```bash
# Start Ollama (Linux/macOS)
ollama serve

# Or on Windows, run:
start_ollama.bat

# Pull required models
ollama pull deepseek-r1:7b    # Clinical reasoning
ollama pull llava:7b           # Image analysis
```

### Step 3: Prepare Dataset

Place your dataset at `data/raw/final_multimodal_dataset.csv` with these columns:

| Column | Description |
|--------|-------------|
| `patient_id` | Unique patient identifier |
| `uid` | Record unique ID |
| `modality` | XRAY, CT, or MRI |
| `organ` | Target anatomical region |
| `projection` | Image view/orientation |
| `filename` | Path to image file (.png) |
| `indication` | Clinical indication |
| `comparison` | Prior studies comparison |
| `findings` | Radiological findings |
| `impression` | Radiologist impression |
| `MeSH` | Medical Subject Headings |
| `Problems` | Identified problems |

### Step 4: Ingest Data

```bash
# Embed and upload to Qdrant
python -m ingestion.ingest_to_qdrant

# Verify ingestion
python -m tests.test_qdrant_basic
python -m tests.test_patient_retrieval
```

---

## Usage

### Run the Application

```bash
python app.py
```

### Example Session

```
🧠 Multimodal Clinical Decision Support System
==================================================
Enter Patient ID: 1
Enter Clinical Query: Is there any pulmonary abnormality?

[INFO] [RouterNode] Routing modalities
[DEBUG] Modalities selected: ['XRAY']
[INFO] [XRAYAgent] Retrieving XRAY data for patient_id=1
[INFO] [LLaVA] Analyzing image: data/raw/images/patient1_chest.png

🔎 RETRIEVED EVIDENCE:
- [XRAY] Indication: Chest pain
  Findings: Bilateral lower lobe infiltrates...

🧠 FINAL CLINICAL RESPONSE:
Diagnosis / Impression:
- Bilateral pulmonary infiltrates consistent with pneumonia [R1]

Supporting Evidence:
- Chest X-ray shows bilateral opacities [R1]
- No pleural effusion identified [R1-IMAGE]
- Heart size within normal limits [R1]

Next Steps / Recommendations:
- Consider CT chest for further characterization
- Clinical correlation with symptoms recommended

📊 EVALUATION METRICS
========================================
Precision@K: 0.714
Recall@K: 0.833
MRR: 1.0
Groundedness: 0.857
ClinicalCorrectness: 1
Completeness: 1.0
========================================
```

---

## Agents

### 1. Modality Router Agent
**File:** `agents/modality_router_agent.py`

Routes queries based on anatomical keywords:
| Keyword | Selected Modality |
|---------|-------------------|
| pancreas | CT |
| prostate | MRI |
| lung, chest, pulmonary | XRAY |
| other | XRAY + CT + MRI |

### 2. Modality Agents (X-Ray, CT, MRI)
**Files:** `agents/xray_agent.py`, `ct_agent.py`, `mri_agent.py`

Perform filtered retrieval from Qdrant:
- Filter by `patient_id` and `modality`
- Dual-vector search (text + image embeddings)
- Returns top-k similar records

### 3. Evidence Aggregation Agent
**File:** `agents/evidence_aggregation_agent.py`

- Merges results from all modality agents
- Deduplicates by record ID
- Extracts payload (report_text, image_path, etc.)

### 4. Image Insight Agent
**File:** `agents/image_insight_agent_ollama.py` (default)

Uses LLaVA 7B to describe visible findings:
- Anatomical observations only
- No diagnosis or speculation
- Max 3 sentences, ≤12 words each

### 5. Clinical Reasoning Agent
**File:** `agents/clinical_reasoning_agent.py`

Uses DeepSeek-R1:7B to generate:
- **Diagnosis/Impression** (1 sentence with citation)
- **Supporting Evidence** (2-4 bullets with citations)
- **Next Steps** (1-2 recommendations)

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision@K** | `\|Retrieved ∩ Relevant\| / K` | Fraction of retrieved docs that are relevant |
| **Recall@K** | `\|Retrieved ∩ Relevant\| / \|Relevant\|` | Fraction of relevant docs retrieved |
| **MRR** | `1 / rank_of_first_relevant` | Mean Reciprocal Rank |
| **Groundedness** | `cited_sentences / total_sentences` | Proportion with `[Rx]` citations |
| **Clinical Correctness** | `1 if key_terms match else 0` | Alignment with ground truth |
| **Completeness** | `(diag + evidence + recommend) / 3` | Coverage of response sections |

---

## Configuration

### Ollama Endpoint

Default: `http://localhost:11434/api/generate`

Modify in:
- `agents/clinical_reasoning_agent.py`
- `agents/image_insight_agent_ollama.py`

### Qdrant Settings

| Setting | Value |
|---------|-------|
| Storage Path | `data/qdrant/` |
| Collection | `clinical_mmrag` |
| Vector Dimensions | 512 (CLIP) |
| Distance Metric | Cosine |

---

## Troubleshooting

### Ollama Not Starting
```bash
# Check if running
curl http://localhost:11434/api/generate

# Windows: Run as administrator
ollama serve

# Linux: Check logs
journalctl -u ollama
```

### Model Not Found
```bash
# List installed models
ollama list

# Pull missing model
ollama pull deepseek-r1:7b
ollama pull llava:7b
```

### CUDA Out of Memory
```python
# Use CPU fallback in embeddings/image_embeddings.py
DEVICE = "cpu"  # Instead of "cuda"
```

### Empty Retrieval Results
```bash
# Verify data ingestion
python -m tests.test_qdrant_basic

# Check patient exists
python -m tests.test_patient_retrieval
```

<p align="center">
  <b>Built for Clinical AI Research</b> 🏥
</p>