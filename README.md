# Intelligent Candidate Discovery & Ranking — Redrob Hackathon

Welcome to the **Intelligent Candidate Discovery & Ranking** repository. This is an agentic AI-driven pipeline designed to parse a Job Description (JD), retrieve candidates from a 100,000-candidate pool, score and rank them, detect and filter honeypots, generate rank explanations, and export a validated submission.

---

## 📂 Repository Structure

The project follows a modular and clean structure separating source code, configurations, data inputs/caches, documentation, outputs, and unit testing:

```
AI_HACKATHON/
│
├── configs/                   # System configurations
│   ├── retrieval.yaml         # Candidate retrieval settings
│   ├── ranking.yaml           # Multicriteria scoring & honeypot filtering weights
│   ├── llm.yaml               # LLM model options (Gemini, OpenAI, etc.)
│   └── evaluation.yaml        # Evaluation metrics & submission export paths
│
├── data/                      # Data storage & pipeline cache
│   ├── raw/                   # Uncompressed raw data (e.g. candidates.jsonl)
│   ├── processed/             # Cleaned/processed intermediate states
│   ├── embeddings/            # Cached candidate embeddings (MinHash or Vector)
│   └── cache/                 # API caches and state checkpoints
│
├── docs/                      # Technical Documentation
│   ├── architecture.md        # Architectural system components and diagrams
│   ├── api_contracts.md       # Schemas & interfaces between modules
│   ├── scoring_logic.md       # Detail on ranking mathematical scoring & traps
│   └── workflow.md            # Execution sequence of agentic steps
│
├── notebooks/                 # Jupyter Notebooks for exploratory data analysis (EDA)
│
├── outputs/                   # Output storage
│   ├── rankings/              # Top candidate rank lists
│   ├── reports/               # Evaluation metric reports
│   ├── explanations/          # LLM explanations for rankings
│   └── submissions/           # Final submission CSVs
│
├── src/                       # Main Python codebase
│   ├── agents/                # LangGraph/Multi-agent systems code
│   ├── ingestion/             # Candidate record parsing and stream loaders
│   ├── preprocessing/         # Text cleaning and preprocessing functions
│   ├── jd_parser/             # Parses requirements from Job Descriptions
│   ├── retrieval/             # Fast candidate pre-filtering (hybrid/BM25)
│   ├── ranking/               # Multi-criteria rank scoring
│   ├── explainability/        # Justification generation for rankings
│   ├── evaluation/            # Internal validation and metrics evaluation
│   ├── graph/                 # State graph definition
│   ├── models/                # LLM and Embedding wrapper APIs
│   ├── prompts/               # System and agent prompt templates
│   ├── utils/                 # Utility functions and common tools
│   ├── interfaces/            # Pipeline base class definitions
│   ├── schemas/               # Pydantic validation models
│   ├── state/                 # State-tracking classes
│   └── main.py                # Pipeline entrypoint execution script
│
├── tests/                     # Unit & integration tests
│
└── README.md                  # This documentation
```

---

## 🚦 Getting Started

### 1. Requirements

Make sure you have the required dependencies installed:
```bash
pip install pandas pyyaml python-docx pydantic
```

### 2. Prepare the Candidate Pool

Make sure you unpack the candidate dataset as described in `README.txt`:
```bash
gunzip -k candidates.jsonl.gz
```

### 3. Run the Pipeline

Execute the main pipeline driver:
```bash
python -m src.main
```
This runs the full pipeline end-to-end, loading configs, reading `job_description.txt`, running retrieval/ranking, generating explanations, writing the final submission CSV, and triggering local validation checks.

### 4. Validate Submission

To validate any external or intermediate submission file, execute:
```bash
python validate_submission.py outputs/submissions/team_submission.csv
```

---

## 📘 Detailed Documentation

- 🔍 **Architecture & Design**: See [architecture.md](file:///c:/Users/piyush%20pawar/Desktop/AI_Hackthonnnn/docs/architecture.md) for flow diagrams and layout.
- 📜 **Data Interfaces**: See [api_contracts.md](file:///c:/Users/piyush%20pawar/Desktop/AI_Hackthonnnn/docs/api_contracts.md) for model data structures.
- 🎯 **Scoring & Honeypot Filters**: See [scoring_logic.md](file:///c:/Users/piyush%20pawar/Desktop/AI_Hackthonnnn/docs/scoring_logic.md) for how the ranking works and honeypots are avoided.
- 🔄 **Execution Workflow**: See [workflow.md](file:///c:/Users/piyush%20pawar/Desktop/AI_Hackthonnnn/docs/workflow.md) to trace pipeline sequential operations.
