# Redrob Hackathon — Participant Bundle

Welcome to the **Intelligent Candidate Discovery & Ranking Challenge**.

## What's in this bundle

| File / Folder | Description |
| :--- | :--- |
| File / Folder | Description |
| :--- | :--- |
| `candidates.jsonl.gz` | The 100,000-candidate pool you'll rank. Gzipped JSONL (~52 MB compressed, ~465 MB uncompressed). |
| `sample_candidates.json` | First 50 candidates as pretty-printed JSON. Use this to inspect the schema quickly. |
| `job_description.md` | The job description you're ranking candidates against. Read it carefully — including the section at the end specifically for hackathon participants. |
| `submission_spec.md` | Read this in full before starting. Submission format, rules, compute constraints, evaluation stages. |
| `submission_metadata_template.yaml` | Template for the metadata you'll provide alongside your submission. |
| `candidate_schema.json` | JSON Schema describing every field in a candidate record. |
| `redrob_signals_doc.md` | Reference for the 23 behavioral signals in each candidate's `redrob_signals` object. |
| `sample_submission.csv` | A format reference. Not a high-quality ranking — just an example of the CSV structure your submission should match. |
| `validate_submission.py` | Format validator. Run this on your submission before uploading. |
| `extract_docs.py` | A utility script for document extraction. |
| `job_description.docx` | Sample job description in DOCX format. |
| `job_description.txt` | Sample job description in plain text format. |
| `redrob_signals_doc.docx` | Reference documentation for Redrob signals in DOCX format. |
| `redrob_signals_doc.txt` | Reference documentation for Redrob signals in plain text format. |
| `submission_spec.docx` | Submission specification in DOCX format. |
| `submission_spec.txt` | Submission specification in plain text format. |

---

## Getting Started

### 1. Read the Docs (~30 minutes)
In this order:
1. `job_description.md` — understand what role you're ranking candidates for
2. `submission_spec.md` — understand the rules and evaluation pipeline
3. `redrob_signals_doc.md` — understand the trap candidates and signal envelopes
4. `candidate_schema.json` — understand the candidate data structure
5. Open `sample_candidates.json` and skim a few candidates to see what real data looks like

### 2. Unpack the Candidate Pool
```bash
gunzip -k candidates.jsonl.gz   # -k keeps the .gz; you get both files
wc -l candidates.jsonl          # should print 100000
```

Or load the gzipped file directly in Python:
```python
import gzip, json
with gzip.open("candidates.jsonl.gz", "rt") as f:
    candidates = [json.loads(line) for line in f if line.strip()]
print(len(candidates))  # 100000
```

### 3. Build Your Ranker
Your job: produce a CSV with the top 100 candidates for the JD, ranked best-fit first, with a 1-2 sentence reasoning for each.

The format is described in `submission_spec.md` Section 2-3. The compute constraints are in Section 3 (5 min, 16 GB, CPU only, no network during ranking).

### 4. Validate Before Submitting
This catches format errors before you upload. The validator handles both `.jsonl` and gzipped `.jsonl.gz` files.

### 5. Submit
Submit via the portal. You'll be asked for:
- The CSV file
- All the metadata from `submission_metadata_template.yaml` (team name, GitHub repo, sandbox link, AI tools declaration, etc. — see `submission_spec.md` Section 10 for the full list)

> [!NOTE]
> **Sandbox link is required:** A working hosted environment (HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, or Binder) where your ranker can be run on a small sample. See Section 10.5 for what counts as a valid sandbox.

---

## Key Things to Know
- **No live leaderboard:** Scores are revealed only after submissions close. There is no feedback during the competition.

### Note on Data Files
Many files like `candidates.jsonl`, `sample_candidates.json`, `job_description.docx`, `job_description.txt`, `redrob_signals_doc.docx`, `redrob_signals_doc.txt`, `sample_submission.csv`, `submission_metadata_template.yaml`, `submission_spec.docx`, `submission_spec.txt` are provided as part of the dataset and challenge specifications. They are typically located in the `data/` or `docs/` directories for better organization and clarity.

The `candidate_schema.json` defines the structure for candidate profiles and is located in `src/schemas` alongside the Pydantic models for consistency.

Utility scripts like `extract_docs.py` and `validate_submission.py` are moved to `src/utils/` to centralize helper functions.
- **Three submissions max:** Your last valid submission counts.
- **AI tools are allowed:** Declare them honestly. The evaluation is designed so that AI-assisted work where you did real engineering succeeds, while AI-only submissions fail at Stages 3-5.
- **The dataset contains traps:** Keyword stuffers, plain-language Tier 5s, behavioral twins, and ~80 honeypots with subtly impossible profiles. Submissions with honeypot rate > 10% in top 100 are disqualified. See `redrob_signals_doc.md`.
- **Interview for top teams:** You will be interviewed if you reach the top X. Be prepared to walk through your architecture and defend your design choices.

## Asking for Help
If you find a bug in the bundle (e.g., schema doesn't match data, validator rejects valid format) please report it via the official hackathon support channel.

Good luck!
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
