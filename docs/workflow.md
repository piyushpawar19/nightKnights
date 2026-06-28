# Pipeline Workflow

This document details the sequential execution of the candidate ranking pipeline.

```mermaid
sequenceDiagram
    autonumber
    actor Participant as User / Run Script
    participant Main as src.main
    participant Config as configs/
    participant Parser as src.jd_parser
    participant Ingest as src.ingestion
    participant Retrieve as src.retrieval
    participant Rank as src.ranking
    participant Explainer as src.explainability
    participant Validator as validate_submission.py

    Participant ->> Main: Execute pipeline
    Main ->> Config: Load retrieval, ranking, & LLM configs
    Main ->> Parser: Parse job_description.txt
    Parser -->> Main: Return target JD requirements
    Main ->> Ingest: Read candidate pool (candidates.jsonl)
    Ingest -->> Main: Return preprocessed candidates
    Main ->> Retrieve: Filter candidate pool
    Retrieve -->> Main: Return top 500 retrieved candidates
    Main ->> Rank: Score top 500 & filter honeypots
    Rank -->> Main: Return top 100 ranked candidates
    Main ->> Explainer: Generate rank justifications
    Explainer -->> Main: Return ranked list with reasonings
    Main ->> Main: Save CSV to outputs/submissions/
    Main ->> Validator: Execute format validation on output
    Validator -->> Main: Validation Report (Pass/Fail)
    Main -->> Participant: Execution summary
```

## Workflow Phases

### Phase 1: Setup & Initialization
- The pipeline starts from `src/main.py`.
- It loads runtime parameters from `configs/retrieval.yaml`, `configs/ranking.yaml`, `configs/llm.yaml`, and `configs/evaluation.yaml`.

### Phase 2: Extraction & Parsing
- The `src/jd_parser` module reads `job_description.txt` to extract required skills, experience thresholds, and key responsibilities.

### Phase 3: Filtering & Retrieval
- Due to the memory and time constraints, ranking 100,000 candidates directly with an LLM or complex scorer is unfeasible.
- The `src/retrieval` module performs a rapid pre-filter, returning the top 500 candidates.

### Phase 4: Scoring & Deduplication
- The `src/ranking` module scores the top 500 candidates based on skill match and signal relevance.
- Disqualifies candidates matching any honeypot or trap indicators as defined in `docs/scoring_logic.md`.

### Phase 5: Generating Reasonings
- For the top 100 candidates, `src/explainability` calls the configured LLM or heuristic model to write a 1-2 sentence explanation of why they match.

### Phase 6: Submission Export & Verification
- The pipeline outputs a CSV file to `outputs/submissions/`.
- It runs `validate_submission.py` programmatically to guarantee formatting issues do not block upload success.
