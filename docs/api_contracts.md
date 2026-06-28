
# API Contracts & Data Schemas

This document defines the data models and boundaries between components in the pipeline.

## 1. Candidate Record Input Schema

Each candidate record in `candidates.jsonl` follows `candidate_schema.json`:

```json
{
  "candidate_id": "CAND_0000001",
  "personal_info": {
    "full_name": "John Doe",
    "email": "johndoe@example.com"
  },
  "work_experience": [
    {
      "company": "Tech Corp",
      "role": "Software Engineer",
      "duration_months": 36,
      "description": "Built scalable web services..."
    }
  ],
  "skills": ["Python", "Docker", "SQL"],
  "education": [
    {
      "degree": "B.S. Computer Science",
      "institution": "State University",
      "graduation_year": 2021
    }
  ],
  "redrob_signals": {
    "keyword_stuffer_score": 0.05,
    "behavioral_twin_flag": false,
    "tier5_lie_flag": false,
    "impossible_profile_flag": false
  }
}
```

## 2. Ingestion to Retrieval Contract

The Retrieval component expects a generator or iterator of processed candidate records:

```python
class CandidateRecord(BaseModel):
    candidate_id: str
    skills: List[str]
    experience_months: int
    education_level: str
    combined_text: str
    redrob_signals: Dict[str, Any]
```

## 3. Retrieval to Ranking Contract

The Retrieval component returns a list of candidate summaries containing candidate IDs and retrieval scores:

```python
class RetrievedCandidate(BaseModel):
    candidate_id: str
    retrieval_score: float
    retrieval_method: str  # 'semantic', 'lexical', or 'hybrid'
```

## 4. Ranking to Explainability Contract

The Ranker outputs the ranked list of top candidates along with their computed features:

```python
class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    score: float
    features: Dict[str, float]
```

## 5. Final Submission Format

The final CSV file (`outputs/submissions/<team_id>.csv`) must match:

| Column Name  | Format / Constraint | Description |
|---|---|---|
| `candidate_id` | `CAND_XXXXXXX` (7 digits) | Unique candidate identifier |
| `rank` | `1` to `100` (inclusive) | Candidate's rank (1 is best) |
| `score` | Floating point | Score representing relevance/confidence |
| `reasoning` | 1-2 sentences | Justification for this candidate's rank |

