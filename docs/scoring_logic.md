# Scoring Logic & Trap Candidate Filtering

This document describes how candidates are scored and how honeypot/trap candidates are detected and disqualified.

## 1. Multi-Criteria Scoring

The score for a candidate is a combination of:
- **Skill Alignment ($S_{skill}$)**: Similarity match between job description skills and candidate skills.
- **Experience Match ($S_{exp}$)**: Comparison of years of relevant experience against requested role seniority.
- **Education Alignment ($S_{edu}$)**: Match score for degrees and institution reputation signals.
- **Agentic Evaluation ($S_{llm}$)**: LLM reasoning assessment on candidate history.

$$Score = w_{skill} \cdot S_{skill} + w_{exp} \cdot S_{exp} + w_{edu} \cdot S_{edu} + w_{llm} \cdot S_{llm}$$

*Note: Default weight parameters are defined in `configs/ranking.yaml`.*

## 2. Honeypot and Trap Detection

To prevent submission disqualification, candidates matching any of the following profiles must be filtered out before the top 100 ranking list is finalized.

> [!WARNING]
> If more than 10% (10 out of 100) of the submitted candidates contain honeypot signals, the submission is **disqualified**.

### Trap Types

1. **Keyword Stuffers**:
   - Candidates listing excessive key terms repeatedly in their profiles without backing them up in work experience descriptions.
   - Checked via: `redrob_signals.keyword_stuffer_score > 0.8`.

2. **Behavioral Twins**:
   - Duplicate profiles containing slightly modified details or identical experience structures under different names.
   - Checked via: `redrob_signals.behavioral_twin_flag == True`.

3. **Tier-5 Lies**:
   - Candidates claiming high-tier experience at non-existent companies or stating degrees from non-accredited institutions.
   - Checked via: `redrob_signals.tier5_lie_flag == True`.

4. **Impossible Profiles**:
   - Profiles with logically inconsistent timelines (e.g. graduation year before birth year, overlapping full-time roles, 10+ years experience in a framework that existed for only 3 years).
   - Checked via: `redrob_signals.impossible_profile_flag == True`.
