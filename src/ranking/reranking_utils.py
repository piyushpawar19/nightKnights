import logging
from typing import List, Dict, Any, Tuple
from functools import lru_cache
import os
from joblib import Memory

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "cache", "joblib_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)
import yaml
import os

logger = logging.getLogger(__name__)

@memory.cache # Cache normalized scores for reuse
def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalizes a score to a [0, 1] range."""
    if not (min_val <= score <= max_val):
        logger.warning(f"Score {score} is outside expected range [{min_val}, {max_val}]. Clamping.")
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))

def stable_sort_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sorts candidates stably by final_score in descending order. Uses previous_rank for tie-breaking.

    Args:
        candidates (List[Dict[str, Any]]): A list of candidate dictionaries, each expected to have
                                            `final_score` and `previous_rank` keys.

    Returns:
        List[Dict[str, Any]]: The stably sorted list of candidates with `new_rank` assigned.
    """
    # Sort by previous_rank (ascending) first for stable tie-breaking, then by final_score (descending)
    # Ensure tuple keys for lru_cache compatibility
    sorted_candidates = sorted(candidates, key=lambda x: (x.get("previous_rank", float("inf")), -x.get("final_score", -float("inf"))))
    
    # Assign new ranks
    for i, candidate in enumerate(sorted_candidates):
        candidate["new_rank"] = i + 1
    return sorted_candidates

@memory.cache # Cache ranking weights as they are unlikely to change during runtime
def load_ranking_weights(config_path: str = "nightKnights/configs/ranking.yaml") -> Dict[str, float]:
    """Loads ranking weights from a YAML configuration file."""
    try:
        # Resolve path relative to the project root
        project_root = Path(__file__).resolve().parents[2]
        full_config_path = project_root / config_path # Corrected path resolution
        
        with open(full_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        weights = config.get("reranker_weights", {})
        if not isinstance(weights, dict):
            raise TypeError("reranker_weights in config must be a dictionary.")
        return weights
    except FileNotFoundError:
        logger.error(f"Config file not found at {full_config_path}") # Use full_config_path in error message
        return {"hybrid_score_weight": 0.5, "recruiter_score_weight": 0.5} # Default weights
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file at {full_config_path}: {e}") # Use full_config_path
        return {"hybrid_score_weight": 0.5, "recruiter_score_weight": 0.5} # Default weights
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading ranking weights: {e}")
        return {"hybrid_score_weight": 0.5, "recruiter_score_weight": 0.5} # Default weights
