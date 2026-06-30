import math
from typing import List, Union

def safe_get(data: dict, keys: List[str], default: any = None) -> any:
    """Safely retrieves a value from a nested dictionary using a list of keys."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

def calculate_overlap(list1: List[str], list2: List[str]) -> float:
    """Calculates the overlap percentage between two lists."""
    if not list1 or not list2: # Handle empty lists to prevent ZeroDivisionError
        return 0.0
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1).union(list2))
    return intersection / union if union > 0 else 0.0

def calculate_percentage(part: Union[int, float], whole: Union[int, float]) -> float:
    """Calculates a percentage, handling division by zero."""
    if whole == 0:
        return 0.0
    return part / whole

def normalize_score(score: Union[int, float], max_score: Union[int, float], min_score: Union[int, float] = 0) -> float:
    """Normalizes a score to a [0, 1] range."""
    if max_score == min_score:
        return 0.0 if score <= min_score else 1.0
    normalized = (score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized)) # Clamp between 0 and 1

def cosine_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two text strings (simple bag-of-words)."""
    if not text1 or not text2: # Handle empty strings
        return 0.0

    def get_word_counts(text):
        words = text.lower().split()
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return counts

    vec1 = get_word_counts(text1)
    vec2 = get_word_counts(text2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return numerator / denominator

def safe_float(value: any, default: float = 0.0) -> float:
    """Converts a value to float, safely handling non-numeric inputs."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
