"""
BM25 Retrieval Agent — sparse lexical retrieval for the Retrieval Engine.

Builds and queries a BM25 index over ``CandidateProfile.search_text`` only.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import yaml

from src.models.candidate_profile import CandidateProfile

logger = logging.getLogger(__name__)

StructuredJD = dict[str, Any]
RetrievalResult = dict[str, Union[str, float]]
QueryInput = Union[str, StructuredJD]

PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")

DEFAULT_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
        "they", "what", "which", "who", "whom", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "now", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "any", "if", "because",
        "until", "while", "our", "your", "their", "my", "his", "her", "its",
    }
)

INDEX_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BM25AgentError(Exception):
    """Base exception for BM25 Retrieval Agent errors."""


class IndexNotReadyError(BM25AgentError):
    """Raised when search is attempted before the index is built."""


class InvalidProfileError(BM25AgentError):
    """Raised when a CandidateProfile cannot be indexed."""


class IndexPersistenceError(BM25AgentError):
    """Raised when index save/load fails in an unrecoverable way."""


class EmptyCorpusError(BM25AgentError):
    """Raised when index construction receives no valid documents."""


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _load_retrieval_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load retrieval configuration from ``configs/retrieval.yaml`` if present."""
    path = config_path or (_project_root() / "configs" / "retrieval.yaml")
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read retrieval config at %s: %s", path, exc)
        return {}


def _resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """Resolve *path* relative to project root when not absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base or _project_root()) / candidate


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class DocumentTokenizer:
    """
    Reusable tokenizer for BM25 documents and queries.

    Supports configurable stop-word removal, stemming, and lemmatization.
    """

    def __init__(
        self,
        *,
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        use_lemmatization: bool = False,
        strategy: str = "standard",
        custom_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.strategy = strategy.lower()
        self.stopwords = set(DEFAULT_ENGLISH_STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(word.lower() for word in custom_stopwords)

        self._stemmer = self._init_stemmer() if use_stemming else None
        self._lemmatizer = self._init_lemmatizer() if use_lemmatization else None

    @classmethod
    def from_config(cls, tokenizer_cfg: dict[str, Any]) -> "DocumentTokenizer":
        """Build tokenizer from a configuration dictionary."""
        return cls(
            remove_stopwords=bool(tokenizer_cfg.get("remove_stopwords", True)),
            use_stemming=bool(tokenizer_cfg.get("use_stemming", False)),
            use_lemmatization=bool(tokenizer_cfg.get("use_lemmatization", False)),
            strategy=str(tokenizer_cfg.get("strategy", "standard")),
            custom_stopwords=tokenizer_cfg.get("custom_stopwords"),
        )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize *text* according to the configured strategy."""
        if not text or not isinstance(text, str):
            return []

        normalized = WHITESPACE_PATTERN.sub(" ", text.strip().lower())
        if self.strategy == "minimal":
            tokens = normalized.split()
            return [token for token in tokens if token]

        cleaned = PUNCTUATION_PATTERN.sub(" ", normalized)
        tokens = cleaned.split()
        filtered: list[str] = []

        for token in tokens:
            if not token:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            token = self._lemmatize(token)
            token = self._stem(token)
            if token:
                filtered.append(token)
        return filtered

    def _lemmatize(self, token: str) -> str:
        if self._lemmatizer is None:
            return token
        try:
            return self._lemmatizer.lemmatize(token)
        except Exception:
            return token

    def _stem(self, token: str) -> str:
        if self._stemmer is None:
            return token
        try:
            return self._stemmer.stem(token)
        except Exception:
            return token

    @staticmethod
    def _init_stemmer() -> Any:
        try:
            from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

            return PorterStemmer()
        except ImportError:
            logger.warning("nltk not installed; stemming disabled.")
            return None

    @staticmethod
    def _init_lemmatizer() -> Any:
        try:
            from nltk.stem import WordNetLemmatizer  # type: ignore[import-untyped]

            return WordNetLemmatizer()
        except ImportError:
            logger.warning("nltk not installed; lemmatization disabled.")
            return None


# ---------------------------------------------------------------------------
# BM25 backend abstraction
# ---------------------------------------------------------------------------


class BM25Backend(ABC):
    """Abstract BM25 scoring backend (swap ``rank_bm25`` or pure-Python impl)."""

    @abstractmethod
    def build(self, tokenized_corpus: list[list[str]]) -> None:
        """Fit the backend on a tokenized corpus."""

    @abstractmethod
    def score(self, query_tokens: list[str]) -> list[float]:
        """Return BM25 scores for every document in the corpus."""

    @abstractmethod
    def corpus_size(self) -> int:
        """Return number of indexed documents."""

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        """Serialize backend-specific state."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> "BM25Backend":
        """Restore backend from serialized state."""


class RankBM25Backend(BM25Backend):
    """Backend powered by the ``rank_bm25`` library."""

    name = "rank_bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._model: Any = None

    def build(self, tokenized_corpus: list[list[str]]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        self._model = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def score(self, query_tokens: list[str]) -> list[float]:
        if self._model is None:
            return []
        scores = self._model.get_scores(query_tokens)
        return [float(value) for value in scores]

    def corpus_size(self) -> int:
        if self._model is None:
            return 0
        return len(self._model.doc_freqs)

    def export_state(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "k1": self.k1,
            "b": self.b,
            "model": self._model,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RankBM25Backend":
        backend = cls(k1=float(state.get("k1", 1.5)), b=float(state.get("b", 0.75)))
        backend._model = state.get("model")
        return backend


class PurePythonBM25Backend(BM25Backend):
    """Fallback Okapi BM25 implementation when ``rank_bm25`` is unavailable."""

    name = "pure_python"

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._corpus: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    def build(self, tokenized_corpus: list[list[str]]) -> None:
        self._corpus = tokenized_corpus
        self._doc_lens = [len(doc) for doc in tokenized_corpus]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0

        self._doc_freqs = Counter()
        for document in tokenized_corpus:
            self._doc_freqs.update(set(document))

        num_docs = len(tokenized_corpus)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            self._idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        avg_idf = sum(self._idf.values()) / len(self._idf) if self._idf else 0.0
        eps_floor = self.epsilon * avg_idf
        for term in self._idf:
            if self._idf[term] < 0:
                self._idf[term] = eps_floor

    def score(self, query_tokens: list[str]) -> list[float]:
        if not self._corpus:
            return []

        scores = [0.0] * len(self._corpus)
        for term in query_tokens:
            if term not in self._idf:
                continue
            idf = self._idf[term]
            for doc_idx, document in enumerate(self._corpus):
                term_freq = document.count(term)
                if term_freq == 0:
                    continue
                doc_len = self._doc_lens[doc_idx]
                numerator = term_freq * (self.k1 + 1.0)
                denominator = term_freq + self.k1 * (
                    1.0 - self.b + self.b * doc_len / self._avgdl
                )
                scores[doc_idx] += idf * numerator / denominator
        return scores

    def corpus_size(self) -> int:
        return len(self._corpus)

    def export_state(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "corpus": self._corpus,
            "doc_freqs": dict(self._doc_freqs),
            "doc_lens": self._doc_lens,
            "avgdl": self._avgdl,
            "idf": self._idf,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PurePythonBM25Backend":
        backend = cls(
            k1=float(state.get("k1", 1.5)),
            b=float(state.get("b", 0.75)),
            epsilon=float(state.get("epsilon", 0.25)),
        )
        backend._corpus = state.get("corpus", [])
        backend._doc_freqs = Counter(state.get("doc_freqs", {}))
        backend._doc_lens = state.get("doc_lens", [])
        backend._avgdl = float(state.get("avgdl", 0.0))
        backend._idf = state.get("idf", {})
        return backend


def _create_backend(k1: float, b: float, backend_name: Optional[str] = None) -> BM25Backend:
    """Instantiate the preferred BM25 backend, falling back when needed."""
    preferred = (backend_name or "rank_bm25").lower()
    if preferred == "rank_bm25":
        try:
            import rank_bm25  # noqa: F401

            return RankBM25Backend(k1=k1, b=b)
        except ImportError:
            logger.warning("rank_bm25 not installed; using pure-Python BM25 backend.")
    return PurePythonBM25Backend(k1=k1, b=b)


def _restore_backend(state: dict[str, Any]) -> BM25Backend:
    backend_name = state.get("backend", RankBM25Backend.name)
    if backend_name == PurePythonBM25Backend.name:
        return PurePythonBM25Backend.from_state(state)
    return RankBM25Backend.from_state(state)


# ---------------------------------------------------------------------------
# BM25 Retrieval Agent
# ---------------------------------------------------------------------------


class BM25RetrievalAgent:
    """
    Sparse BM25 retrieval over ``CandidateProfile.search_text``.

    Indexes only the consolidated search text field produced by
    ``CandidateProfileBuilder``. Does not perform dense retrieval, ranking,
    or profile construction.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 2000,
        index_path: str | Path | None = None,
        backend: Optional[str] = None,
        tokenizer: Optional[DocumentTokenizer] = None,
        config_path: str | Path | None = None,
    ) -> None:
        config = _load_retrieval_config(
            Path(config_path) if config_path is not None else None
        )
        lexical_cfg = config.get("lexical_search", {}) if isinstance(config, dict) else {}
        bm25_params = lexical_cfg.get("bm25_params", {}) if isinstance(lexical_cfg, dict) else {}
        tokenizer_cfg = lexical_cfg.get("tokenizer", {}) if isinstance(lexical_cfg, dict) else {}

        self.k1 = float(bm25_params.get("k1", k1))
        self.b = float(bm25_params.get("b", b))
        self.top_k = int(lexical_cfg.get("top_k", top_k))
        self.backend_name = backend or lexical_cfg.get("backend", "rank_bm25")

        default_index = _project_root() / "data" / "cache" / "bm25_index"
        configured_index = lexical_cfg.get("index_path", index_path or default_index)
        self.index_path = _resolve_path(configured_index)

        self._tokenizer = tokenizer or DocumentTokenizer.from_config(tokenizer_cfg)
        self._backend: Optional[BM25Backend] = None
        self._candidate_ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._tokenized_corpus: list[list[str]] = []

        logger.debug(
            "BM25RetrievalAgent initialized (k1=%.2f, b=%.2f, top_k=%d, backend=%s)",
            self.k1,
            self.b,
            self.top_k,
            self.backend_name,
        )

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "BM25RetrievalAgent":
        """Construct an agent using defaults from ``configs/retrieval.yaml``."""
        return cls(config_path=config_path)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(self, profiles: Iterable[CandidateProfile]) -> int:
        """
        Build a fresh BM25 index from candidate profiles.

        Returns:
            Number of documents indexed.
        """
        start = time.perf_counter()
        self.clear_index()

        candidate_ids, tokenized_corpus = self._build_corpus(profiles)
        if not candidate_ids:
            raise EmptyCorpusError("No valid candidate profiles with search_text to index.")

        self._candidate_ids = candidate_ids
        self._tokenized_corpus = tokenized_corpus
        self._id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids)}

        self._backend = _create_backend(self.k1, self.b, self.backend_name)
        self._backend.build(self._tokenized_corpus)

        elapsed = time.perf_counter() - start
        logger.info(
            "BM25 index built: %d documents in %.2fs",
            len(self._candidate_ids),
            elapsed,
        )
        return len(self._candidate_ids)

    def add_documents(self, profiles: Iterable[CandidateProfile]) -> int:
        """
        Incrementally add profiles to the corpus and rebuild the BM25 index.

        Returns:
            Number of documents added.
        """
        new_ids, new_corpus = self._build_corpus(profiles)
        if not new_ids:
            logger.warning("add_documents called with no valid profiles.")
            return 0

        added = 0
        for candidate_id, tokens in zip(new_ids, new_corpus):
            if candidate_id in self._id_to_index:
                logger.warning("Updating existing candidate via add_documents: %s", candidate_id)
                self._replace_document(candidate_id, tokens)
            else:
                self._candidate_ids.append(candidate_id)
                self._tokenized_corpus.append(tokens)
                self._id_to_index[candidate_id] = len(self._candidate_ids) - 1
                added += 1

        self._rebuild_backend()
        logger.info("Added %d documents to BM25 corpus (corpus_size=%d).", added, self.corpus_size())
        return added

    def remove_documents(self, candidate_ids: Iterable[str]) -> int:
        """
        Remove candidates from the corpus and rebuild the index.

        Returns:
            Number of documents removed.
        """
        ids_to_remove = set(candidate_ids)
        if not ids_to_remove:
            return 0

        kept_ids: list[str] = []
        kept_corpus: list[list[str]] = []
        removed = 0

        for candidate_id, tokens in zip(self._candidate_ids, self._tokenized_corpus):
            if candidate_id in ids_to_remove:
                if candidate_id not in self._id_to_index:
                    logger.warning("Cannot remove missing candidate_id: %s", candidate_id)
                    continue
                removed += 1
                continue
            kept_ids.append(candidate_id)
            kept_corpus.append(tokens)

        for missing_id in ids_to_remove - set(self._id_to_index):
            logger.warning("Cannot remove missing candidate_id: %s", missing_id)

        if removed:
            self._candidate_ids, self._tokenized_corpus, self._id_to_index = self._reindex(
                kept_ids,
                kept_corpus,
            )
            self._rebuild_backend()
            logger.info("Removed %d documents from BM25 corpus.", removed)
        return removed

    def update_documents(self, profiles: Iterable[CandidateProfile]) -> int:
        """
        Update existing documents or insert new ones.

        Returns:
            Number of profiles successfully upserted.
        """
        updated = 0
        for profile in profiles:
            try:
                candidate_id, tokens = self._prepare_document(profile)
            except InvalidProfileError as exc:
                logger.warning("Skipping invalid profile during update: %s", exc)
                continue

            if candidate_id in self._id_to_index:
                self._replace_document(candidate_id, tokens)
            else:
                self._candidate_ids.append(candidate_id)
                self._tokenized_corpus.append(tokens)
                self._id_to_index[candidate_id] = len(self._candidate_ids) - 1
            updated += 1

        if updated:
            self._rebuild_backend()
            logger.info("Updated %d documents in BM25 corpus.", updated)
        return updated

    def clear_index(self) -> None:
        """Drop all indexed documents and reset internal state."""
        self._backend = None
        self._candidate_ids = []
        self._id_to_index = {}
        self._tokenized_corpus = []
        logger.info("BM25 index cleared.")

    def corpus_size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._candidate_ids)

    def is_index_ready(self) -> bool:
        """Return True when the BM25 index is built and searchable."""
        return self._backend is not None and self.corpus_size() > 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: QueryInput,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Search the BM25 index with a structured JD or raw query string.

        Args:
            query: Structured JD dict or plain-text query.
            top_k: Maximum number of results (defaults to configured top_k).

        Returns:
            List of ``{"candidate_id": str, "score": float}`` sorted by score desc.
        """
        if not self.is_index_ready():
            raise IndexNotReadyError("BM25 index is not built. Call build_index() first.")

        limit = top_k if top_k is not None else self.top_k
        query_text = self._prepare_query(query)
        query_tokens = self._tokenize(query_text)

        if not query_tokens:
            logger.warning("Empty BM25 query after tokenization; returning no results.")
            return []

        start = time.perf_counter()
        scores = self._backend.score(query_tokens) if self._backend else []
        elapsed = time.perf_counter() - start

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for idx in ranked_indices[:limit]:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append(
                {
                    "candidate_id": self._candidate_ids[idx],
                    "score": round(score, 6),
                }
            )

        logger.info(
            "BM25 search completed: query_tokens=%d, results=%d, latency=%.3fs",
            len(query_tokens),
            len(results),
            elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path | None = None) -> Path:
        """
        Persist the BM25 index to disk for reuse between runs.

        Returns:
            Path to the saved index directory.
        """
        if not self.is_index_ready() or self._backend is None:
            raise IndexNotReadyError("Cannot save an empty or uninitialized BM25 index.")

        target = _resolve_path(path) if path is not None else self.index_path
        target.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": INDEX_VERSION,
            "candidate_ids": self._candidate_ids,
            "tokenized_corpus": self._tokenized_corpus,
            "backend_state": self._backend.export_state(),
            "tokenizer": {
                "remove_stopwords": self._tokenizer.remove_stopwords,
                "use_stemming": self._tokenizer.use_stemming,
                "use_lemmatization": self._tokenizer.use_lemmatization,
                "strategy": self._tokenizer.strategy,
            },
            "k1": self.k1,
            "b": self.b,
            "top_k": self.top_k,
        }

        data_path = target / "bm25_index.pkl"
        meta_path = target / "bm25_index.meta.json"

        try:
            with data_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": INDEX_VERSION,
                        "corpus_size": self.corpus_size(),
                        "backend": self._backend.export_state().get("backend"),
                        "index_file": data_path.name,
                    },
                    handle,
                    indent=2,
                )
        except (OSError, pickle.PicklingError) as exc:
            raise IndexPersistenceError(f"Failed to save BM25 index to {target}: {exc}") from exc

        logger.info("BM25 index saved to %s (%d documents).", target, self.corpus_size())
        return target

    def load_index(self, path: str | Path | None = None) -> int:
        """
        Load a previously saved BM25 index.

        Returns:
            Number of documents loaded.
        """
        target = _resolve_path(path) if path is not None else self.index_path
        data_path = target / "bm25_index.pkl"
        meta_path = target / "bm25_index.meta.json"

        if not data_path.is_file():
            logger.debug("BM25 cache miss: index file not found at %s", data_path)
            raise IndexPersistenceError(f"BM25 index file not found: {data_path}")

        try:
            with data_path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise IndexPersistenceError(f"Corrupted BM25 index at {data_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise IndexPersistenceError("Invalid BM25 index payload format.")

        self._candidate_ids = list(payload.get("candidate_ids", []))
        self._tokenized_corpus = list(payload.get("tokenized_corpus", []))
        self._id_to_index = {cid: idx for idx, cid in enumerate(self._candidate_ids)}
        self.k1 = float(payload.get("k1", self.k1))
        self.b = float(payload.get("b", self.b))
        self.top_k = int(payload.get("top_k", self.top_k))

        tokenizer_cfg = payload.get("tokenizer", {})
        if tokenizer_cfg:
            self._tokenizer = DocumentTokenizer.from_config(tokenizer_cfg)

        backend_state = payload.get("backend_state", {})
        self._backend = _restore_backend(backend_state)

        if meta_path.is_file():
            logger.info("BM25 cache hit: loaded index from %s", target)
        else:
            logger.info("BM25 index loaded from %s (metadata file missing).", target)

        logger.info("BM25 index loaded: %d documents.", self.corpus_size())
        return self.corpus_size()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)

    def _prepare_query(self, query: QueryInput) -> str:
        """Convert structured JD input into a single retrieval query string."""
        if isinstance(query, str):
            return query.strip()

        if not isinstance(query, dict):
            logger.warning("Unsupported query type %s; treating as empty.", type(query).__name__)
            return ""

        parts: list[str] = []
        for key in ("must_have", "nice_to_have", "experience", "industry", "behavior"):
            value = query.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if text:
                        parts.append(text)
            else:
                text = str(value).strip()
                if text:
                    parts.append(text)

        return " ".join(parts)

    def _prepare_document(self, profile: CandidateProfile) -> tuple[str, list[str]]:
        """Validate a profile and tokenize its ``search_text`` field."""
        if not isinstance(profile, CandidateProfile):
            raise InvalidProfileError(
                f"Expected CandidateProfile, got {type(profile).__name__}."
            )

        candidate_id = profile.candidate_id
        if not candidate_id:
            raise InvalidProfileError("CandidateProfile is missing candidate_id.")

        search_text = profile.search_text or ""
        if not search_text.strip():
            raise InvalidProfileError(
                f"CandidateProfile {candidate_id} has empty search_text."
            )

        tokens = self._tokenize(search_text)
        if not tokens:
            raise InvalidProfileError(
                f"CandidateProfile {candidate_id} produced no tokens from search_text."
            )
        return candidate_id, tokens

    def _build_corpus(
        self,
        profiles: Iterable[CandidateProfile],
    ) -> tuple[list[str], list[list[str]]]:
        """Build parallel candidate ID and tokenized corpus lists."""
        candidate_ids: list[str] = []
        tokenized_corpus: list[list[str]] = []

        for profile in profiles:
            try:
                candidate_id, tokens = self._prepare_document(profile)
            except InvalidProfileError as exc:
                logger.warning("Skipping invalid profile: %s", exc)
                continue
            candidate_ids.append(candidate_id)
            tokenized_corpus.append(tokens)

        return candidate_ids, tokenized_corpus

    def _replace_document(self, candidate_id: str, tokens: list[str]) -> None:
        index = self._id_to_index[candidate_id]
        self._tokenized_corpus[index] = tokens

    def _rebuild_backend(self) -> None:
        if not self._tokenized_corpus:
            self._backend = None
            return
        self._backend = _create_backend(self.k1, self.b, self.backend_name)
        self._backend.build(self._tokenized_corpus)

    @staticmethod
    def _reindex(
        candidate_ids: list[str],
        tokenized_corpus: list[list[str]],
    ) -> tuple[list[str], list[list[str]], dict[str, int]]:
        id_to_index = {cid: idx for idx, cid in enumerate(candidate_ids)}
        return candidate_ids, tokenized_corpus, id_to_index


# ---------------------------------------------------------------------------
# Demonstration entry point
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Demonstrate BM25 indexing, search, and persistence."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    from src.retrieval.dataset_loader import DatasetLoader
    from src.retrieval.profile_builder_agent import CandidateProfileBuilder

    root = _project_root()
    dataset_path = root / "sample_candidates.json"
    index_dir = root / "data" / "cache" / "bm25_demo_index"

    print("\n=== BM25 Retrieval Agent Demo ===\n")

    # 1. Load candidate profiles
    loader = DatasetLoader(dataset_path, batch_size=64, cache_enabled=True)
    builder = CandidateProfileBuilder()
    records = loader.load()
    profiles = builder.build_profiles(records)
    print(f"Loaded and built {len(profiles)} candidate profiles.")

    # 2. Build BM25 index
    agent = BM25RetrievalAgent.from_config()
    indexed = agent.build_index(profiles)
    print(f"BM25 index ready with {indexed} documents.")

    # 3. Sample structured JD query
    structured_jd: StructuredJD = {
        "must_have": [
            "Python",
            "embeddings",
            "retrieval",
            "ranking",
            "machine learning",
            "vector database",
        ],
        "nice_to_have": ["LLM fine-tuning", "FAISS", "sentence-transformers"],
        "experience": "5-9 years production ML systems",
        "industry": "AI talent intelligence",
        "behavior": ["ship quickly", "evaluation frameworks", "hybrid search"],
    }

    # 4. Top-10 retrieval
    results = agent.search(structured_jd, top_k=10)
    print("\n--- Top 10 BM25 Results ---")
    for rank, hit in enumerate(results, start=1):
        print(f"  {rank:2d}. {hit['candidate_id']}  score={hit['score']:.4f}")

    # 5. Save and reload index
    saved_path = agent.save_index(index_dir)
    print(f"\nIndex saved to: {saved_path}")

    reloaded = BM25RetrievalAgent.from_config()
    reloaded.load_index(index_dir)
    print(f"Index reloaded: corpus_size={reloaded.corpus_size()}, ready={reloaded.is_index_ready()}")

    verify = reloaded.search(structured_jd, top_k=3)
    print("\n--- Verification search after reload (top 3) ---")
    for hit in verify:
        print(f"  {hit['candidate_id']}  score={hit['score']:.4f}")

    print("\n=== Demo complete ===\n")


if __name__ == "__main__":
    _demo()
