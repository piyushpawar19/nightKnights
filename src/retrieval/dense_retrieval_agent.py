import yaml
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class DenseRetrievalAgent:
    """
    Agent responsible for semantic retrieval using dense vector embeddings.
    Strictly handles in-memory generation and search. Persistence is handled by VectorStoreManager.
    """

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        self.config_path = config_path
        self._load_config()
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        
        # Mapping from faiss internal sequential integer ID to candidate_id string
        self.int_to_id: Dict[int, str] = {}
        self.id_to_int: Dict[str, int] = {}
        self._next_id = 0

    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            sem_config = config.get("semantic_search", {})
            self.model_name = sem_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            self.device = sem_config.get("device", "cpu")
            self.embedding_dim = sem_config.get("embedding_dim", 384)
            self.batch_size = sem_config.get("batch_size", 128)
            
            # Default to L2 if not specified, IP usually used for cosine similarity (with normalized vectors)
            self.faiss_index_type = sem_config.get("faiss_index_type", "IndexFlatIP")
            self.normalize_embeddings = sem_config.get("normalize_embeddings", True)
            self.default_top_k = config.get("top_k_candidates", 2000)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}. Using defaults.")
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.device = "cpu"
            self.embedding_dim = 384
            self.batch_size = 128
            self.faiss_index_type = "IndexFlatIP"
            self.normalize_embeddings = True
            self.default_top_k = 2000

    def _init_model(self):
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence_transformers is not installed.")
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def _prepare_document(self, candidate: Any) -> str:
        """Extracts the text to be embedded from a CandidateProfile."""
        if hasattr(candidate, "search_text"):
            return candidate.search_text
        elif isinstance(candidate, dict) and "search_text" in candidate:
            return candidate["search_text"]
        
        logger.warning(f"Candidate missing 'search_text' field: {candidate}")
        return ""

    def _prepare_query(self, structured_jd: Dict[str, Any]) -> str:
        """Combines structured JD fields into a semantic query."""
        parts = []
        if structured_jd.get("must_have"):
            parts.append(f"Must have: {', '.join(structured_jd['must_have'])}")
        if structured_jd.get("nice_to_have"):
            parts.append(f"Nice to have: {', '.join(structured_jd['nice_to_have'])}")
        if structured_jd.get("experience"):
            parts.append(f"Experience: {structured_jd['experience']}")
        if structured_jd.get("seniority"):
            parts.append(f"Seniority: {structured_jd['seniority']}")
        if structured_jd.get("industry"):
            parts.append(f"Industry: {structured_jd['industry']}")
        if structured_jd.get("behavioral"):
            parts.append(f"Behavioral traits: {', '.join(structured_jd['behavioral'])}")
        
        return " | ".join(parts)

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings in configurable batches."""
        self._init_model()
        all_embeddings = []
        total_batches = math.ceil(len(texts) / self.batch_size)
        
        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            logger.debug(f"Encoding batch {i+1}/{total_batches}")
            batch_emb = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            
        if not all_embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        return np.vstack(all_embeddings)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Main method to encode a list of texts and optionally normalize."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        embeddings = self._batch_encode(texts)
        if self.normalize_embeddings:
            embeddings = self._normalize_vectors(embeddings)
        return embeddings

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes vectors for Cosine Similarity (IndexFlatIP)."""
        if faiss is None:
            raise ImportError("faiss is not installed.")
        faiss.normalize_L2(vectors)
        return vectors

    def _build_faiss_index(self) -> faiss.Index:
        """Initializes an empty FAISS index wrapped in an IDMap."""
        if faiss is None:
            raise ImportError("faiss is not installed.")
            
        if self.faiss_index_type == "IndexFlatL2":
            base_index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            base_index = faiss.IndexFlatIP(self.embedding_dim)
            
        # Wrap in IDMap to allow adding custom integer IDs (needed for deletion/updating)
        index = faiss.IndexIDMap(base_index)
        logger.info(f"Built FAISS index: {self.faiss_index_type}")
        return index

    def _search_index(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Searches the FAISS index and maps internal integer IDs back to candidate IDs."""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue # FAISS returns -1 for not enough neighbors
            
            candidate_id = self.int_to_id.get(idx)
            if candidate_id is not None:
                results.append({
                    "candidate_id": candidate_id,
                    "score": float(dist)
                })
                
        return results

    # --- Public API ---

    def encode_profiles(self, profiles: List[Any]) -> Tuple[np.ndarray, List[str]]:
        """Extracts text, encodes profiles, and returns embeddings alongside their candidate IDs."""
        texts = []
        ids = []
        for p in profiles:
            c_id = getattr(p, "candidate_id", None) or (p.get("candidate_id") if isinstance(p, dict) else None)
            if c_id is None:
                continue
            texts.append(self._prepare_document(p))
            ids.append(c_id)
            
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []
            
        embeddings = self._encode_texts(texts)
        return embeddings, ids

    def build_index(self, profiles: List[Any]):
        """Builds index from scratch with provided profiles."""
        self.clear_index()
        self.add_profiles(profiles)

    def encode_query(self, structured_jd: Dict[str, Any]) -> np.ndarray:
        """Encodes a structured Job Description query into an embedding."""
        query_text = self._prepare_query(structured_jd)
        if not query_text:
            raise ValueError("Query text resulted in empty string.")
        
        emb = self._encode_texts([query_text])
        return emb

    def search(self, query: Dict[str, Any], top_k: int = 2000) -> List[Dict[str, Any]]:
        """Main search method consumed by the Hybrid Ranker."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized.")
            return []
            
        query_embedding = self.encode_query(query)
        logger.info(f"Searching for top {top_k} candidates.")
        results = self._search_index(query_embedding, top_k)
        
        # Optionally sort just to be safe, though FAISS returns sorted
        # For inner product, higher is better. For L2, lower is better.
        reverse_sort = self.faiss_index_type == "IndexFlatIP"
        results.sort(key=lambda x: x["score"], reverse=reverse_sort)
        
        return results

    def add_profiles(self, profiles: List[Any]):
        """Encodes and adds new profiles to the FAISS index."""
        if not profiles:
            return
            
        embeddings, ids = self.encode_profiles(profiles)
        if len(ids) == 0:
            return
            
        if self.index is None:
            self.index = self._build_faiss_index()
            
        int_ids = []
        for c_id in ids:
            if c_id not in self.id_to_int:
                self.id_to_int[c_id] = self._next_id
                self.int_to_id[self._next_id] = c_id
                self._next_id += 1
            int_ids.append(self.id_to_int[c_id])
            
        int_ids_arr = np.array(int_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, int_ids_arr)
        logger.info(f"Added {len(ids)} profiles. Index size now: {self.index.ntotal}")

    def remove_profiles(self, candidate_ids: List[str]):
        """Removes specific candidate IDs from the FAISS index."""
        if self.index is None or self.index.ntotal == 0:
            return
            
        int_ids_to_remove = []
        for c_id in candidate_ids:
            if c_id in self.id_to_int:
                int_ids_to_remove.append(self.id_to_int[c_id])
                
        if int_ids_to_remove:
            arr = np.array(int_ids_to_remove, dtype=np.int64)
            self.index.remove_ids(arr)
            # Cleanup mappings
            for c_id in candidate_ids:
                if c_id in self.id_to_int:
                    int_id = self.id_to_int.pop(c_id)
                    self.int_to_id.pop(int_id, None)
            logger.info(f"Removed {len(int_ids_to_remove)} profiles. Index size now: {self.index.ntotal}")

    def update_profiles(self, profiles: List[Any]):
        """Removes and re-adds profiles to update their embeddings."""
        ids_to_update = []
        for p in profiles:
            c_id = getattr(p, "candidate_id", None) or (p.get("candidate_id") if isinstance(p, dict) else None)
            if c_id:
                ids_to_update.append(c_id)
                
        if ids_to_update:
            self.remove_profiles(ids_to_update)
            self.add_profiles(profiles)

    def clear_index(self):
        """Clears the FAISS index and mappings completely."""
        self.index = None
        self.int_to_id.clear()
        self.id_to_int.clear()
        self._next_id = 0
        logger.info("Cleared FAISS index.")

    def index_size(self) -> int:
        """Returns the number of candidates currently in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def is_ready(self) -> bool:
        """Checks if the model is loaded and the index has data."""
        return self.model is not None and self.index is not None and self.index.ntotal > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mocking CandidateProfile
    class MockCandidateProfile:
        def __init__(self, c_id: str, search_text: str):
            self.candidate_id = c_id
            self.search_text = search_text
            
    profiles = [
        MockCandidateProfile("c1", "Experienced Python Developer with strong backend skills and ML knowledge."),
        MockCandidateProfile("c2", "Frontend Engineer focusing on React and Vue. UI/UX specialist."),
        MockCandidateProfile("c3", "Data Scientist with experience in PyTorch, NLP, and sentence transformers."),
        MockCandidateProfile("c4", "DevOps Engineer working with Kubernetes, Docker, and AWS infrastructure."),
        MockCandidateProfile("c5", "Junior ML Engineer passionate about AI and deep learning. Python basics.")
    ]
    
    jd = {
        "must_have": ["Python", "Machine Learning"],
        "nice_to_have": ["NLP"],
        "experience": "Mid-level",
        "seniority": "Senior",
        "industry": "Tech",
        "behavioral": ["Team player", "Fast learner"]
    }
    
    # Initialize Agent
    agent = DenseRetrievalAgent()
    
    print("\n--- Building Index ---")
    agent.build_index(profiles)
    print(f"Index size: {agent.index_size()}")
    
    print("\n--- Searching ---")
    results = agent.search(jd, top_k=3)
    
    for rank, res in enumerate(results, 1):
        print(f"{rank}. Candidate ID: {res['candidate_id']} | Score: {res['score']:.4f}")
