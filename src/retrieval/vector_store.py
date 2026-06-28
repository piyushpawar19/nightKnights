import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import faiss
import yaml
import tempfile
import shutil

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages all persistent vector storage used by the Dense Retrieval Agent.
    Handles storage of embeddings, FAISS indexes, candidate IDs, and metadata.
    Does not generate embeddings or perform retrieval.
    """

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        """Initialize the VectorStoreManager with configuration."""
        self.config_path = config_path
        self._load_config()

        # Ensure directories exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_config(self) -> None:
        """Load configuration from yaml file, with fallbacks."""
        config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config {self.config_path}: {e}")

        # Extract paths with defaults
        semantic_config = config.get("semantic_search", {})
        
        # Defaults
        self.embeddings_dir = "data/embeddings"
        self.cache_dir = "data/cache"
        
        self.embeddings_file = os.path.join(self.embeddings_dir, "candidate_embeddings.npy")
        self.ids_file = os.path.join(self.embeddings_dir, "candidate_ids.pkl")
        self.metadata_file = os.path.join(self.embeddings_dir, "embedding_metadata.json")
        self.faiss_index_file = os.path.join(self.cache_dir, "faiss.index")

    # --- Private Helper Methods ---

    def _get_embedding_path(self) -> str:
        return self.embeddings_file

    def _get_ids_path(self) -> str:
        return self.ids_file

    def _get_index_path(self) -> str:
        return self.faiss_index_file

    def _get_metadata_path(self) -> str:
        return self.metadata_file

    def _atomic_write(self, filepath: str, write_func) -> None:
        """
        Safely write to a file by writing to a temporary file first,
        then renaming it to the target filepath.
        """
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(dir=dir_path)
        try:
            with os.fdopen(fd, 'wb') as temp_file:
                write_func(temp_file)
            shutil.move(temp_path, filepath)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise IOError(f"Failed to write to {filepath}: {e}")

    def _validate_embeddings(self, candidate_ids: List[str], embeddings: np.ndarray) -> None:
        """Validate that candidate IDs and embeddings dimensions match."""
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")
        if len(candidate_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(candidate_ids)} candidate IDs but {embeddings.shape[0]} embeddings."
            )

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate the metadata structure."""
        required_keys = ["embedding_dimension", "num_vectors"]
        for key in required_keys:
            if key not in metadata:
                logger.warning(f"Metadata is missing recommended key: {key}")

    # --- Public API ---

    def save_embeddings(self, candidate_ids: List[str], embeddings: np.ndarray) -> None:
        """Persist embeddings and candidate IDs to disk."""
        self._validate_embeddings(candidate_ids, embeddings)
        
        # Save embeddings (.npy)
        def write_npy(f):
            np.save(f, embeddings)
        self._atomic_write(self._get_embedding_path(), write_npy)

        # Save ids (.pkl)
        def write_pkl(f):
            pickle.dump(candidate_ids, f)
        self._atomic_write(self._get_ids_path(), write_pkl)
        
        logger.info(f"Saved {len(candidate_ids)} embeddings to {self._get_embedding_path()}")

    def load_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load candidate IDs and embeddings from disk."""
        emb_path = self._get_embedding_path()
        ids_path = self._get_ids_path()

        if not os.path.exists(emb_path) or not os.path.exists(ids_path):
            logger.info("Embeddings or IDs file missing. Returning empty sets.")
            return [], np.array([])

        try:
            embeddings = np.load(emb_path)
            with open(ids_path, "rb") as f:
                candidate_ids = pickle.load(f)
            self._validate_embeddings(candidate_ids, embeddings)
            logger.info(f"Loaded {len(candidate_ids)} embeddings.")
            return candidate_ids, embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def save_faiss_index(self, index: faiss.Index) -> None:
        """Serialize and save the FAISS index to disk."""
        index_path = self._get_index_path()
        try:
            faiss.write_index(index, index_path)
            logger.info(f"Saved FAISS index with {index.ntotal} vectors to {index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def load_faiss_index(self) -> Optional[faiss.Index]:
        """Load the FAISS index from disk."""
        index_path = self._get_index_path()
        if not os.path.exists(index_path):
            logger.info("FAISS index not found.")
            return None
        try:
            index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors.")
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Persist embedding metadata."""
        self._validate_metadata(metadata)
        def write_json(f):
            # tempfile gives binary mode, so we encode string to bytes
            f.write(json.dumps(metadata, indent=4).encode('utf-8'))
        
        self._atomic_write(self._get_metadata_path(), write_json)
        logger.info(f"Saved metadata to {self._get_metadata_path()}")

    def load_metadata(self) -> Dict[str, Any]:
        """Load embedding metadata."""
        meta_path = self._get_metadata_path()
        if not os.path.exists(meta_path):
            logger.info("Metadata not found.")
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info("Loaded metadata.")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}

    def add_embeddings(self, candidate_ids: List[str], embeddings: np.ndarray) -> None:
        """Append new embeddings to existing storage."""
        if len(candidate_ids) == 0:
            return
            
        existing_ids, existing_embs = self.load_embeddings()
        
        if len(existing_ids) == 0:
            self.save_embeddings(candidate_ids, embeddings)
            return

        # Check dimensions
        if existing_embs.shape[1] != embeddings.shape[1]:
            raise ValueError(f"Dimension mismatch: existing {existing_embs.shape[1]}, new {embeddings.shape[1]}")

        # Ensure no duplicates being added blindly (optional depending on system policy, but good for safety)
        new_ids_set = set(candidate_ids)
        existing_ids_set = set(existing_ids)
        duplicates = new_ids_set.intersection(existing_ids_set)
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate IDs in add_embeddings. They will be appended anyway. Consider update_embeddings.")

        combined_ids = existing_ids + candidate_ids
        combined_embs = np.vstack([existing_embs, embeddings])
        
        self.save_embeddings(combined_ids, combined_embs)
        logger.info(f"Added {len(candidate_ids)} embeddings. Total is now {len(combined_ids)}.")

    def update_embeddings(self, candidate_ids: List[str], embeddings: np.ndarray) -> None:
        """Update existing embeddings or add if missing."""
        self._validate_embeddings(candidate_ids, embeddings)
        if len(candidate_ids) == 0:
            return

        existing_ids, existing_embs = self.load_embeddings()
        if len(existing_ids) == 0:
            self.save_embeddings(candidate_ids, embeddings)
            return

        if existing_embs.shape[1] != embeddings.shape[1]:
            raise ValueError(f"Dimension mismatch: existing {existing_embs.shape[1]}, new {embeddings.shape[1]}")

        existing_id_to_idx = {cid: idx for idx, cid in enumerate(existing_ids)}
        
        ids_to_add = []
        embs_to_add = []
        
        for i, cid in enumerate(candidate_ids):
            if cid in existing_id_to_idx:
                idx = existing_id_to_idx[cid]
                existing_embs[idx] = embeddings[i]
            else:
                ids_to_add.append(cid)
                embs_to_add.append(embeddings[i])
                
        if ids_to_add:
            existing_ids.extend(ids_to_add)
            existing_embs = np.vstack([existing_embs, np.array(embs_to_add)])

        self.save_embeddings(existing_ids, existing_embs)
        logger.info(f"Updated embeddings. New total: {len(existing_ids)}.")

    def delete_embeddings(self, candidate_ids: List[str]) -> None:
        """Remove specific embeddings by candidate ID."""
        if not candidate_ids:
            return
            
        existing_ids, existing_embs = self.load_embeddings()
        if len(existing_ids) == 0:
            return

        ids_to_remove = set(candidate_ids)
        keep_indices = [i for i, cid in enumerate(existing_ids) if cid not in ids_to_remove]
        
        if len(keep_indices) == len(existing_ids):
            logger.info("No embeddings found to delete.")
            return

        new_ids = [existing_ids[i] for i in keep_indices]
        new_embs = existing_embs[keep_indices]
        
        self.save_embeddings(new_ids, new_embs)
        logger.info(f"Deleted {len(existing_ids) - len(new_ids)} embeddings.")

    def get_embedding(self, candidate_id: str) -> Optional[np.ndarray]:
        """Retrieve a single embedding by candidate ID."""
        existing_ids, existing_embs = self.load_embeddings()
        try:
            idx = existing_ids.index(candidate_id)
            return existing_embs[idx]
        except ValueError:
            return None

    def get_candidate_mapping(self) -> Dict[int, str]:
        """Return a mapping from vector index to candidate ID."""
        ids_path = self._get_ids_path()
        if not os.path.exists(ids_path):
            return {}
        try:
            with open(ids_path, "rb") as f:
                candidate_ids = pickle.load(f)
            return {i: cid for i, cid in enumerate(candidate_ids)}
        except Exception as e:
            logger.error(f"Failed to load candidate mapping: {e}")
            return {}

    def clear_store(self) -> None:
        """Clear all embeddings, index, and metadata."""
        paths_to_remove = [
            self._get_embedding_path(),
            self._get_ids_path(),
            self._get_index_path(),
            self._get_metadata_path()
        ]
        cleared_any = False
        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)
                cleared_any = True
        
        if cleared_any:
            logger.info("Vector store cleared.")
        else:
            logger.info("Vector store is already empty.")

    def store_exists(self) -> bool:
        """Check if vector store files exist."""
        return os.path.exists(self._get_embedding_path()) and os.path.exists(self._get_ids_path())

    def validate_store(self) -> bool:
        """Validate integrity of the vector store."""
        try:
            if not self.store_exists():
                logger.info("Store does not exist.")
                return False
                
            ids, embs = self.load_embeddings()
            if len(ids) != embs.shape[0]:
                logger.error("Store validation failed: ID count vs Embedding count mismatch.")
                return False
                
            meta = self.load_metadata()
            if meta and meta.get("num_vectors") != len(ids):
                logger.warning("Metadata num_vectors does not match actual count.")
                
            idx_path = self._get_index_path()
            if os.path.exists(idx_path):
                index = faiss.read_index(idx_path)
                if index.ntotal != len(ids):
                    logger.warning("FAISS index count does not match embedding count.")
                    
            logger.info("Vector store validated successfully.")
            return True
        except Exception as e:
            logger.error(f"Store validation failed with error: {e}")
            return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("--- Testing VectorStoreManager ---")
    store = VectorStoreManager()
    store.clear_store()
    
    # 1. Create dummy embeddings
    dim = 384
    num_candidates = 5
    dummy_ids = [f"candidate_{i}" for i in range(num_candidates)]
    dummy_embeddings = np.random.rand(num_candidates, dim).astype('float32')
    
    # 2. Save embeddings
    print("\n[Save Embeddings]")
    store.save_embeddings(dummy_ids, dummy_embeddings)
    
    # 3. Save a FAISS index
    print("\n[Save FAISS Index]")
    index = faiss.IndexFlatL2(dim)
    index.add(dummy_embeddings)
    store.save_faiss_index(index)
    
    # 4. Save metadata
    print("\n[Save Metadata]")
    meta = {
        "embedding_model": "test-model",
        "embedding_dimension": dim,
        "num_vectors": num_candidates
    }
    store.save_metadata(meta)
    
    # 5. Load everything and validate
    print("\n[Load & Validate]")
    loaded_ids, loaded_embs = store.load_embeddings()
    loaded_index = store.load_faiss_index()
    loaded_meta = store.load_metadata()
    store.validate_store()
    
    # 6. Retrieve single embedding
    print("\n[Get Single Embedding]")
    c0_emb = store.get_embedding("candidate_0")
    print(f"candidate_0 embedding shape: {c0_emb.shape if c0_emb is not None else 'Not Found'}")
    
    # 7. Update embeddings
    print("\n[Update Embeddings]")
    new_ids = ["candidate_0", "candidate_5"]
    new_embs = np.random.rand(2, dim).astype('float32')
    store.update_embeddings(new_ids, new_embs)
    store.validate_store()
    
    # 8. Clear store
    print("\n[Clear Store]")
    store.clear_store()
    print("Store exists:", store.store_exists())
