import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class EmbeddingCache:
    """Persistent SHA-256 disk and memory cache for chunk embeddings."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        self.cache_dir = Path(cache_dir or getattr(settings, "EMBEDDING_CACHE_DIR", ".cache/embeddings"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[float]] = {}
        self._rw_lock = threading.Lock()
        self._initialized = True
        logger.info(f"EmbeddingCache initialized at '{self.cache_dir}'.")

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        h = self.hash_text(text)
        with self._rw_lock:
            if h in self._memory_cache:
                return self._memory_cache[h]
            disk_file = self.cache_dir / f"{h}.json"
            if disk_file.exists():
                try:
                    vec = json.loads(disk_file.read_text(encoding="utf-8"))
                    self._memory_cache[h] = vec
                    return vec
                except Exception:
                    pass
        return None

    def set(self, text: str, vector: list[float]) -> None:
        h = self.hash_text(text)
        with self._rw_lock:
            self._memory_cache[h] = vector
            disk_file = self.cache_dir / f"{h}.json"
            try:
                disk_file.write_text(json.dumps(vector), encoding="utf-8")
            except Exception:
                pass

    def clear(self) -> None:
        with self._rw_lock:
            self._memory_cache.clear()
            for f in self.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass


class RateLimitedEmbeddings(Embeddings):
    """
    A robust wrapper around Google embeddings that handles:
    1. Two-tier SHA-256 chunk caching (reusing unchanged vectors with 0 API calls).
    2. Automatic batching into chunks (preventing burst limit triggers).
    3. Intelligent retry with exponential backoff on 429 RESOURCE_EXHAUSTED errors.
    4. Parsing the exact 'retryDelay' returned by Google's API.
    """

    def __init__(
        self,
        base_embeddings: Embeddings,
        batch_size: int = 20,
        delay_between_batches: float = 0.3,
    ):
        self.base_embeddings = base_embeddings
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.cache = EmbeddingCache()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 1. Lookup from SHA-256 cache
        for idx, text in enumerate(texts):
            cached_vector = self.cache.get(text)
            if cached_vector is not None:
                results[idx] = cached_vector
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if not uncached_texts:
            logger.info(f"All {len(texts)} chunk(s) served from EmbeddingCache (0 Google API calls).")
            return [vec for vec in results if vec is not None]

        logger.info(
            f"Embedding {len(uncached_texts)} new chunk(s) via Google Generative AI "
            f"({len(texts) - len(uncached_texts)} served from cache)..."
        )

        total_chunks = len(uncached_texts)
        for i in range(0, total_chunks, self.batch_size):
            batch = uncached_texts[i : i + self.batch_size]
            batch_indices = uncached_indices[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)..."
            )

            max_retries = 5
            success = False
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.base_embeddings.embed_documents(batch)
                    for text_item, vec_item, orig_idx in zip(batch, batch_embeddings, batch_indices):
                        self.cache.set(text_item, vec_item)
                        results[orig_idx] = vec_item

                    # Polite throttle between batches on free tier
                    if i + self.batch_size < total_chunks:
                        time.sleep(self.delay_between_batches)

                    success = True
                    break
                except Exception as e:
                    err_str = str(e)
                    err_lower = err_str.lower()

                    if (
                        "429" in err_lower
                        or "resource_exhausted" in err_lower
                        or "quota" in err_lower
                        or "rate limit" in err_lower
                    ):
                        if attempt == max_retries - 1:
                            logger.error(
                                f"Failed to embed batch {batch_num} after {max_retries} retries: {e}"
                            )
                            raise e

                        # Extract suggested wait time if available, e.g. "retry in 37.55s" or "'retryDelay': '37s'"
                        wait_time = 25.0 * (attempt + 1)
                        match = re.search(
                            r"(?:retry in|retrydelay['\":\s]+)([\d\.]+)",
                            err_str,
                            re.IGNORECASE,
                        )
                        if match:
                            try:
                                wait_time = float(match.group(1)) + 2.0
                            except ValueError:
                                pass

                        logger.warning(
                            f"Google Embedding rate limit reached (429). Waiting {wait_time:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries}) before retrying..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Non-retryable embedding error: {e}")
                        raise e

            if not success:
                raise RuntimeError(
                    f"Embedding batch {batch_num} failed after {max_retries} attempts."
                )

        return [vec for vec in results if vec is not None]

    def embed_query(self, text: str) -> list[float]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        max_retries = 4
        for attempt in range(max_retries):
            try:
                vec = self.base_embeddings.embed_query(text)
                self.cache.set(text, vec)
                return vec
            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                if (
                    "429" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                    or "rate limit" in err_lower
                ):
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to embed query after {max_retries} retries: {e}"
                        )
                        raise e
                    wait_time = 15.0 * (attempt + 1)
                    match = re.search(
                        r"(?:retry in|retrydelay['\":\s]+)([\d\.]+)",
                        err_str,
                        re.IGNORECASE,
                    )
                    if match:
                        try:
                            wait_time = float(match.group(1)) + 2.0
                        except ValueError:
                            pass
                    logger.warning(
                        f"Google Embedding rate limit on query. Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e
        raise RuntimeError("Failed to embed query after maximum retries.")


class Embedder:
    """
    Handles embedding generation using Google's embedding model
    wrapped with automatic rate-limiting and backoff protection.
    """

    def __init__(self):
        logger.info("Initializing embedding model...")

        base_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self.embedding_model = RateLimitedEmbeddings(
            base_embeddings=base_model,
            batch_size=20,
            delay_between_batches=0.35,
        )

        logger.success("Rate-limited Embedding model initialized.")

    def get_embedding_model(self):
        """
        Returns the embedding model instance.
        """
        return self.embedding_model

    def embed_query(self, query: str):
        """
        Generate embedding for a user query.
        """
        return self.embedding_model.embed_query(query)

    def embed_documents(self, texts: list[str]):
        """
        Generate embeddings for multiple text chunks.
        """
        return self.embedding_model.embed_documents(texts)
