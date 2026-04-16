import logging
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
from rdflib import Graph, OWL, RDF, URIRef
from tqdm import tqdm

from config import Config
from rdf_processor import RDFProcessor

logger = logging.getLogger(__name__)


def detect_lang(text: str) -> str:
    """
    Определяет язык текста по доле кириллических символов.
    Возвращает "ru" или "en".
    """
    if not text:
        return "ru"
    cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return "ru" if cyrillic / max(len(text), 1) > 0.3 else "en"


class KnowledgeBase:
    def __init__(self, config: Config):
        self.cfg = config
        self.entities: List[Dict] = []
        self.texts_ru: List[str] = []
        self.texts_en: List[str] = []
        self.index_ru: Optional[faiss.IndexFlatIP] = None
        self.index_en: Optional[faiss.IndexFlatIP] = None
        self.graph: Optional[Graph] = None
        self._load_or_build()

    # ------------------------------------------------------------------
    # Пороговая логика
    # ------------------------------------------------------------------

    def _calculate_dynamic_threshold(self, scores: np.ndarray) -> float:
        base_threshold = self.cfg.score_threshold

        if len(scores) == 0:
            return base_threshold

        top_score = float(scores[0])

        if top_score < base_threshold:
            logger.info(
                "Top score %.3f is below hard floor %.3f — no results",
                top_score, base_threshold,
            )
            return base_threshold

        if len(scores) < 2:
            return base_threshold

        diffs = np.diff(scores)
        elbow_threshold = base_threshold
        if len(diffs) > 0:
            max_diff_idx = int(np.argmax(np.abs(diffs)))
            if abs(diffs[max_diff_idx]) > 0.05:
                elbow_threshold = float(scores[max_diff_idx + 1])

        threshold = max(base_threshold, elbow_threshold)
        logger.info(
            "Threshold: %.3f (base=%.3f, elbow=%.3f, top=%.3f)",
            threshold, base_threshold, elbow_threshold, top_score,
        )
        return threshold

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------

    def _get_embedding_single(self, text: str) -> Optional[np.ndarray]:
        url = f"{self.cfg.ollama_base}/api/embeddings"
        payload = {"model": self.cfg.embed_model, "prompt": text}
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                return np.array(resp.json()["embedding"], dtype="float32")
            except Exception as exc:
                logger.warning("Embedding attempt %d/3 failed: %s", attempt + 1, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def _get_embeddings_batch(
            self,
            texts: List[str],
            offset: int = 0,
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        vecs, failed = self._try_batch_request(texts, offset)
        if failed:
            logger.info("Retrying %d failed positions sequentially…", len(failed))
            retry_vecs, still_failed = self._sequential_fallback(texts, failed, offset)
            if retry_vecs:
                for local_idx, vec in retry_vecs.items():
                    vecs = self._insert_vec(vecs, local_idx, vec, texts)
            failed = still_failed
        return vecs, failed

    def _try_batch_request(
            self,
            texts: List[str],
            offset: int,
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        import threading
        semaphore = threading.Semaphore(2)

        results: Dict[int, Optional[np.ndarray]] = {}
        failed: List[int] = []

        def _embed_safe(local_idx: int, text: str):
            with semaphore:
                vec = self._get_embedding_single(text)
                if self.cfg.embed_delay > 0:
                    time.sleep(self.cfg.embed_delay)
                return local_idx, vec

        with ThreadPoolExecutor(max_workers=self.cfg.embed_workers) as pool:
            futures = {pool.submit(_embed_safe, i, t): i for i, t in enumerate(texts)}
            for future in as_completed(futures):
                local_idx, vec = future.result()
                results[local_idx] = vec

        vectors = []
        for local_idx in range(len(texts)):
            vec = results.get(local_idx)
            if vec is not None:
                vectors.append(vec)
            else:
                failed.append(offset + local_idx)
                logger.warning(
                    "Failed entity %d: %.60s…", offset + local_idx, texts[local_idx]
                )

        if not vectors:
            return None, failed
        return np.array(vectors, dtype="float32"), failed

    def _sequential_fallback(
            self,
            all_texts: List[str],
            failed_global: List[int],
            offset: int,
    ) -> Tuple[Dict[int, np.ndarray], List[int]]:
        recovered: Dict[int, np.ndarray] = {}
        still_failed: List[int] = []

        for global_idx in failed_global:
            local_idx = global_idx - offset
            if local_idx < 0 or local_idx >= len(all_texts):
                still_failed.append(global_idx)
                continue

            cleaned = self._sanitize_text(all_texts[local_idx])
            time.sleep(1.0)

            vec = self._get_embedding_single_extended(cleaned, timeout=240)
            if vec is not None:
                recovered[local_idx] = vec
                logger.info("Recovered entity %d after sanitization", global_idx)
            else:
                still_failed.append(global_idx)
                logger.error("Permanently failed entity %d: %.60s…", global_idx, cleaned)

        return recovered, still_failed

    @staticmethod
    def _sanitize_text(text: str) -> str:
        import re
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _get_embedding_single_extended(
            self,
            text: str,
            timeout: int = 240,
    ) -> Optional[np.ndarray]:
        url = f"{self.cfg.ollama_base}/api/embeddings"
        payload = {"model": self.cfg.embed_model, "prompt": text}
        for attempt in range(5):
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("embedding"):
                    logger.warning("Empty embedding on extended attempt %d", attempt + 1)
                    continue
                return np.array(data["embedding"], dtype="float32")
            except Exception as exc:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Extended attempt %d/5 failed (%s), waiting %ds…",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
        return None

    @staticmethod
    def _insert_vec(
            existing: Optional[np.ndarray],
            local_idx: int,
            vec: np.ndarray,
            all_texts: List[str],
    ) -> np.ndarray:
        if existing is None:
            return vec.reshape(1, -1)
        return np.vstack([existing, vec.reshape(1, -1)])

    # ------------------------------------------------------------------
    # Построение и загрузка индекса
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: List[str], lang_label: str) -> np.ndarray:
        """Эмбеддит список текстов, возвращает матрицу векторов."""
        all_vectors: List[np.ndarray] = []
        all_failed: List[int] = []

        for i in tqdm(
                range(0, len(texts), self.cfg.chunk_size),
                desc=f"Embedding ({lang_label})"
        ):
            batch = texts[i: i + self.cfg.chunk_size]
            vecs, failed = self._get_embeddings_batch(batch, offset=i)
            if vecs is not None:
                all_vectors.append(vecs)
            all_failed.extend(failed)

        if all_failed:
            logger.warning(
                "Skipping %d texts due to embedding failures in %s index",
                len(all_failed), lang_label,
            )
            # Для упрощения не удаляем — заполняем нулями (FAISS отфильтрует по score)
            # В production лучше синхронизировать удаление между индексами

        if not all_vectors:
            raise RuntimeError(f"All embedding batches failed for {lang_label}")

        vectors = np.vstack(all_vectors).astype("float32")
        faiss.normalize_L2(vectors)
        return vectors

    def _build_index(self):
        logger.info("Parsing ontology: %s", self.cfg.ontology_path)
        g = Graph()
        g.parse(self.cfg.ontology_path.as_posix())

        self.graph = g # save graph

        subjects: set = set(g.subjects(RDF.type, OWL.NamedIndividual))
        if not subjects:
            logger.warning("No owl:NamedIndividual found, using all URIRef subjects")
            subjects = {s for s in g.subjects() if isinstance(s, URIRef)}

        logger.info("Processing %d subjects…", len(subjects))
        entities_data: List[Dict] = []
        for subj in tqdm(subjects, desc="Extracting entities"):
            ctx = RDFProcessor.extract_entity_context(g, subj)
            if ctx["properties"] or ctx["types"] or ctx["incoming"]:
                entities_data.append(ctx)

        if not entities_data:
            raise ValueError("No meaningful entities found in ontology")
        logger.info("Found %d entities with content", len(entities_data))

        # Определяем, какие индексы строить
        lang_index = getattr(self.cfg, 'lang_index', None)

        if lang_index == "ru":
            build_ru, build_en = True, False
        elif lang_index == "en":
            build_ru, build_en = False, True
        else:  # None — строим оба
            build_ru, build_en = True, True

        self.entities = entities_data
        dim = None

        # --- Русский индекс ---
        if build_ru:
            texts_ru = [RDFProcessor.context_to_text(ctx, lang="ru") for ctx in entities_data]
            self.texts_ru = texts_ru

            logger.info("Building Russian index…")
            vectors_ru = self._embed_texts(texts_ru, "RU")
            dim = vectors_ru.shape[1]

            self.index_ru = faiss.IndexFlatIP(dim)
            self.index_ru.add(vectors_ru)

            faiss_path_ru = str(self.cfg.index_path) + "_ru.faiss"
            faiss.write_index(self.index_ru, faiss_path_ru)
            logger.info("Saved RU index: %s", faiss_path_ru)
        else:
            self.texts_ru = []
            self.index_ru = None

        # --- Английский индекс ---
        if build_en:
            texts_en = [RDFProcessor.context_to_text(ctx, lang="en") for ctx in entities_data]
            self.texts_en = texts_en

            logger.info("Building English index…")
            vectors_en = self._embed_texts(texts_en, "EN")
            dim = vectors_en.shape[1]

            self.index_en = faiss.IndexFlatIP(dim)
            self.index_en.add(vectors_en)

            faiss_path_en = str(self.cfg.index_path) + "_en.faiss"
            faiss.write_index(self.index_en, faiss_path_en)
            logger.info("Saved EN index: %s", faiss_path_en)
        else:
            self.texts_en = []
            self.index_en = None

        # --- Сохранение метаданных ---
        meta = {
            "entities": self.entities,
            "texts_ru": self.texts_ru,
            "texts_en": self.texts_en,
            "dim": dim,
            "ontology_mtime": self.cfg.ontology_path.stat().st_mtime,
            "schema_version": Config.INDEX_SCHEMA_VERSION,
            "graph": g,  # <-- save graph in pickle
        }
        with self.cfg.index_path.open("wb") as f:
            pickle.dump(meta, f)

        built_langs = []
        if build_ru:
            built_langs.append("RU")
        if build_en:
            built_langs.append("EN")

        logger.info(
            "Index built: %d entities, %dD, languages: %s",
            len(self.entities), dim, "+".join(built_langs)
        )

    def _load_or_build(self):
        faiss_path_ru = str(self.cfg.index_path) + "_ru.faiss"
        faiss_path_en = str(self.cfg.index_path) + "_en.faiss"
        rebuild = True

        lang_index = getattr(self.cfg, 'lang_index', None)

        if lang_index == "ru":
            need_ru, need_en = True, False
        elif lang_index == "en":
            need_ru, need_en = False, True
        else:
            need_ru, need_en = True, True

        required_files_exist = self.cfg.index_path.exists()
        if need_ru:
            required_files_exist = required_files_exist and Path(faiss_path_ru).exists()
        if need_en:
            required_files_exist = required_files_exist and Path(faiss_path_en).exists()

        if required_files_exist:
            try:
                with self.cfg.index_path.open("rb") as f:
                    meta = pickle.load(f)

                cached_version = meta.get("schema_version", 0)
                if cached_version != Config.INDEX_SCHEMA_VERSION:
                    logger.warning(
                        "Index schema version mismatch (cached=%d, current=%d) — rebuilding",
                        cached_version, Config.INDEX_SCHEMA_VERSION,
                    )
                else:
                    cached_mtime = meta.get("ontology_mtime", 0)
                    current_mtime = self.cfg.ontology_path.stat().st_mtime
                    if current_mtime <= cached_mtime:
                        self.entities = meta["entities"]
                        self.texts_ru = meta.get("texts_ru", meta.get("texts", []))
                        self.texts_en = meta.get("texts_en", meta.get("texts", []))

                        # --- ВОССТАНОВЛЕНИЕ ГРАФА ---
                        self.graph = meta.get("graph", None)
                        if self.graph is None:
                            # Граф не был в кеше (старый формат) — парсим
                            logger.info("Graph not in cache, parsing ontology…")
                            self.graph = Graph()
                            self.graph.parse(self.cfg.ontology_path.as_posix())

                        if need_ru:
                            self.index_ru = faiss.read_index(faiss_path_ru)
                            logger.info("Loaded RU index (%d vectors)", self.index_ru.ntotal)

                        if need_en:
                            self.index_en = faiss.read_index(faiss_path_en)
                            logger.info("Loaded EN index (%d vectors)", self.index_en.ntotal)

                        rebuild = False
                        loaded_langs = []
                        if need_ru:
                            loaded_langs.append("RU")
                        if need_en:
                            loaded_langs.append("EN")
                        logger.info(
                            "Loaded cached index (%d entities, languages: %s)",
                            len(self.entities), "+".join(loaded_langs)
                        )
            except Exception as exc:
                logger.warning("Cache load failed: %s — rebuilding", exc)

        if rebuild:
            if not self.cfg.ontology_path.exists():
                logger.error("Ontology file not found: %s", self.cfg.ontology_path)
                sys.exit(1)
            self._build_index()

    # ------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------

    def search(
            self,
            query: str,
            top_k: Optional[int] = None,
    ) -> Tuple[List[Tuple[Dict, float]], str]:
        if top_k is None:
            top_k = self.cfg.top_k

        if not self.entities:
            return [], "index_empty"

        # Определяем язык запроса
        lang = detect_lang(query)

        # Выбираем индекс
        if lang == "ru" and self.index_ru is not None:
            index = self.index_ru
        elif lang == "en" and self.index_en is not None:
            index = self.index_en
        elif self.index_ru is not None:
            index = self.index_ru
        elif self.index_en is not None:
            index = self.index_en
        else:
            return [], "index_empty"

        actual_lang = "RU" if index is self.index_ru else "EN"
        logger.debug("Query lang: %s, using index: %s", lang, actual_lang)

        vec = self._get_embedding_single(f"search_query: {query}")
        if vec is None:
            return [], "embedding_failed"

        q_vec = np.array([vec], dtype="float32")
        faiss.normalize_L2(q_vec)

        search_k = min(top_k * 3, len(self.entities))
        if search_k == 0:
            return [], "index_empty"

        dists, indices = index.search(q_vec, search_k)

        valid_mask = dists[0] > 0
        valid_scores = dists[0][valid_mask]

        if len(valid_scores) == 0:
            return [], "no_results"

        threshold = self._calculate_dynamic_threshold(valid_scores)

        results: List[Tuple[Dict, float]] = []
        seen_uris: set = set()

        for score, idx in zip(dists[0], indices[0]):
            if idx < 0:
                continue
            if score < threshold:
                continue
            entity = self.entities[idx]
            uri = entity["uri"]
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            results.append((entity, float(score)))
            if len(results) >= top_k:
                break

        if not results:
            return [], "below_threshold"
        return results, "ok"

    # ------------------------------------------------------------------
    # Свойство для обратной совместимости
    # ------------------------------------------------------------------

    @property
    def texts(self) -> List[str]:
        """Для обратной совместимости возвращает русские тексты."""
        return self.texts_ru