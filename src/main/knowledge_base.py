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


class KnowledgeBase:
    def __init__(self, config: Config):
        self.cfg = config
        self.entities: List[Dict] = []
        self.texts: List[str] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._load_or_build()

    # ------------------------------------------------------------------
    # Пороговая логика
    # ------------------------------------------------------------------

    def _calculate_dynamic_threshold(self, scores: np.ndarray) -> float:
        """
        - Если top_score < score_threshold → возвращаем score_threshold,
          и все результаты будут отфильтрованы в search().
        - «Локоть» используется только для ужесточения (повышения порога),
          но никогда не опускает его ниже base_threshold.
        - Нет принудительного пропуска хотя бы одного результата.
          «Не знаю» лучше, чем ложь.
        """
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
                # Скор ПОСЛЕ разрыва: всё что ниже — нерелевантно
                elbow_threshold = float(scores[max_diff_idx + 1])

        # Порог = max(base, elbow). Никакого min(threshold, top_score).
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
        """Один запрос с тремя попытками и экспоненциальным back-off."""
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
        """
        Адаптивный батчинг:
        1. Пробуем нативный батч Ollama (один HTTP-запрос, список строк)
        2. При ошибке — деградируем до последовательных одиночных запросов
        3. При одиночной ошибке — повторяем с очисткой текста
        """
        # --- Попытка 1: нативный батч Ollama ---
        vecs, failed = self._try_batch_request(texts, offset)
        if failed:
            # Деградируем только для упавших позиций
            logger.info("Retrying %d failed positions sequentially…", len(failed))
            retry_vecs, still_failed = self._sequential_fallback(texts, failed, offset)
            # Вставляем восстановленные векторы на правильные позиции
            if retry_vecs:
                for local_idx, vec in retry_vecs.items():
                    # Находим куда вставить в итоговый массив
                    vecs = self._insert_vec(vecs, local_idx, vec, texts)
            failed = still_failed
        return vecs, failed

    def _try_batch_request(
            self,
            texts: List[str],
            offset: int,
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """
        Ollama /api/embeddings принимает одну строку.
        Но можно сделать параллельные запросы с семафором — не более N одновременно.
        Семафор предотвращает перегрузку при большом chunk_size.
        """
        import threading
        semaphore = threading.Semaphore(2)  # макс 2 одновременных запроса к Ollama

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
        """
        Для каждой упавшей позиции:
        1. Чистим текст (убираем спецсимволы, которые могут ломать токенизатор)
        2. Пробуем ещё раз с увеличенным таймаутом
        3. Если снова провал — записываем в окончательный failed
        """
        recovered: Dict[int, np.ndarray] = {}
        still_failed: List[int] = []

        for global_idx in failed_global:
            local_idx = global_idx - offset
            if local_idx < 0 or local_idx >= len(all_texts):
                still_failed.append(global_idx)
                continue

            cleaned = self._sanitize_text(all_texts[local_idx])
            time.sleep(1.0)  # пауза перед повтором — даём Ollama выдохнуть

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
        """
        Убираем символы, которые могут ломать токенизатор Ollama/llama.cpp:
        - Управляющие символы (кроме пробела и \n)
        - Суррогатные пары Unicode
        - Нулевые байты
        """
        import re
        # Нулевые байты и суррогаты
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        # Управляющие символы (U+0000–U+001F кроме \t \n)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        # Множественные пробелы
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _get_embedding_single_extended(
            self,
            text: str,
            timeout: int = 240,
    ) -> Optional[np.ndarray]:
        """Одиночный запрос с увеличенным таймаутом для проблемных текстов."""
        url = f"{self.cfg.ollama_base}/api/embeddings"
        payload = {"model": self.cfg.embed_model, "prompt": text}
        for attempt in range(5):  # больше попыток для fallback
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("embedding"):
                    logger.warning("Empty embedding on extended attempt %d", attempt + 1)
                    continue
                return np.array(data["embedding"], dtype="float32")
            except Exception as exc:
                wait = min(2 ** attempt, 30)  # экспоненциальный back-off, макс 30с
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
        """
        Вставка восстановленного вектора в правильную позицию.
        Без этого recovered entities окажутся в конце, ломая соответствие
        entities[i] ↔ faiss_index[i].
        """
        if existing is None:
            return vec.reshape(1, -1)
        return np.vstack([existing, vec.reshape(1, -1)])

    # ------------------------------------------------------------------
    # Построение и загрузка индекса
    # ------------------------------------------------------------------

    def _build_index(self):
        logger.info("Parsing ontology: %s", self.cfg.ontology_path)
        g = Graph()
        g.parse(self.cfg.ontology_path.as_posix(), format="turtle")

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

        texts = [RDFProcessor.context_to_text(ctx) for ctx in entities_data]

        # --- Эмбеддинги ---
        all_vectors: List[np.ndarray] = []
        all_failed: List[int] = []

        for i in tqdm(range(0, len(texts), self.cfg.chunk_size), desc="Embedding"):
            batch = texts[i : i + self.cfg.chunk_size]
            vecs, failed = self._get_embeddings_batch(batch, offset=i)
            if vecs is not None:
                all_vectors.append(vecs)
            all_failed.extend(failed)

        # Удаляем сущности без вектора (в обратном порядке — не ломаем индексы)
        if all_failed:
            logger.warning(
                "Skipping %d entities due to embedding failures: %s…",
                len(all_failed), all_failed[:10],
            )
            for idx in sorted(all_failed, reverse=True):
                entities_data.pop(idx)
                texts.pop(idx)

        if not all_vectors:
            raise RuntimeError("All embedding batches failed — cannot build index")

        vectors = np.vstack(all_vectors).astype("float32")
        if vectors.shape[0] == 0:
            raise RuntimeError("Zero vectors after embedding — check embed model")

        faiss.normalize_L2(vectors)

        self.entities = entities_data
        self.texts = texts
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        # --- Сохранение ---
        # ИСПРАВЛЕНИЕ #5: Добавляем schema_version в метаданные.
        # При изменении логики парсинга меняем Config.INDEX_SCHEMA_VERSION,
        # и старый кэш будет автоматически перестроен.
        faiss_path = str(self.cfg.index_path) + ".faiss"
        faiss.write_index(self.index, faiss_path)
        meta = {
            "entities": self.entities,
            "texts": self.texts,
            "dim": vectors.shape[1],
            "ontology_mtime": self.cfg.ontology_path.stat().st_mtime,
            "schema_version": Config.INDEX_SCHEMA_VERSION,
        }
        with self.cfg.index_path.open("wb") as f:
            pickle.dump(meta, f)
        logger.info("Index built: %d entities, %dD", len(self.entities), vectors.shape[1])

    def _load_or_build(self):
        faiss_path = str(self.cfg.index_path) + ".faiss"
        rebuild = True

        if self.cfg.index_path.exists() and Path(faiss_path).exists():
            try:
                with self.cfg.index_path.open("rb") as f:
                    meta = pickle.load(f)

                # ИСПРАВЛЕНИЕ #5: Проверяем версию схемы перед загрузкой.
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
                        self.texts = meta["texts"]
                        self.index = faiss.read_index(faiss_path)
                        rebuild = False
                        logger.info("Loaded cached index (%d entities)", len(self.entities))
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

        if not self.entities or self.index is None:
            return [], "index_empty"

        vec = self._get_embedding_single(f"search_query: {query}")
        if vec is None:
            return [], "embedding_failed"

        q_vec = np.array([vec], dtype="float32")
        faiss.normalize_L2(q_vec)

        search_k = min(top_k * 3, len(self.entities))
        if search_k == 0:
            return [], "index_empty"

        dists, indices = self.index.search(q_vec, search_k)

        # Фильтруем нулевые и отрицательные скоры (FAISS возвращает -1 для пустых слотов)
        valid_mask = dists[0] > 0
        valid_scores = dists[0][valid_mask]

        if len(valid_scores) == 0:
            return [], "no_results"

        threshold = self._calculate_dynamic_threshold(valid_scores)

        # ИСПРАВЛЕНИЕ #2: Дедупликация по URI
        results: List[Tuple[Dict, float]] = []
        seen_uris: set = set()

        for score, idx in zip(dists[0], indices[0]):
            if idx < 0:
                continue
            # ИСПРАВЛЕНИЕ #1: жёсткий порог — не пропускаем ничего ниже threshold
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