import logging
import time
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import faiss
from rdflib import URIRef

from config import Config
from knowledge_base import KnowledgeBase, detect_lang
from llm_client import LLMClient
from rdf_processor import RDFProcessor, get_neighbors

logger = logging.getLogger(__name__)


class RAGApp:
    def __init__(self):
        self.cfg = Config()
        self.kb = KnowledgeBase(self.cfg)
        self.llm = LLMClient(self.cfg)

        # Индекс URI → позиция в self.kb.entities для быстрого поиска
        self._uri_to_idx: Dict[str, int] = {
            e["uri"]: i for i, e in enumerate(self.kb.entities)
        }

    # ------------------------------------------------------------------
    # Графовое обогащение
    # ------------------------------------------------------------------

    def _expand_with_neighbors(
        self,
        faiss_results: List[Tuple[Dict, float]],
        query_vec: Optional[np.ndarray] = None,
        max_neighbors: int = 10,
        max_depth: int = 2,
        max_per_step: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Расширяет FAISS-результаты соседями из графа.

        Порядок:
          1. Исходные FAISS-результаты (самые релевантные по эмбеддингу)
          2. Соседи, отсортированные по:
             a) глубине (depth=1 раньше depth=2)
             b) косинусной близости к запросу (если есть query_vec)

        Returns:
            Объединённый и дедуплицированный список (entity, score)
        """
        if self.kb.graph is None:
            logger.warning("Graph not available, skipping neighbor expansion")
            return faiss_results

        # Собираем URI, которые уже есть в результатах
        seen_uris: Set[str] = {e["uri"] for e, _ in faiss_results}

        # Собираем всех соседей для всех FAISS-результатов
        candidate_neighbors: List[Dict] = []  # {"uri": ..., "depth": ..., "entity": ...}

        for entity, score in faiss_results:
            uri_str = entity["uri"]
            try:
                entity_ref = URIRef(uri_str)
            except Exception:
                continue

            neighbors = get_neighbors(
                self.kb.graph,
                entity_ref,
                max_depth=max_depth,
                max_per_step=max_per_step,
            )

            for neighbor_info in neighbors:
                neighbor_uri = str(neighbor_info["uri"])
                if neighbor_uri in seen_uris:
                    continue

                # Ищем сущность в нашем индексе
                idx = self._uri_to_idx.get(neighbor_uri)
                if idx is None:
                    continue  # Нет в индексе — пропускаем

                neighbor_entity = self.kb.entities[idx]

                # Проверяем, что у соседа есть значимый контент
                if not neighbor_entity.get("properties") and not neighbor_entity.get("types"):
                    continue

                seen_uris.add(neighbor_uri)
                candidate_neighbors.append({
                    "entity": neighbor_entity,
                    "depth": neighbor_info["depth"],
                    "idx": idx,
                })

        if not candidate_neighbors:
            return faiss_results

        # --- Ранжирование соседей ---
        if query_vec is not None:
            # Переранжируем по косинусной близости к запросу
            neighbor_scores = self._score_neighbors(candidate_neighbors, query_vec)
        else:
            # Фоллбек: ранжируем только по глубине (depth=1 → score=0.5, depth=2 → score=0.3)
            neighbor_scores = []
            for cn in candidate_neighbors:
                pseudo_score = 0.5 if cn["depth"] == 1 else 0.3
                neighbor_scores.append((cn["entity"], pseudo_score))

        # Сортируем соседей: сначала по depth, потом по score внутри depth
        neighbor_scores.sort(key=lambda x: (-x[1],))

        # Ограничиваем количество соседей
        neighbor_scores = neighbor_scores[:max_neighbors]

        # Объединяем: FAISS первыми, соседи следом
        combined = list(faiss_results) + neighbor_scores

        logger.info(
            "Graph expansion: %d FAISS + %d neighbors = %d total",
            len(faiss_results), len(neighbor_scores), len(combined),
        )
        return combined

    def _score_neighbors(
        self,
        candidates: List[Dict],
        query_vec: np.ndarray,
    ) -> List[Tuple[Dict, float]]:
        """
        Вычисляет косинусную близость соседей к запросу,
        используя уже построенный FAISS-индекс.
        """
        results = []

        for cn in candidates:
            idx = cn["idx"]
            entity = cn["entity"]

            # Берём вектор из FAISS-индекса напрямую
            # Определяем, какой индекс использовать
            index = self.kb.index_ru or self.kb.index_en
            if index is None or idx >= index.ntotal:
                # Фоллбек по глубине
                pseudo_score = 0.5 if cn["depth"] == 1 else 0.3
                results.append((entity, pseudo_score))
                continue

            try:
                entity_vec = index.reconstruct(idx)
                # Косинусное сходство (векторы уже нормализованы при индексации)
                score = float(np.dot(query_vec.flatten(), entity_vec.flatten()))
                # Дисконтируем по глубине
                depth_discount = 0.9 if cn["depth"] == 1 else 0.75
                score *= depth_discount
                results.append((entity, score))
            except Exception as exc:
                logger.debug("Failed to reconstruct vector for idx %d: %s", idx, exc)
                pseudo_score = 0.5 if cn["depth"] == 1 else 0.3
                results.append((entity, pseudo_score))

        return results

    # ------------------------------------------------------------------
    # Основной метод ответа
    # ------------------------------------------------------------------

    def answer_question(
        self, query: str
    ) -> Tuple[str, List[Tuple[Dict, float]], str]:
        results, status = self.kb.search(query)

        if status == "below_threshold":
            return (
                "😕 Не нашёл достаточно релевантной информации. "
                "Попробуйте переформулировать вопрос.",
                [],
                status,
            )
        if status in ("no_results", "index_empty", "embedding_failed"):
            return f"❌ Поиск не дал результатов (статус: {status}).", [], status
        if not results:
            return "❌ Не найдено релевантной информации.", [], "empty"

        # --- Графовое обогащение ---
        query_vec = self.kb._get_embedding_single(f"search_query: {query}")
        if query_vec is not None:
            q_vec = np.array([query_vec], dtype="float32")
            faiss.normalize_L2(q_vec)
            query_vec_normalized = q_vec
        else:
            query_vec_normalized = None

        results = self._expand_with_neighbors(
            faiss_results=results,
            query_vec=query_vec_normalized,
            max_neighbors=self.cfg.top_k,  # не больше чем top_k соседей
            max_depth=2,
            max_per_step=5,
        )

        lang = detect_lang(query)
        system = self.llm.build_system_prompt(results, lang=lang)
        answer = self.llm.call(system=system, user=query)
        return answer, results, status

    # ------------------------------------------------------------------
    # Вспомогательный вывод
    # ------------------------------------------------------------------

    @staticmethod
    def _print_sources(
        retrieved: List[Tuple[Dict, float]],
        elapsed: float,
        status: str,
        lang: str = "ru",
    ) -> None:
        print("\n" + "─" * 70)
        print(f"📌 ИСТОЧНИКИ  (за {elapsed:.2f}с | статус: {status})")
        print("─" * 70)

        for i, (entity, score) in enumerate(retrieved, 1):
            label = entity.get(f"label_{lang}", entity["label"])
            types = entity.get(f"types_{lang}", entity["types"])
            props = entity.get(f"properties_{lang}", entity["properties"])

            print(f"\n[{i}] {label}  (score: {score:.3f})")

            if types:
                type_prefix = "Тип" if lang == "ru" else "Type"
                print(f"    {type_prefix}: {', '.join(types)}")

            shown = 0
            for prop, vals in list(props.items())[:3]:
                if not vals:
                    continue
                val_short = vals[0][:80] + "…" if len(vals[0]) > 80 else vals[0]
                print(f"    • {prop}: {val_short}")
                shown += 1
                if shown >= 3:
                    break

            if entity.get("incoming"):
                refs = ", ".join(entity["incoming"][:2])
                ref_prefix = "Ссылаются" if lang == "ru" else "Referenced by"
                print(f"    ← {ref_prefix}: {refs}")

    def _describe_ontology(self) -> str:
        """
        Сэмплирует сущности из индекса и просит LLM описать онтологию.
        Вызывается один раз при запуске.
        """
        import random

        entities = self.kb.entities
        if not entities:
            return "Индекс пуст."

        # Берём случайную выборку — достаточно для общего описания
        sample_size = min(60, len(entities))
        sample = random.sample(entities, sample_size)

        # Собираем компактный контекст: только label + types
        lines = []
        for e in sample:
            label = e.get("label_en") or e.get("label", "")
            types = e.get("types_en") or e.get("types", [])
            type_str = f" [{', '.join(types[:2])}]" if types else ""
            lines.append(f"- {label}{type_str}")

        context = "\n".join(lines)
        total = len(entities)

        system = (
            "You are an ontology analyst. "
            "The user will give you a sample of entities from a knowledge base. "
            "Your task: in 4-6 sentences, describe what this ontology is about, "
            "what its main topics and sections are, and what kinds of entities it contains. "
            "Be specific and concise. Do not list entities — synthesize."
        )
        user = (
            f"Here is a random sample of {sample_size} entities out of {total} total "
            f"from the knowledge base:\n\n{context}\n\n"
            "Describe this ontology briefly."
        )

        return self.llm.call(system=system, user=user)
    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("\n" + "=" * 70)
        print("🤖  Ontology RAG System")
        print("=" * 70)
        print(f"📚 Entities : {len(self.kb.entities)}")
        print(f"🧠 LLM      : {self.cfg.llm_model}")
        print(f"🔍 Embedding: {self.cfg.embed_model}")
        print(f"📊 Top-K    : {self.cfg.top_k}")
        print(f"📐 Threshold: {self.cfg.score_threshold}")
        print(f"🔗 Graph    : {'loaded' if self.kb.graph else 'not available'}")
        print("=" * 70)
        print("\n📖 Анализирую онтологию…")
        description = self._describe_ontology()
        print("\n" + "─" * 70)
        print("📖 ОБ ЭТОЙ ОНТОЛОГИИ:")
        print("─" * 70)
        print(description)
        print("─" * 70 + "\n")
        print("Введите вопрос (или 'exit' для выхода)\n")

        while True:
            try:
                query = input("💬 Вопрос > ").strip()

                if query.lower() in ("exit", "quit", "q"):
                    print("👋 До свидания!")
                    break

                if not query:
                    continue

                print("\n🔍 Поиск информации…")
                start_time = time.time()

                answer, retrieved, status = self.answer_question(query)
                lang = detect_lang(query)

                elapsed = time.time() - start_time

                print("\n" + "─" * 70)
                print("📝 ОТВЕТ:")
                print("─" * 70)
                print(answer)

                if retrieved:
                    self._print_sources(retrieved, elapsed, status, lang=lang)
                else:
                    print(f"\n⏱️  Время: {elapsed:.2f}с | Статус: {status}")

                print("\n" + "=" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 Прервано пользователем")
                break

            except Exception as exc:
                logger.error("Unhandled error: %s", exc, exc_info=True)
                print(f"\n❌ Ошибка: {exc}")