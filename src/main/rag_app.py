import logging
import time
from typing import List, Tuple, Dict

from config import Config
from knowledge_base import KnowledgeBase
from llm_client import LLMClient
from rdf_processor import RDFProcessor

logger = logging.getLogger(__name__)


class RAGApp:
    def __init__(self):
        self.cfg = Config()
        self.kb = KnowledgeBase(self.cfg)
        self.llm = LLMClient(self.cfg)

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

        system = self.llm.build_system_prompt(results)
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
    ) -> None:
        """
        Печатает блок источников под ответом.
        Вынесено в отдельный метод, чтобы run() не превращался
        в простыню форматирования.
        """
        print("\n" + "─" * 70)
        print(f"📌 ИСТОЧНИКИ  (за {elapsed:.2f}с | статус: {status})")
        print("─" * 70)

        for i, (entity, score) in enumerate(retrieved, 1):
            # Заголовок источника
            print(f"\n[{i}] {entity['label']}  (score: {score:.3f})")

            # Тип(ы) сущности
            if entity["types"]:
                print(f"    Тип: {', '.join(entity['types'])}")

            # До 3 свойств, по 1 значению каждого — не перегружаем экран
            shown = 0
            for prop, vals in list(entity["properties"].items())[:3]:
                if not vals:
                    continue
                # Обрезаем длинные значения для читаемости в терминале
                val_short = vals[0][:80] + "…" if len(vals[0]) > 80 else vals[0]
                print(f"    • {prop}: {val_short}")
                shown += 1
                if shown >= 3:
                    break

            # Входящие ссылки, если есть
            if entity.get("incoming"):
                refs = ", ".join(entity["incoming"][:2])
                print(f"    ← {refs}")

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
        print("=" * 70)
        print("Введите вопрос (или 'exit' для выхода)\n")

        while True:
            try:
                query = input("💬 Вопрос > ").strip()

                # --- Выход ---
                if query.lower() in ("exit", "quit", "q"):
                    print("👋 До свидания!")
                    break

                # --- Пустой ввод ---
                if not query:
                    continue

                # --- Поиск + генерация ---
                print("\n🔍 Поиск информации…")
                start_time = time.time()

                answer, retrieved, status = self.answer_question(query)

                elapsed = time.time() - start_time

                # --- Ответ ---
                print("\n" + "─" * 70)
                print("📝 ОТВЕТ:")
                print("─" * 70)
                print(answer)

                # --- Источники (только если есть что показать) ---
                if retrieved:
                    self._print_sources(retrieved, elapsed, status)
                else:
                    # Нет источников — показываем хотя бы время и статус
                    print(f"\n⏱️  Время: {elapsed:.2f}с | Статус: {status}")

                print("\n" + "=" * 70)

            except KeyboardInterrupt:
                # Ctrl+C во время ввода — вежливый выход
                print("\n\n👋 Прервано пользователем")
                break

            except Exception as exc:
                # Любая неожиданная ошибка — логируем с трейсбеком,
                # но не роняем весь цикл: пользователь может задать другой вопрос
                logger.error("Unhandled error: %s", exc, exc_info=True)
                print(f"\n❌ Ошибка: {exc}")