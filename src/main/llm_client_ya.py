import logging
import os
from typing import List, Tuple, Dict

import openai

from config import Config

logger = logging.getLogger(__name__)

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER", "b1g1gk23e33479u6qi0c")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "")
YANDEX_CLOUD_MODEL = os.getenv("YANDEX_CLOUD_MODEL", "qwen3.6-35b-a3b/latest")


class LLMClientYa:
    def __init__(self, config: Config):
        self.cfg = config
        api_key = YANDEX_CLOUD_API_KEY
        if not api_key:
            raise ValueError("YANDEX_CLOUD_API_KEY env var is not set")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://ai.api.cloud.yandex.net/v1",
            project=YANDEX_CLOUD_FOLDER,
        )
        self._model = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}"

    def call(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> str:
        try:
            response = self._client.responses.create(
                model=self._model,
                temperature=temperature,
                instructions=system,
                input=user,
                max_output_tokens=500,
            )
            return response.output_text.strip()
        except Exception as exc:
            logger.error("Yandex Cloud LLM call failed: %s", exc)
            return f"[Ошибка генерации: {exc}]"

    def build_system_prompt(self, context_parts: List[Tuple[Dict, float]], lang: str = "ru") -> str:
        context_str = "\n\n".join(
            f"[{i}] (score: {score:.3f})\n{self._format_context(entity, lang)}"
            for i, (entity, score) in enumerate(context_parts, 1)
        )
        if lang == "ru":
            return (
                "Ты — ассистент по онтологии, факты из которой приходят тебе в КОНТЕКСТЕ.\n"
                "Отвечай ТОЛЬКО на основе предоставленного контекста, кратко и точно.\n"
                "Отвечай СТРОГО НА РУССКОМ ЯЗЫКЕ, даже если контекст содержит английский текст.\n"
                "Ссылайся на источники как [1], [2] и т.д.\n"
                "НЕ ВКЛЮЧАЙ в ответ технические метрики, такие как score, confidence или рейтинги из контекста.\n"
                "Если ответа нет в контексте, скажи 'Не знаю', НЕ ИСПОЛЬЗУЙ внешние знания\n\n"
                f"КОНТЕКСТ:\n{context_str}"
            )
        else:
            return (
                "You are an assistant based on ontology, facts of which you get of CONTEXT.\n"
                "Answer ONLY based on the provided context, precisely and concisely.\n"
                "Cite sources as [1], [2], etc.\n"
                "DO NOT include technical metrics like score, confidence, or ratings from the context in your answer.\n"
                "If the context lacks sufficient info, say 'I don't know' — DO NOT USE external knowledge.\n\n"
                f"CONTEXT:\n{context_str}"
            )

    @staticmethod
    def _format_context(entity: Dict, lang: str = "ru") -> str:
        from rdf_processor import RDFProcessor
        return RDFProcessor.context_to_display(entity, lang=lang)
