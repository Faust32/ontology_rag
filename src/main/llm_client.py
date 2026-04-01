import logging
from typing import List, Tuple, Dict

import requests

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Config):
        self.cfg = config

    def call(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> str:
        """
        ИСПРАВЛЕНИЕ #6: Используем API /api/chat с role-based messages
        вместо сырого /api/generate + ручной сборки промпта.

        Почему это важно:
        - Ollama сама применяет правильный шаблон модели
          (Llama-3 chat tokens, ChatML и т.д.) — не нужно хардкодить.
        - Модель чётко разделяет инструкции (system) и данные (user),
          что снижает риск prompt injection через содержимое онтологии.
        - При обновлении модели шаблон меняется автоматически.
        """
        url = f"{self.cfg.ollama_base}/api/chat"
        payload = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096,
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"[Ошибка генерации: {exc}]"

    def build_system_prompt(self, context_parts: List[Tuple[Dict, float]], lang: str = "ru") -> str:
        """
        Системный промпт на языке запроса.
        lang: "ru" или "en" — определяется автоматически в RAGApp.
        """
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
                "Если ответа нет в контексте, скажи 'Не знаю', НЕ ИСПОЛЬЗУЙ внешние знания\n\n"
                f"КОНТЕКСТ:\n{context_str}"
            )
        else:
            return (
                "You are an assistant based on ontology, facts of which you get of CONTEXT.\n"
                "Answer ONLY based on the provided context, precisely and concisely.\n"
                "Cite sources as [1], [2], etc.\n"
                "If the context lacks sufficient info, say 'I don't know' — DO NOT USE external knowledge.\n\n"
                f"CONTEXT:\n{context_str}"
            )

    @staticmethod
    def _format_context(entity: Dict, lang: str = "ru") -> str:
        from rdf_processor import RDFProcessor
        return RDFProcessor.context_to_display(entity, lang=lang)