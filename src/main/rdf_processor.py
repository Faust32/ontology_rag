import logging
from itertools import islice
from typing import Dict, List

from rdflib import Graph, RDFS, RDF, OWL, URIRef, Literal, BNode

logger = logging.getLogger(__name__)

# Максимум символов в одном значении свойства.
# Nomic-embed-text поддерживает 8192 токенов (~32 000 символов),
# поэтому режем только совсем гигантские значения.
_MAX_VALUE_LEN = 1500
_MAX_INCOMING = 5
_MAX_SUPERCLASSES = 3  # Уровней вверх по иерархии классов


class RDFProcessor:
    SKIP_PREDICATES = {RDFS.label, RDF.type, OWL.sameAs}
    SKIP_TYPES = {
        "NamedIndividual", "Class", "ObjectProperty",
        "DatatypeProperty", "Thing", "Resource",
    }

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def pick_label(g: Graph, entity: URIRef, preferred_langs=("ru", "en")) -> str:
        labels = list(g.objects(entity, RDFS.label))
        by_lang: Dict[str, str] = {}
        for lit in labels:
            if isinstance(lit, Literal) and lit.language:
                by_lang[lit.language] = str(lit)
        for lang in preferred_langs:
            if lang in by_lang:
                return by_lang[lang]
        for lit in labels:
            if isinstance(lit, Literal):
                return str(lit)
        uri_str = str(entity)
        return uri_str.split("#")[-1] if "#" in uri_str else uri_str.split("/")[-1]

    @staticmethod
    def _resolve_bnode(g: Graph, node: BNode, depth: int = 0) -> List[str]:
        """
        ИСПРАВЛЕНИЕ #2: Рекурсивное извлечение значений из B-Node.
        Ограничиваем глубину рекурсии, чтобы не зациклиться
        на циклических графах (OWL Restrictions могут образовывать петли).
        Возвращаем список строковых значений, найденных внутри узла.
        """
        if depth > 2:
            return []
        values: List[str] = []
        # Пробуем стандартные предикаты значений
        for pred in (RDF.value, RDFS.label, RDFS.comment):
            for obj in g.objects(node, pred):
                if isinstance(obj, Literal):
                    values.append(str(obj).strip())
                elif isinstance(obj, URIRef):
                    values.append(RDFProcessor.pick_label(g, obj))
        # Рекурсивно обходим вложенные B-Node
        for _, obj in g.predicate_objects(node):
            if isinstance(obj, BNode):
                values.extend(RDFProcessor._resolve_bnode(g, obj, depth + 1))
        return values

    @staticmethod
    def _get_superclasses(g: Graph, type_uri: URIRef, max_levels: int) -> List[str]:
        """
        ИСПРАВЛЕНИЕ #3: Подъем по иерархии через rdfs:subClassOf.
        Позволяет передать контексту, что "Python" → "OOP Language"
        → "Programming Language", не теряя высокоуровневые категории.
        Используем итеративный BFS вместо рекурсии, чтобы гарантированно
        выйти при циклах (OWL допускает циклы через эквивалентные классы).
        """
        result: List[str] = []
        visited = {type_uri}
        queue = [type_uri]
        level = 0
        while queue and level < max_levels:
            next_queue = []
            for current in queue:
                for parent in g.objects(current, RDFS.subClassOf):
                    if not isinstance(parent, URIRef) or parent in visited:
                        continue
                    visited.add(parent)
                    label = RDFProcessor.pick_label(g, parent)
                    if label not in RDFProcessor.SKIP_TYPES:
                        result.append(label)
                    next_queue.append(parent)
            queue = next_queue
            level += 1
        return result

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------

    @staticmethod
    def extract_entity_context(g: Graph, entity: URIRef) -> Dict:
        label = RDFProcessor.pick_label(g, entity)

        # --- Типы + иерархия ---
        types: List[str] = []
        seen_types = set()
        for t in g.objects(entity, RDF.type):
            if not isinstance(t, URIRef):
                continue
            type_label = RDFProcessor.pick_label(g, t)
            if type_label in RDFProcessor.SKIP_TYPES or type_label in seen_types:
                continue
            seen_types.add(type_label)
            types.append(type_label)
            # Добавляем родительские классы, чтобы не терять иерархию
            for super_label in RDFProcessor._get_superclasses(g, t, _MAX_SUPERCLASSES):
                if super_label not in seen_types:
                    seen_types.add(super_label)
                    types.append(super_label)

        # --- Свойства ---
        properties: Dict[str, List[str]] = {}
        for p, o in g.predicate_objects(entity):
            if p in RDFProcessor.SKIP_PREDICATES:
                continue
            pred_label = RDFProcessor.pick_label(g, p)

            if isinstance(o, Literal):
                val = str(o).strip()
                if len(val) > _MAX_VALUE_LEN:
                    # Обрезаем по границе слова
                    cut = val[:_MAX_VALUE_LEN].rsplit(" ", 1)
                    val = cut[0] if len(cut) > 1 and cut[0] else val[:_MAX_VALUE_LEN]
                if val:
                    properties.setdefault(pred_label, []).append(val)

            elif isinstance(o, URIRef):
                obj_label = RDFProcessor.pick_label(g, o)
                properties.setdefault(pred_label, []).append(obj_label)

            elif isinstance(o, BNode):
                # ИСПРАВЛЕНИЕ #2: обрабатываем B-Node вместо молчаливого пропуска
                bnode_vals = RDFProcessor._resolve_bnode(g, o)
                if bnode_vals:
                    properties.setdefault(pred_label, []).extend(bnode_vals)
                else:
                    logger.debug(
                        "BNode with no resolvable values for predicate %s on %s",
                        pred_label, label,
                    )

        # --- Входящие ссылки ---
        # ИСПРАВЛЕНИЕ: сначала фильтруем служебные предикаты, потом берём срез,
        # иначе 5 слотов могут уйти на rdf:type и аннотации.
        incoming: List[str] = []
        for s, p in islice(
            (
                (s, p)
                for s, p in g.subject_predicates(entity)
                if p not in RDFProcessor.SKIP_PREDICATES
            ),
            _MAX_INCOMING,
        ):
            subj_label = RDFProcessor.pick_label(g, s)
            pred_label = RDFProcessor.pick_label(g, p)
            incoming.append(f"{subj_label} → {pred_label}")

        return {
            "uri": str(entity),
            "label": label,
            "types": types,
            "properties": properties,
            "incoming": incoming,
        }

    @staticmethod
    def context_to_text(ctx: Dict) -> str:
        parts = [f"Название: {ctx['label']}"]
        if ctx["types"]:
            parts.append(f"Тип: {', '.join(ctx['types'])}")
        for prop, values in ctx["properties"].items():
            clean_vals = [v for v in values if v]
            if not clean_vals:
                continue
            parts.append(f"{ctx['label']} {prop}: {'; '.join(clean_vals)}")
        if ctx["incoming"]:
            parts.append(f"Ссылаются: {', '.join(ctx['incoming'])}")

        text = "search_document: " + " | ".join(parts)

        # НОВОЕ: жёсткий лимит на финальную строку перед отправкой
        # ~6000 символов ≈ 1500 токенов — безопасный запас для любой конфигурации Ollama
        MAX_CHARS = 6000
        if len(text) > MAX_CHARS:
            logger.debug(
                "Text truncated for entity '%s': %d → %d chars",
                ctx['label'], len(text), MAX_CHARS,
            )
            text = text[:MAX_CHARS].rsplit(" | ", 1)[0]  # обрезаем по границе блока
        return text

    @staticmethod
    def context_to_display(ctx: Dict) -> str:
        lines = [ctx["label"]]
        if ctx["types"]:
            lines.append(f"  Тип: {', '.join(ctx['types'])}")
        for prop, values in ctx["properties"].items():
            for val in values:
                lines.append(f"  • {prop}: {val}")
        if ctx["incoming"]:
            lines.append(f"  ← Ссылаются: {', '.join(ctx['incoming'][:3])}")
        return "\n".join(lines)