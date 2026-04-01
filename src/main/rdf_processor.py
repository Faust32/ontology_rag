import logging
from itertools import islice
from typing import Dict, List, Optional

from rdflib import Graph, RDFS, RDF, OWL, URIRef, Literal, BNode

logger = logging.getLogger(__name__)

_MAX_VALUE_LEN = 1500
_MAX_INCOMING = 5
_MAX_SUPERCLASSES = 3


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
        """Возвращает лучший label (предпочтительно ru, затем en)."""
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
    def pick_label_by_lang(g: Graph, entity: URIRef, lang: str) -> Optional[str]:
        """Возвращает label для конкретного языка или None."""
        for lit in g.objects(entity, RDFS.label):
            if isinstance(lit, Literal) and lit.language == lang:
                return str(lit)
        return None

    @staticmethod
    def pick_labels_both(g: Graph, entity: URIRef) -> Dict[str, str]:
        """Возвращает словарь {lang: label} для ru и en."""
        labels = list(g.objects(entity, RDFS.label))
        result: Dict[str, str] = {}
        fallback = None

        for lit in labels:
            if isinstance(lit, Literal):
                if lit.language in ("ru", "en"):
                    result[lit.language] = str(lit)
                elif fallback is None:
                    fallback = str(lit)

        # Если нет языковых меток — используем URI
        if not result:
            uri_str = str(entity)
            local = uri_str.split("#")[-1] if "#" in uri_str else uri_str.split("/")[-1]
            fallback = fallback or local
            result["ru"] = fallback
            result["en"] = fallback

        # Заполняем недостающий язык
        if "ru" not in result:
            result["ru"] = result.get("en", fallback or "")
        if "en" not in result:
            result["en"] = result.get("ru", fallback or "")

        return result

    @staticmethod
    def _resolve_bnode(g: Graph, node: BNode, depth: int = 0) -> List[str]:
        if depth > 2:
            return []
        values: List[str] = []
        for pred in (RDF.value, RDFS.label, RDFS.comment):
            for obj in g.objects(node, pred):
                if isinstance(obj, Literal):
                    values.append(str(obj).strip())
                elif isinstance(obj, URIRef):
                    values.append(RDFProcessor.pick_label(g, obj))
        for _, obj in g.predicate_objects(node):
            if isinstance(obj, BNode):
                values.extend(RDFProcessor._resolve_bnode(g, obj, depth + 1))
        return values

    @staticmethod
    def _resolve_bnode_by_lang(g: Graph, node: BNode, lang: str, depth: int = 0) -> List[str]:
        """Извлекает значения из BNode для конкретного языка."""
        if depth > 2:
            return []
        values: List[str] = []
        for pred in (RDF.value, RDFS.label, RDFS.comment):
            for obj in g.objects(node, pred):
                if isinstance(obj, Literal):
                    if obj.language == lang or obj.language is None:
                        values.append(str(obj).strip())
                elif isinstance(obj, URIRef):
                    label = RDFProcessor.pick_label_by_lang(g, obj, lang)
                    if label:
                        values.append(label)
                    else:
                        values.append(RDFProcessor.pick_label(g, obj))
        for _, obj in g.predicate_objects(node):
            if isinstance(obj, BNode):
                values.extend(RDFProcessor._resolve_bnode_by_lang(g, obj, lang, depth + 1))
        return values

    @staticmethod
    def _get_superclasses(g: Graph, type_uri: URIRef, max_levels: int) -> List[str]:
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

    @staticmethod
    def _get_superclasses_by_lang(g: Graph, type_uri: URIRef, max_levels: int, lang: str) -> List[str]:
        """Получает суперклассы с метками на конкретном языке."""
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
                    label = RDFProcessor.pick_label_by_lang(g, parent, lang)
                    if label is None:
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
        """
        Извлекает контекст сущности с раздельным хранением по языкам.

        Возвращает словарь с полями:
        - uri, label (основной, для совместимости)
        - label_ru, label_en
        - types, types_ru, types_en
        - properties, properties_ru, properties_en
        - incoming
        """
        labels = RDFProcessor.pick_labels_both(g, entity)
        label = labels.get("ru") or labels.get("en", "")

        # --- Типы + иерархия (для обоих языков) ---
        types: List[str] = []
        types_ru: List[str] = []
        types_en: List[str] = []
        seen_types = set()

        for t in g.objects(entity, RDF.type):
            if not isinstance(t, URIRef):
                continue

            type_label = RDFProcessor.pick_label(g, t)
            if type_label in RDFProcessor.SKIP_TYPES or type_label in seen_types:
                continue
            seen_types.add(type_label)
            types.append(type_label)

            # Русские метки типов
            type_label_ru = RDFProcessor.pick_label_by_lang(g, t, "ru") or type_label
            types_ru.append(type_label_ru)
            for super_label in RDFProcessor._get_superclasses_by_lang(g, t, _MAX_SUPERCLASSES, "ru"):
                if super_label not in types_ru:
                    types_ru.append(super_label)

            # Английские метки типов
            type_label_en = RDFProcessor.pick_label_by_lang(g, t, "en") or type_label
            types_en.append(type_label_en)
            for super_label in RDFProcessor._get_superclasses_by_lang(g, t, _MAX_SUPERCLASSES, "en"):
                if super_label not in types_en:
                    types_en.append(super_label)

            # Общий список (для совместимости)
            for super_label in RDFProcessor._get_superclasses(g, t, _MAX_SUPERCLASSES):
                if super_label not in seen_types:
                    seen_types.add(super_label)
                    types.append(super_label)

        # --- Свойства (раздельно по языкам) ---
        properties: Dict[str, List[str]] = {}
        properties_ru: Dict[str, List[str]] = {}
        properties_en: Dict[str, List[str]] = {}

        for p, o in g.predicate_objects(entity):
            if p in RDFProcessor.SKIP_PREDICATES:
                continue

            pred_label = RDFProcessor.pick_label(g, p)
            pred_label_ru = RDFProcessor.pick_label_by_lang(g, p, "ru") or pred_label
            pred_label_en = RDFProcessor.pick_label_by_lang(g, p, "en") or pred_label

            if isinstance(o, Literal):
                val = str(o).strip()
                if len(val) > _MAX_VALUE_LEN:
                    cut = val[:_MAX_VALUE_LEN].rsplit(" ", 1)
                    val = cut[0] if len(cut) > 1 and cut[0] else val[:_MAX_VALUE_LEN]
                if val:
                    properties.setdefault(pred_label, []).append(val)
                    # Распределяем по языкам
                    if isinstance(o, Literal) and o.language == "ru":
                        properties_ru.setdefault(pred_label_ru, []).append(val)
                    elif isinstance(o, Literal) and o.language == "en":
                        properties_en.setdefault(pred_label_en, []).append(val)
                    else:
                        # Без языка — в оба
                        properties_ru.setdefault(pred_label_ru, []).append(val)
                        properties_en.setdefault(pred_label_en, []).append(val)

            elif isinstance(o, URIRef):
                obj_labels = RDFProcessor.pick_labels_both(g, o)
                obj_label = obj_labels.get("ru") or obj_labels.get("en", "")
                properties.setdefault(pred_label, []).append(obj_label)
                properties_ru.setdefault(pred_label_ru, []).append(obj_labels.get("ru", obj_label))
                properties_en.setdefault(pred_label_en, []).append(obj_labels.get("en", obj_label))

            elif isinstance(o, BNode):
                bnode_vals = RDFProcessor._resolve_bnode(g, o)
                if bnode_vals:
                    properties.setdefault(pred_label, []).extend(bnode_vals)

                bnode_vals_ru = RDFProcessor._resolve_bnode_by_lang(g, o, "ru")
                if bnode_vals_ru:
                    properties_ru.setdefault(pred_label_ru, []).extend(bnode_vals_ru)
                elif bnode_vals:
                    properties_ru.setdefault(pred_label_ru, []).extend(bnode_vals)

                bnode_vals_en = RDFProcessor._resolve_bnode_by_lang(g, o, "en")
                if bnode_vals_en:
                    properties_en.setdefault(pred_label_en, []).extend(bnode_vals_en)
                elif bnode_vals:
                    properties_en.setdefault(pred_label_en, []).extend(bnode_vals)

        # --- Входящие ссылки ---
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
            "label_ru": labels.get("ru", label),
            "label_en": labels.get("en", label),
            "types": types,
            "types_ru": types_ru,
            "types_en": types_en,
            "properties": properties,
            "properties_ru": properties_ru,
            "properties_en": properties_en,
            "incoming": incoming,
        }

    @staticmethod
    def context_to_text(ctx: Dict, lang: str = "ru") -> str:
        """
        Генерирует текст для эмбеддинга на указанном языке.

        Args:
            ctx: Контекст сущности из extract_entity_context
            lang: "ru" или "en"
        """
        if lang == "en":
            label = ctx.get("label_en", ctx["label"])
            types = ctx.get("types_en", ctx["types"])
            props = ctx.get("properties_en", ctx["properties"])
        else:
            label = ctx.get("label_ru", ctx["label"])
            types = ctx.get("types_ru", ctx["types"])
            props = ctx.get("properties_ru", ctx["properties"])

        parts = [f"Название: {label}" if lang == "ru" else f"Name: {label}"]

        if types:
            type_prefix = "Тип" if lang == "ru" else "Type"
            parts.append(f"{type_prefix}: {', '.join(types)}")

        for prop, values in props.items():
            clean_vals = [v for v in values if v]
            if not clean_vals:
                continue
            parts.append(f"{label} {prop}: {'; '.join(clean_vals)}")

        if ctx["incoming"]:
            ref_prefix = "Ссылаются" if lang == "ru" else "Referenced by"
            parts.append(f"{ref_prefix}: {', '.join(ctx['incoming'])}")

        text = "search_document: " + " | ".join(parts)

        MAX_CHARS = 600
        if len(text) > MAX_CHARS:
            logger.debug(
                "Text truncated for entity '%s': %d → %d chars",
                label, len(text), MAX_CHARS,
            )
            text = text[:MAX_CHARS].rsplit(" | ", 1)[0]
        return text

    @staticmethod
    def context_to_display(ctx: Dict, lang: str = "ru") -> str:
        """Форматирует контекст для отображения пользователю на нужном языке."""
        label = ctx.get(f"label_{lang}", ctx["label"])
        types = ctx.get(f"types_{lang}", ctx["types"])
        props = ctx.get(f"properties_{lang}", ctx["properties"])

        lines = [label]
        if types:
            type_prefix = "Тип" if lang == "ru" else "Type"
            lines.append(f"  {type_prefix}: {', '.join(types)}")
        for prop, values in props.items():
            for val in values:
                lines.append(f"  • {prop}: {val}")
        if ctx["incoming"]:
            ref_prefix = "Ссылаются" if lang == "ru" else "Referenced by"
            lines.append(f"  ← {ref_prefix}: {', '.join(ctx['incoming'][:3])}")
        return "\n".join(lines)