"""
eval.py — Оценка качества RAG-системы на онтологии.

Метрики (адаптированы из Потапова & Лукашевич, 2025):
  Retrieval:
    - Hit Rate@k      — хотя бы один нужный URI найден в top-k
    - Precision@k     — доля правильных URI среди retrieved (≈ Context Relevance)
    - Recall@k        — доля найденных нужных URI из всех нужных (≈ Completeness)
    - MRR             — Mean Reciprocal Rank (позиция первого правильного)
  Generation:
    - Exact Match     — ожидаемый ответ присутствует в тексте ответа LLM
    - Adherence       — все слова ответа встречаются в retrieved-контексте

Использование:
    python eval.py                          # стандартный прогон (русский)
    python eval.py --language en            # только английские кейсы
    python eval.py --language ru            # только русские кейсы
    python eval.py --language both          # оба языка (ru + en)
    python eval.py --top-k 5               # кастомный k
    python eval.py --skip-generation       # только retrieval, без LLM
    python eval.py --verbose               # подробный вывод по каждому вопросу
    python eval.py --debug-failures        # детальный дебаг провалившихся кейсов
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Тестовые кейсы — РУССКИЙ ЯЗЫК
# ---------------------------------------------------------------------------

TEST_CASES_RU = [
    {
        "query": "Кто создал Python и какой у него механизм трансляции?",
        "expected_uris": ["Python"],
        "expected_answer": "интерпрет",
    },
    {
        "query": "Кто является автором языка Java?",
        "expected_uris": ["Java"],
        "expected_answer": "гослинг",
    },
    {
        "query": "Что такое Haskell?",
        "expected_uris": ["Haskell"],
        "expected_answer": "функциональн",
    },
    {
        "query": "Для каких задач используется Prolog?",
        "expected_uris": ["Prolog"],
        "expected_answer": "искусственн",
    },
    {
        "query": "Кто создал LISP и в каком году?",
        "expected_uris": ["LISP"],
        "expected_answer": "маккарти",
    },
    {
        "query": "Кто разработал язык C и когда он был впервые выпущен?",
        "expected_uris": ["C"],
        "expected_answer": "1972",
    },
    {
        "query": "Кто создал язык Perl?",
        "expected_uris": ["Perl"],
        "expected_answer": "ларри уолл",
    },
    {
        "query": "Что такое BASIC и для кого он создавался?",
        "expected_uris": ["BASIC"],
        "expected_answer": "начинающ",
    },
    {
        "query": "Что такое SQL?",
        "expected_uris": ["SQL"],
        "expected_answer": "реляционн",
    },
    {
        "query": "Какие подъязыки входят в состав SQL?",
        "expected_uris": ["SQL", "DDL", "DML", "DCL", "TCL"],
        "expected_answer": None,
    },
    {
        "query": "Кто создал PostgreSQL?",
        "expected_uris": ["PostgreSQL"],
        "expected_answer": "стоунбрейкер",
    },
    {
        "query": "Для чего используется язык Cypher?",
        "expected_uris": ["Cypher"],
        "expected_answer": "граф",
    },
    {
        "query": "Что такое SPARQL?",
        "expected_uris": ["SPARQL"],
        "expected_answer": "rdf",
    },
    {
        "query": "Что такое JQL и для какой системы он предназначен?",
        "expected_uris": ["JQL", "Jira"],
        "expected_answer": "jira",
    },
    {
        "query": "Кто является автором формата JSON?",
        "expected_uris": ["JSON"],
        "expected_answer": "крокфорд",
    },
    {
        "query": "На основе каких языков создан YAML?",
        "expected_uris": ["YAML", "JSON", "XML"],
        "expected_answer": None,
    },
    {
        "query": "Что такое BSON?",
        "expected_uris": ["BSON"],
        "expected_answer": "бинарн",
    },
    {
        "query": "Для чего используется CSV?",
        "expected_uris": ["CSV"],
        "expected_answer": "таблич",
    },
    {
        "query": "Кто разработал Protocol Buffers?",
        "expected_uris": ["ProtocolBuffers"],
        "expected_answer": "google",
    },
    {
        "query": "Что такое TOML и для чего он используется?",
        "expected_uris": ["TOML"],
        "expected_answer": "конфигурац",
    },
    {
        "query": "Для каких технологий используется Avro?",
        "expected_uris": ["Avro"],
        "expected_answer": "apache",
    },
    {
        "query": "Кто создал LaTeX и для чего он используется?",
        "expected_uris": ["LaTeX"],
        "expected_answer": "лэмпорт",
    },
    {
        "query": "Что такое TeX?",
        "expected_uris": ["TeX"],
        "expected_answer": "кнут",
    },
    {
        "query": "Какая организация разработала XML?",
        "expected_uris": ["XML"],
        "expected_answer": "консорциум",
    },
    {
        "query": "Что такое HTML?",
        "expected_uris": ["HTML"],
        "expected_answer": "разметк",
    },
    {
        "query": "Что такое RDF и как он представляет данные?",
        "expected_uris": ["RDF"],
        "expected_answer": "триплет",
    },
    {
        "query": "Что такое ePub?",
        "expected_uris": ["ePub"],
        "expected_answer": "электронн",
    },
    {
        "query": "Какие типы диаграмм входят в UML?",
        "expected_uris": ["UML", "ДиаграммаКлассов", "ДиаграммаПоследовательности", "UML-диаграмма"],
        "expected_answer": None,
    },
    {
        "query": "Что такое DOT и с каким инструментом он связан?",
        "expected_uris": ["DOT", "Graphviz"],
        "expected_answer": "graphviz",
    },
    {
        "query": "Что такое функциональная парадигма программирования?",
        "expected_uris": ["ФункциональнаяПарадигмаПрограммирования"],
        "expected_answer": "функц",
    },
    {
        "query": "Что такое логическая парадигма программирования?",
        "expected_uris": ["ЛогическаяПарадигмаПрограммирования"],
        "expected_answer": "логическ",
    },
    {
        "query": "Кто создал MASM?",
        "expected_uris": ["MASM"],
        "expected_answer": "microsoft",
    },
    {
        "query": "Какие языки используют интерпретацию вместо компиляции?",
        "expected_uris": ["Python", "LISP", "Prolog", "Perl"],
        "expected_answer": None,
    },
    {
        "query": "Назови языки программирования с компиляцией кода.",
        "expected_uris": ["Java", "C", "Haskell"],
        "expected_answer": None,
    },
    {
        "query": "Какие существуют диалекты SQL?",
        "expected_uris": ["SQL", "MySQL", "PostgreSQL", "SQLite"],
        "expected_answer": None,
    },
    {
        "query": "Что такое язык программирования Rust?",
        "expected_uris": [],
        "expected_answer": None,
    },
    {
        "query": "Какая столица у Франции?",
        "expected_uris": [],
        "expected_answer": None,
    },
]

# ---------------------------------------------------------------------------
# Тестовые кейсы — АНГЛИЙСКИЙ ЯЗЫК
# ---------------------------------------------------------------------------

TEST_CASES_EN = [
    {
        "query": "Who created Python and what is its translation mechanism?",
        "expected_uris": ["Python"],
        "expected_answer": "interpret",
    },
    {
        "query": "Who is the author of the Java language?",
        "expected_uris": ["Java"],
        "expected_answer": "gosling",
    },
    {
        "query": "What is Haskell?",
        "expected_uris": ["Haskell"],
        "expected_answer": "functional",
    },
    {
        "query": "What tasks is Prolog used for?",
        "expected_uris": ["Prolog"],
        "expected_answer": "artificial",
    },
    {
        "query": "Who created LISP and in what year?",
        "expected_uris": ["LISP"],
        "expected_answer": "mccarthy",
    },
    {
        "query": "Who developed the C language and when was it first released?",
        "expected_uris": ["C"],
        "expected_answer": "1972",
    },
    {
        "query": "Who created the Perl language?",
        "expected_uris": ["Perl"],
        "expected_answer": "larry wall",
    },
    {
        "query": "What is BASIC and for whom was it created?",
        "expected_uris": ["BASIC"],
        "expected_answer": "beginner",
    },
    {
        "query": "What is SQL?",
        "expected_uris": ["SQL"],
        "expected_answer": "relational",
    },
    {
        "query": "What sublanguages are part of SQL?",
        "expected_uris": ["SQL", "DDL", "DML", "DCL", "TCL"],
        "expected_answer": None,
    },
    {
        "query": "Who created PostgreSQL?",
        "expected_uris": ["PostgreSQL"],
        "expected_answer": "stonebraker",
    },
    {
        "query": "What is the Cypher language used for?",
        "expected_uris": ["Cypher"],
        "expected_answer": "graph",
    },
    {
        "query": "What is SPARQL?",
        "expected_uris": ["SPARQL"],
        "expected_answer": "rdf",
    },
    {
        "query": "What is JQL and for which system is it designed?",
        "expected_uris": ["JQL", "Jira"],
        "expected_answer": "jira",
    },
    {
        "query": "Who is the author of the JSON format?",
        "expected_uris": ["JSON"],
        "expected_answer": "crockford",
    },
    {
        "query": "Based on which languages was YAML created?",
        "expected_uris": ["YAML", "JSON", "XML"],
        "expected_answer": None,
    },
    {
        "query": "What is BSON?",
        "expected_uris": ["BSON"],
        "expected_answer": "binary",
    },
    {
        "query": "What is CSV used for?",
        "expected_uris": ["CSV"],
        "expected_answer": "tabular",
    },
    {
        "query": "Who developed Protocol Buffers?",
        "expected_uris": ["ProtocolBuffers"],
        "expected_answer": "google",
    },
    {
        "query": "What is TOML and what is it used for?",
        "expected_uris": ["TOML"],
        "expected_answer": "configur",
    },
    {
        "query": "For which technologies is Avro used?",
        "expected_uris": ["Avro"],
        "expected_answer": "apache",
    },
    {
        "query": "Who created LaTeX and what is it used for?",
        "expected_uris": ["LaTeX"],
        "expected_answer": "lamport",
    },
    {
        "query": "What is TeX?",
        "expected_uris": ["TeX"],
        "expected_answer": "knuth",
    },
    {
        "query": "Which organization developed XML?",
        "expected_uris": ["XML"],
        "expected_answer": "consortium",
    },
    {
        "query": "What is HTML?",
        "expected_uris": ["HTML"],
        "expected_answer": "markup",
    },
    {
        "query": "What is RDF and how does it represent data?",
        "expected_uris": ["RDF"],
        "expected_answer": "triplet",
    },
    {
        "query": "What is ePub?",
        "expected_uris": ["ePub"],
        "expected_answer": "electronic",
    },
    {
        "query": "What diagram types are included in UML?",
        "expected_uris": ["UML", "ДиаграммаКлассов", "ДиаграммаПоследовательности", "UML-диаграмма"],
        "expected_answer": None,
    },
    {
        "query": "What is DOT and which tool is it associated with?",
        "expected_uris": ["DOT", "Graphviz"],
        "expected_answer": "graphviz",
    },
    {
        "query": "What is the functional programming paradigm?",
        "expected_uris": ["ФункциональнаяПарадигмаПрограммирования"],
        "expected_answer": "funct",
    },
    {
        "query": "What is the logical programming paradigm?",
        "expected_uris": ["ЛогическаяПарадигмаПрограммирования"],
        "expected_answer": "logic",
    },
    {
        "query": "Who created MASM?",
        "expected_uris": ["MASM"],
        "expected_answer": "microsoft",
    },
    {
        "query": "Which languages use interpretation instead of compilation?",
        "expected_uris": ["Python", "LISP", "Prolog", "Perl"],
        "expected_answer": None,
    },
    {
        "query": "Name programming languages with code compilation.",
        "expected_uris": ["Java", "C", "Haskell"],
        "expected_answer": None,
    },
    {
        "query": "What SQL dialects exist?",
        "expected_uris": ["SQL", "MySQL", "PostgreSQL", "SQLite"],
        "expected_answer": None,
    },
    {
        "query": "What is the Rust programming language?",
        "expected_uris": [],
        "expected_answer": None,
    },
    {
        "query": "What is the capital of France?",
        "expected_uris": [],
        "expected_answer": None,
    },
]


# ---------------------------------------------------------------------------
# Вспомогательные структуры
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    query: str
    expected_uris: List[str]
    retrieved_uris: List[str]
    retrieved_texts: List[str]
    retrieved_scores: List[float]
    status: str

    answer: Optional[str] = None
    expected_answer: Optional[str] = None

    hit: bool = False
    precision: float = 0.0
    recall: float = 0.0
    reciprocal_rank: float = 0.0

    exact_match: Optional[bool] = None
    adherence: Optional[bool] = None

    match_details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# URI-матчинг
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.lower().strip()


def _extract_local_name(uri: str) -> str:
    if '#' in uri:
        return uri.split('#')[-1]
    return uri.split('/')[-1]


def _uri_matches(uri: str, pattern: str) -> bool:
    uri_lower = _normalize(uri)
    pattern_lower = _normalize(pattern)

    if pattern_lower in uri_lower:
        return True

    local_name = _normalize(_extract_local_name(uri))
    if pattern_lower == local_name or pattern_lower in local_name:
        return True

    pattern_variants = [
        pattern_lower,
        pattern_lower.replace('_', ''),
        pattern_lower.replace(' ', ''),
        pattern_lower.replace('-', ''),
    ]
    local_variants = [
        local_name,
        local_name.replace('_', ''),
        local_name.replace(' ', ''),
        local_name.replace('-', ''),
    ]

    for pv in pattern_variants:
        for lv in local_variants:
            if pv in lv or lv in pv:
                return True

    return False


def _any_match(uri: str, patterns: List[str]) -> Tuple[bool, Optional[str]]:
    for p in patterns:
        if _uri_matches(uri, p):
            return True, p
    return False, None


# ---------------------------------------------------------------------------
# Вычисление метрик
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(result: CaseResult) -> None:
    expected = result.expected_uris
    retrieved = result.retrieved_uris

    result.match_details = {
        "matches": [],
        "unmatched_expected": list(expected),
        "unmatched_retrieved": [],
    }

    if not expected:
        result.hit = len(retrieved) == 0
        result.precision = 1.0 if len(retrieved) == 0 else 0.0
        result.recall = 1.0
        result.reciprocal_rank = 0.0
        result.match_details["unmatched_retrieved"] = list(retrieved)
        return

    matched_flags = []
    found_patterns = set()

    for uri in retrieved:
        matched, pattern = _any_match(uri, expected)
        matched_flags.append(matched)
        if matched:
            result.match_details["matches"].append({
                "uri": uri,
                "matched_pattern": pattern
            })
            found_patterns.add(pattern)
        else:
            result.match_details["unmatched_retrieved"].append(uri)

    result.match_details["unmatched_expected"] = [
        p for p in expected if p not in found_patterns
    ]

    result.hit = any(matched_flags)
    n_relevant_retrieved = sum(matched_flags)
    result.precision = n_relevant_retrieved / len(retrieved) if retrieved else 0.0
    result.recall = len(found_patterns) / len(expected)

    for rank, matched in enumerate(matched_flags, start=1):
        if matched:
            result.reciprocal_rank = 1.0 / rank
            break


def compute_generation_metrics(result: CaseResult) -> None:
    if result.answer is None:
        return

    answer_lower = result.answer.lower()

    if result.expected_answer is not None:
        result.exact_match = result.expected_answer.lower() in answer_lower

    context_lower = " ".join(result.retrieved_texts).lower()
    words = re.findall(r"[а-яёa-z]{4,}", answer_lower)
    if words and context_lower:
        coverage = sum(1 for w in words if w in context_lower) / len(words)
        result.adherence = coverage >= 0.7
    else:
        result.adherence = None


# ---------------------------------------------------------------------------
# Агрегация
# ---------------------------------------------------------------------------

def aggregate(results: List[CaseResult], language: str) -> Dict:
    def _mean(values):
        v = [x for x in values if x is not None]
        return sum(v) / len(v) if v else None

    n = len(results)
    hit_rate = sum(r.hit for r in results) / n
    precision = _mean([r.precision for r in results])
    recall = _mean([r.recall for r in results])
    mrr = _mean([r.reciprocal_rank for r in results])

    em_results = [r.exact_match for r in results if r.exact_match is not None]
    adherence_r = [r.adherence for r in results if r.adherence is not None]

    return {
        "n_cases": n,
        "language": language,
        "retrieval": {
            "hit_rate": round(hit_rate, 3),
            "precision_k": round(precision, 3) if precision is not None else None,
            "recall_k": round(recall, 3) if recall is not None else None,
            "mrr": round(mrr, 3) if mrr is not None else None,
        },
        "generation": {
            "exact_match": round(sum(em_results) / len(em_results), 3) if em_results else None,
            "adherence": round(sum(adherence_r) / len(adherence_r), 3) if adherence_r else None,
            "n_evaluated": len(em_results),
        },
    }


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------
def print_case(r: CaseResult, verbose: bool, debug_failures: bool) -> None:
    status_icon = "✅" if r.hit else "❌"
    print(f"\n{status_icon} [{r.status}] {r.query!r}")

    # === ДЕБАГ: всегда показываем для провалившихся кейсов ===
    if not r.hit and r.expected_uris:
        print(f"   🔍 RETRIEVAL FAILED:")
        print(f"   ├─ Expected: {r.expected_uris}")
        print(f"   ├─ Retrieved URIs (normalized):")
        for i, (uri, score) in enumerate(zip(r.retrieved_uris[:5], r.retrieved_scores[:5])):
            local = _extract_local_name(uri)
            print(f"   │  {i + 1}. [{score:.3f}] local='{local}' | full='{uri}'")

        # Проверяем матчинг вручную
        print(f"   ├─ Manual match check:")
        for pattern in r.expected_uris:
            print(f"   │  Pattern '{pattern}':")
            for uri in r.retrieved_uris[:3]:
                local = _extract_local_name(uri)
                matched = _uri_matches(uri, pattern)
                print(f"   │    vs '{local}': {matched}")

    # === ДЕБАГ: generation metrics ===
    if r.expected_answer is not None and r.answer is not None:
        answer_lower = r.answer.lower()
        expected_lower = r.expected_answer.lower()
        found = expected_lower in answer_lower
        print(f"   📝 GENERATION CHECK:")
        print(f"   ├─ Expected substring: '{r.expected_answer}' (lower: '{expected_lower}')")
        print(f"   ├─ Found in answer: {found}")
        print(f"   └─ Answer preview: '{r.answer[:200]}...'")

    if verbose:
        print(f"   Expected URIs : {r.expected_uris}")
        print(f"   Retrieved     : {[(u, f'{s:.3f}') for u, s in zip(r.retrieved_uris, r.retrieved_scores)]}")
        print(f"   Hit={r.hit}  P={r.precision:.2f}  R={r.recall:.2f}  RR={r.reciprocal_rank:.2f}")
        if r.answer is not None:
            print(f"   Answer        : {r.answer[:120]!r}{'...' if len(r.answer) > 120 else ''}")
            print(f"   Exact Match   : {r.exact_match}  Adherence: {r.adherence}")

    if debug_failures and not r.hit and r.expected_uris:
        print(f"\n   🔍 DEBUG INFO:")
        print(f"   ├─ Expected patterns: {r.expected_uris}")
        print(f"   ├─ Retrieved URIs:")
        for i, (uri, score) in enumerate(zip(r.retrieved_uris, r.retrieved_scores)):
            local = _extract_local_name(uri)
            print(f"   │  {i + 1}. [{score:.3f}] {local}")
            print(f"   │     Full: {uri}")

        if r.match_details.get("matches"):
            print(f"   ├─ Matches found:")
            for m in r.match_details["matches"]:
                print(f"   │  ✓ {m['uri']} matched '{m['matched_pattern']}'")

        if r.match_details.get("unmatched_expected"):
            print(f"   ├─ NOT FOUND in retrieved:")
            for p in r.match_details["unmatched_expected"]:
                print(f"   │  ✗ Pattern '{p}' not matched")

        if r.match_details.get("unmatched_retrieved"):
            print(f"   └─ Retrieved but not expected:")
            for u in r.match_details["unmatched_retrieved"][:5]:
                print(f"      • {_extract_local_name(u)}")


def print_failure_summary(results: List[CaseResult]) -> None:
    failures = [r for r in results if not r.hit and r.expected_uris]

    if not failures:
        print("\n✅ Все кейсы с ожидаемыми URI прошли успешно!")
        return

    print(f"\n{'=' * 60}")
    print(f"🔴 ПРОВАЛИВШИЕСЯ КЕЙСЫ: {len(failures)}")
    print(f"{'=' * 60}")

    for r in failures:
        print(f"\n❌ Query: {r.query}")
        print(f"   Expected: {r.expected_uris}")
        print(f"   Got (top-3):")
        for uri, score in zip(r.retrieved_uris[:3], r.retrieved_scores[:3]):
            print(f"      [{score:.3f}] {_extract_local_name(uri)}")


def print_summary(agg: Dict, elapsed: float) -> None:
    print("\n" + "=" * 60)
    print("📊  ИТОГИ ОЦЕНКИ")
    print("=" * 60)
    r = agg["retrieval"]
    g = agg["generation"]
    lang_label = agg.get("language", "unknown")
    print(f"Язык               : {lang_label}")
    print(f"Кейсов             : {agg['n_cases']}")
    print(f"\n── Retrieval ──────────────────────────")
    print(f"  Hit Rate@k       : {r['hit_rate']:.3f}")
    print(f"  Precision@k      : {r['precision_k']:.3f}" if r['precision_k'] is not None else "  Precision@k      : —")
    print(f"  Recall@k         : {r['recall_k']:.3f}" if r['recall_k'] is not None else "  Recall@k         : —")
    print(f"  MRR              : {r['mrr']:.3f}" if r['mrr'] is not None else "  MRR              : —")
    print(f"\n── Generation ({g['n_evaluated']} кейсов) ──────────────")
    print(f"  Exact Match      : {g['exact_match']:.3f}" if g['exact_match'] is not None else "  Exact Match      : —")
    print(f"  Adherence        : {g['adherence']:.3f}" if g['adherence'] is not None else "  Adherence        : —")
    print(f"\n⏱️  Время           : {elapsed:.1f}с")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Парсер аргументов для eval.py (отдельный от config.py)
# ---------------------------------------------------------------------------

def parse_eval_args() -> argparse.Namespace:
    """Парсер аргументов специфичных для eval.py."""
    parser = argparse.ArgumentParser(
        description="Evaluation скрипт для Ontology RAG",
        allow_abbrev=False,
        add_help=False
    )
    parser.add_argument("--top-k", type=int, default=8,
                        help="Количество retrieved сущностей (default: 8)")
    parser.add_argument("--language", "-l", type=str, default="ru",
                        choices=["ru", "en", "both"],
                        help="Язык тестовых кейсов: ru, en или both (default: ru)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Пропустить генерацию LLM, только retrieval метрики")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Подробный вывод по каждому кейсу")
    parser.add_argument("--debug-failures", "-d", action="store_true",
                        help="Детальный дебаг провалившихся кейсов")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Сохранить результаты в JSON-файл")

    # Парсим только известные аргументы, остальные игнорируем
    args, unknown = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_eval(
        top_k: int,
        language: str,
        skip_generation: bool,
        verbose: bool,
        debug_failures: bool,
        output_json: Optional[Path],
) -> None:
    try:
        from config import Config
        from knowledge_base import KnowledgeBase
        from llm_client import LLMClient
    except ImportError as e:
        print(f"❌ Не удалось импортировать модули проекта: {e}")
        print("   Запускайте eval.py из той же директории, что и rag_app.py")
        sys.exit(1)

    # Выбираем тестовые кейсы в зависимости от языка
    if language == "ru":
        test_cases = TEST_CASES_RU
        lang_label = "Russian"
    elif language == "en":
        test_cases = TEST_CASES_EN
        lang_label = "English"
    else:  # both
        test_cases = TEST_CASES_RU + TEST_CASES_EN
        lang_label = "Both (RU+EN)"

    print(f"🔧 Инициализация системы…")
    print(f"🌐 Язык тестов: {lang_label}")
    cfg = Config()
    cfg.top_k = top_k
    kb = KnowledgeBase(cfg)
    llm = LLMClient(cfg) if not skip_generation else None

    results: List[CaseResult] = []
    start = time.time()

    for i, case in enumerate(test_cases, 1):
        # print(f"\r⏳ Кейс {i}/{len(test_cases)}: {case['query'][:50]!r}", end="", flush=True)

        retrieved, status = kb.search(case["query"], top_k=top_k)

        retrieved_uris = [e["uri"] for e, _ in retrieved]
        retrieved_scores = [s for _, s in retrieved]
        retrieved_texts = [
            " ".join([e["label"]] + [v for vals in e.get("properties", {}).values() for v in vals])
            for e, _ in retrieved
        ]

        cr = CaseResult(
            query=case["query"],
            expected_uris=case["expected_uris"],
            retrieved_uris=retrieved_uris,
            retrieved_scores=retrieved_scores,
            retrieved_texts=retrieved_texts,
            status=status,
            expected_answer=case.get("expected_answer"),
        )

        compute_retrieval_metrics(cr)

        if llm is not None and retrieved:
            system = llm.build_system_prompt(retrieved)
            cr.answer = llm.call(system=system, user=case["query"])
            compute_generation_metrics(cr)

        results.append(cr)
        #print_case(cr, verbose, debug_failures)

    elapsed = time.time() - start

    #if debug_failures:
        #print_failure_summary(results)

    agg = aggregate(results, language=lang_label)
    #print_summary(agg, elapsed)

    if output_json:
        if not output_json.is_absolute():
            from config import PROJECT_ROOT
            output_json = PROJECT_ROOT / output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)  # создаём папки
        payload = {
            "summary": agg,
            "elapsed_s": round(elapsed, 2),
            "cases": [
                {
                    "query": r.query,
                    "status": r.status,
                    "hit": r.hit,
                    "precision": r.precision,
                    "recall": r.recall,
                    "mrr": r.reciprocal_rank,
                    "exact_match": r.exact_match,
                    "adherence": r.adherence,
                    "retrieved_uris": r.retrieved_uris,
                    "answer": r.answer,
                    "match_details": r.match_details,
                }
                for r in results
            ],
        }
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n💾 Результаты сохранены в {output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Парсим аргументы eval.py ДО импорта Config
    args = parse_eval_args()

    run_eval(
        top_k=args.top_k,
        language=args.language,
        skip_generation=args.skip_generation,
        verbose=args.verbose,
        debug_failures=args.debug_failures,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()