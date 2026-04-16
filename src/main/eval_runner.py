import subprocess
from pathlib import Path

# --- Настройки путей ---
PYTHON_EXE = "/Users/i.a.babushkin/PycharmProjects/pythonProject/.venv/bin/python"
EVAL_SCRIPT = "src/main/eval.py"
PROJECT_ROOT = Path("/Users/i.a.babushkin/PycharmProjects/ontology_rag")
BENCHMARK_DIR = PROJECT_ROOT / "src/benchmark/graph"
RESOURCES_DIR = PROJECT_ROOT / "resources"

# Файлы индексов, которые нужно чистить при смене эмбеддинг-модели
INDEX_FILES = [
    "ontology_index.pkl",
    "ontology_index.pkl_en.faiss",
    "ontology_index.pkl_ru.faiss"
]

# --- Конфигурация матриц тестирования ---
LANGUAGES = ["en"]

EMBED_MODELS = {
    "bge-m3": "bge-m3:latest",
    # "mxbai-embed": "mxbai-embed-large:latest",
    # "nomic-embed": "nomic-embed-text:latest"
}

LLM_MODELS = {
    # "deepseek-r1": "deepseek-r1:8b",
    "llama-8b": "llama3:8b",
    # "microsoft-phi4": "phi4-mini:latest",
    # "mistral-7b": "mistral:7b",
    # "qwen3-8b": "qwen3:8b"
}

THRESHOLDS = [0.6]


def clean_indices():
    """Удаляет файлы индекса перед сменой модели эмбеддингов."""
    print("\n🧹 Очистка старых индексов...")
    for file_name in INDEX_FILES:
        file_path = RESOURCES_DIR / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"🗑️ Удален: {file_name}")
            except Exception as e:
                print(f"⚠️ Не удалось удалить {file_name}: {e}")
        else:
            print(f"📁 Файл {file_name} не найден, пропускаем.")


def run_command(cmd):
    print(f"\n🚀 Запуск: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при выполнении: {e}")


def should_clean_indices(embed_folder):
    """
    Проверяет, нужно ли чистить индексы для данной эмбеддинг-модели.
    Если ВСЕ тесты для этой модели уже имеют готовые JSON — чистка не нужна.
    """
    for lang in LANGUAGES:
        for llm_folder in LLM_MODELS:
            for thr in THRESHOLDS:
                output_dir = BENCHMARK_DIR / lang / embed_folder / llm_folder
                thr_str = str(thr).replace('.', '')
                output_file = output_dir / f"output_thr-{thr_str}.json"
                if not output_file.exists():
                    return True
    return False


def main():
    if not (PROJECT_ROOT / EVAL_SCRIPT).exists():
        print(f"Ошибка: Не найден {EVAL_SCRIPT}. Проверьте пути.")
        return

    total_skipped = 0
    total_run = 0

    # Внешний цикл по эмбеддингам — именно здесь мы чистим индексы
    for embed_folder, embed_model in EMBED_MODELS.items():
        print(f"\n" + "=" * 50)
        print(f"🧬 ПЕРЕКЛЮЧЕНИЕ НА ЭМБЕДДИНГ: {embed_model}")
        print("=" * 50)

        # Чистим только если есть хотя бы один незавершённый тест для этой модели
        if should_clean_indices(embed_folder):
            clean_indices()
        else:
            print("✅ Все тесты для этого эмбеддинга уже выполнены, пропускаем очистку индексов.")
            total_skipped += len(LANGUAGES) * len(LLM_MODELS) * len(THRESHOLDS)
            continue

        for lang in LANGUAGES:
            for llm_folder, llm_model in LLM_MODELS.items():
                for thr in THRESHOLDS:
                    output_dir = BENCHMARK_DIR / lang / embed_folder / llm_folder
                    output_dir.mkdir(parents=True, exist_ok=True)

                    thr_str = str(thr).replace('.', '')
                    output_file = output_dir / f"output_thr-{thr_str}.json"

                    # --- Проверка: если JSON уже есть — скипаем ---
                    if output_file.exists():
                        print(f"\n⏭️ SKIP: {output_file.relative_to(PROJECT_ROOT)} уже существует, пропускаем.")
                        total_skipped += 1
                        continue

                    cmd = [
                        PYTHON_EXE, str(PROJECT_ROOT / EVAL_SCRIPT),
                        "--language", lang,
                        "--lang-index", lang,
                        "--embed-model", embed_model,
                        "--llm-model", llm_model,
                        "--score-threshold", str(thr),
                        "--output-json", str(output_file)
                    ]

                    run_command(cmd)
                    total_run += 1

    # --- Итоговая статистика ---
    print(f"\n" + "=" * 50)
    print(f"📊 ИТОГО: выполнено {total_run}, пропущено {total_skipped}")
    print("=" * 50)


if __name__ == "__main__":
    main()