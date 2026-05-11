# Ontology RAG

## Технологии

- **Python 3.9+**
- **FAISS** — векторный поиск (Facebook AI Similarity Search)
- **Ollama** — локальная LLM (Llama 3, Mistral, Qwen и др.)
- **RDFLib** — работа с RDF/OWL онтологиями
- **NumPy** — векторные вычисления
- **Requests** — HTTP-запросы к API эмбеддингов

## Установка

### 1. Клонирование репозитория

```bash
cd /path/to/ontology_rag
```

### 2. Создание виртуального окружения

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# или
.venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r src/requirements.txt
```

### 4. Настройка окружения

Создайте файл `.env` в корне проекта:

```bash
# Ollama API
OLLAMA_BASE_URL=http://localhost:11434

# Модели
LLM_MODEL=llama3:8b
EMBED_MODEL=nomic-embed-text

# Параметры поиска
TOP_K=8
SCORE_THRESHOLD=0.6
CHUNK_SIZE=16

# Параметры эмбеддинга
EMBED_DELAY=0.2
EMBED_WORKERS=1
```

### 5. Установка Ollama

Убедитесь, что Ollama установлена и запущена:

```bash
# Установка 

# (macOS)
brew install ollama

# (windows)
irm https://ollama.com/install.ps1 | iex

# (linux)
curl -fsSL https://ollama.com/install.sh | sh

# Запуск сервера
ollama serve

# Pull моделей
ollama pull llama3:8b
ollama pull bge-m3
```

## Запуск

### Быстрый старт

```bash
cd src/main
python main.py
```

### Параметры командной строки

```bash
python main.py \
  --ontology /path/to/ontology.ttl \
  --index /path/to/index.pkl \
  --llm-host http://localhost:11434 \
  --llm-model llama3:8b \
  --embed-model nomic-embed-text \
  --top-k 8 \
  --chunk-size 16 \
  --score-threshold 0.6 \
  --embed-delay 0.2 \
  --embed-workers 1 \
  --lang-index en  # ru или en для выбора языка индекса
```

#### Описание параметров

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--ontology` | `resources/ontology.ttl` | Путь к файлу онтологии (TTL/RDF/OWL) |
| `--index` | `resources/ontology_index.pkl` | Путь к файлу индекса |
| `--llm-host` | `http://localhost:11434` | URL Ollama API |
| `--llm-model` | `llama3:8b` | Модель для генерации ответов |
| `--embed-model` | `nomic-embed-text` | Модель для эмбеддингов |
| `--top-k` | `8` | Количество результатов поиска |
| `--chunk-size` | `16` | Размер батча для эмбеддингов |
| `--score-threshold` | `0.6` | Порог релевантности |
| `--embed-delay` | `0.2` | Задержка между запросами эмбеддингов (сек) |
| `--embed-workers` | `1` | Количество параллельных воркеров |
| `--lang-index` | `en` | Язык индекса (`ru` или `en`) |

## Структура проекта

```
ontology_rag/
├── src/
│   ├── main/
│   │   ├── main.py           # Точка входа
│   │   ├── rag_app.py        # Основное приложение RAG
│   │   ├── knowledge_base.py # База знаний и поиск
│   │   ├── llm_client.py     # Клиент для LLM
│   │   ├── rdf_processor.py  # Обработка RDF/OWL
│   │   ├── eval.py           # Оценка качества
│   │   ├── eval_runner.py    # Запуск оценки
│   │   └── config.py         # Конфигурация
│   ├── benchmark/
│   │   └── prog_langs/       # Бенчмарки для разных языков и моделей
│   └── requirements.txt      # Зависимости Python
├── resources/
│   ├── ontology.ttl          # Онтология в формате Turtle
│   ├── ontology.rdf          # Онтология в формате RDF
│   ├── cvdo.owl              # OWL онтология
│   └── question.txt          # Примеры вопросов
└── README.md                 # Документация
```

## Бенчмарки

В проекте предусмотрена система бенчмарков для оценки качества поиска:

```bash
# Запуск оценки
python src/main/eval_runner.py \
  --ontology resources/ontology.ttl \
  --questions resources/questions.jsonl \
  --output results.json
```

Результаты бенчмарков хранятся в `src/benchmark/prog_langs/` с разбивкой по:
- Языку вопросов (en/ru)
- Модели эмбеддингов (bge-m3, mxbai-embed, nomic-embed)
- LLM (llama-8b, mistral-7b, qwen3-8b, microsoft-phi4, deepseek-r1)
- Порогу релевантности (0.5, 0.6, 0.65, 0.7)

## Конфигурация

### Настройка порогов

- `SCORE_THRESHOLD` (0.6) — минимальный порог релевантности
- Динамический порог адаптируется на основе распределения скорингов
- Elbow-метод определяет естественный разрыв в релевантности

### Оптимизация производительности

- `EMBED_WORKERS` — увеличьте для параллелизации запросов эмбеддингов
- `CHUNK_SIZE` — оптимален в диапазоне 8-32
- `EMBED_DELAY` — предотвращает rate limiting API

### Выбор моделей

Рекомендуемые комбинации:

| Задача | LLM | Embedding |
|--------|-----|-----------|
| Базовая | llama3:8b | nomic-embed-text |
| Качество | qwen3:8b | bge-m3 |
| Скорость | mistral-7b | mxbai-embed |

## Разработка

### Добавление новой онтологии

1. Поместите файл онтологии в `resources/`
2. Запустите с указанием пути:
   ```bash
   python src/main/main.py --ontology resources/my_ontology.owl
   ```
