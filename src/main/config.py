import argparse
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCES_DIR = PROJECT_ROOT / "ontology_rag/resources"

def build_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ontology-RAG CLI")
    parser.add_argument("--ontology", default=str(RESOURCES_DIR / "cvdo.owl"))
    parser.add_argument("--index", default=str(RESOURCES_DIR / "ontology_index.pkl"))
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--llm-model", default="llama3:8b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--embed-delay", type=float, default=0.2)
    parser.add_argument("--embed-workers", type=int, default=1,
                        help="Parallel workers for embedding requests")
    parser.add_argument("--index-version", type=int, default=3)
    parser.add_argument("--lang-index", type=str, default="en",
                        choices=["ru", "en"],
                        help="Load only the specified language index (ru or en). "
                             "If not set, en index is loaded.")
    args, _ = parser.parse_known_args()
    return args


class Config:
    INDEX_SCHEMA_VERSION = 5  # add graph in cache and fix owl format indexing

    def __init__(self, args: Optional[argparse.Namespace] = None):
        if args is None:
            args = build_config()
        self.ontology_path = Path(args.ontology)
        self.index_path = Path(args.index)
        self.ollama_base = args.llm_host
        self.llm_model = args.llm_model
        self.embed_model = args.embed_model
        self.top_k = args.top_k
        self.chunk_size = args.chunk_size
        self.score_threshold = args.score_threshold
        self.embed_delay = args.embed_delay
        self.embed_workers = args.embed_workers
        self.lang_index = getattr(args, 'lang_index', None)