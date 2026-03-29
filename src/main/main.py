import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from rag_app import RAGApp


if __name__ == "__main__":
    RAGApp().run()