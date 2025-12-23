import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
import faiss
import requests
from dotenv import load_dotenv
from filelock import FileLock
from rdflib import Graph, RDFS, URIRef, Literal
from sentence_transformers import SentenceTransformer

load_dotenv()


def get_args():
    parser = argparse.ArgumentParser(description="Ontology-RAG CLI with E5")
    parser.add_argument("--ontology", default="ontology.ttl", help="Path to ontology file (.ttl)")
    parser.add_argument("--index", default="ontology_index_e5.pkl", help="Path to FAISS index pickle")
    parser.add_argument("--log", default="rag_log.jsonl", help="Path to log file (jsonl)")
    parser.add_argument("--llama-model", default="llama3:8b", help="Ollama model name")
    return parser.parse_args()


ARGS = get_args()
ONTOLOGY_FILE = Path(ARGS.ontology)
# EMBEDDING_FILE was causing the error and is unused, so it is removed.
INDEX_PATH = Path(ARGS.index)
LOG_PATH = Path(ARGS.log)
MODEL_NAME = ARGS.llama_model  # Fixed: Set model name from args
FILELOCK = FileLock(str(LOG_PATH) + ".lock")
FACTS = []
MODEL = None
INDEX = None


def pick_literal(literals, preferred_langs=("ru", "en")):
    by_lang = {lit.language: str(lit) for lit in literals if isinstance(lit, Literal)}
    for lang in preferred_langs:
        if lang in by_lang:
            return by_lang[lang]
    # If preferred langs not found, return any or None
    if by_lang:
        return next(iter(by_lang.values()))
    # Handle case where literals have no language tag
    str_literals = [str(l) for l in literals if isinstance(l, Literal)]
    return str_literals[0] if str_literals else None


def get_entity_text(g: Graph, entity_uri: URIRef):
    labels = list(g.objects(entity_uri, RDFS.label))
    label = pick_literal(labels)

    # Fallback if no label found
    if not label:
        try:
            label = g.qname(entity_uri).split(":")[1]
        except:
            label = str(entity_uri).split("/")[-1]

    text_values = []
    skip_preds = {
        RDFS.label,
        RDFS.subClassOf,
        URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
    }

    for p, o in g.predicate_objects(entity_uri):
        if p not in skip_preds and isinstance(o, Literal):
            text_values.append(str(o))

    combined_text = " ".join(sorted(set(text_values)))
    try:
        node_id = g.qname(entity_uri).split(":")[1]
    except:
        node_id = str(entity_uri).split("/")[-1]

    return f"{label} ({node_id}): {combined_text}", label


def load_data():
    global FACTS, MODEL, INDEX
    print("Loading E5 model (this might take a while first time)...")
    MODEL = SentenceTransformer('intfloat/multilingual-e5-large')

    if INDEX_PATH.exists():
        print(f"Loading cached index from {INDEX_PATH}...")
        with INDEX_PATH.open("rb") as f:
            data = pickle.load(f)
            FACTS = data["facts"]
            embeddings = data["vectors"]
    else:
        print("Parsing ontology...")
        if not ONTOLOGY_FILE.exists():
            raise FileNotFoundError(f"Ontology file not found: {ONTOLOGY_FILE}")

        g = Graph()
        g.parse(ONTOLOGY_FILE.as_posix(), format="turtle")

        raw_facts = []
        # --- Logic to extract facts from Graph ---
        # Iterate over all subjects that are likely classes or named individuals
        subjects = set(g.subjects())
        for subj in subjects:
            if isinstance(subj, URIRef):
                text, _ = get_entity_text(g, subj)
                if text:
                    raw_facts.append(text)
        # -----------------------------------------

        FACTS = raw_facts
        if not FACTS:
            raise ValueError("Facts list is empty! Check ontology file.")

        print(f"Encoding {len(FACTS)} facts with E5...")

        # E5 expects "passage:" prefix for documents
        facts_to_encode = [f"passage: {fact}" for fact in FACTS]

        embeddings = MODEL.encode(facts_to_encode, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        embeddings = embeddings.astype("float32")

        with INDEX_PATH.open("wb") as f:
            pickle.dump({
                "facts": FACTS,
                "vectors": embeddings
            }, f)

    # FAISS Index
    dimension = embeddings.shape[1]
    INDEX = faiss.IndexFlatIP(dimension)
    INDEX.add(embeddings)
    print("System ready.")


def retrieve_facts(query: str, k: int = 5):
    # E5 expects "query:" prefix for questions
    query_text = f"query: {query}"
    q_vec = MODEL.encode([query_text], normalize_embeddings=True).astype("float32")

    dists, idx = INDEX.search(q_vec, k)
    results = []

    for j, i in enumerate(idx[0]):
        if i < 0: continue
        # Distance threshold (Inner Product)
        if dists[0][j] >= 0.7:
            results.append((FACTS[i], dists[0][j]))

    return [fact for fact, _ in results], [dist for _, dist in results]


def call_llama(prompt: str, model: str = MODEL_NAME, temperature: float = 0.5):
    full_prompt = f"Отвечай только на русском языке.\n\n{prompt}"
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "temperature": temperature,
                "stream": False
            },
            timeout=300
        )
        resp.raise_for_status()
        data = resp.json()
        return data["response"].strip()
    except Exception as e:
        return f"Error calling LLM: {e}"


def ask_with_rag_local(query: str, k: int = 5):
    context_facts, distances = retrieve_facts(query, k)

    if not context_facts:
        return "К сожалению, в онтологии нет информации по вашему запросу.", []

    context = "\n".join(f"[{i + 1}] {fact}" for i, fact in enumerate(context_facts))

    prompt = f"""
    You are a helpful assistant. Use the following ontology entities and facts to answer the user's question.
    Attention: The facts provided below are entities found in the ontology that are semantically close to the words in the user's query.
    Context Information:
    {context}
    User question:
    {query}
    Please answer the question based *only* on the provided context. Cite the facts using [n].
    """
    return call_llama(prompt), distances


def log_interaction(entry: dict):
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with FILELOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    try:
        load_data()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    print(f"\n🔮 RAG Ready (Model: {MODEL_NAME}). Ask me something (type 'exit' to quit)")

    while True:
        try:
            q = input("\n> ").strip()
            if not q: continue
            if q.lower() in {"exit", "quit"}:
                break

            print("\n🔍 Поиск в онтологии...")
            answer_with_rag, distances = ask_with_rag_local(q)

            print("\n📝 Ответ:")
            print(answer_with_rag)

            facts_with_scores = []
            retrieved, dists = retrieve_facts(q, k=5)
            for fact, dist in zip(retrieved, dists):
                facts_with_scores.append({
                    "fact": fact[:100] + "...",
                    "distance": round(float(dist), 4)
                })

            log_interaction({
                "question": q,
                "answer_with_rag": answer_with_rag,
                "retrieved_facts": facts_with_scores,
                "top1_distance": round(float(facts_with_scores[0]["distance"]), 4) if facts_with_scores else None,
            })
            print(f"\n✅ Записано в лог ({LOG_PATH}).")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()