import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from filelock import FileLock
from groq import Groq
from rdflib import Graph, RDFS, URIRef, Literal, OWL
from sentence_transformers import SentenceTransformer

load_dotenv()

def get_args():
    parser = argparse.ArgumentParser(description="Ontology-RAG CLI")
    parser.add_argument("--ontology", default="ontology.ttl", help="Path to ontology file (.ttl)")
    parser.add_argument("--index", default="ontology_index.pkl", help="Path to FAISS index pickle")
    parser.add_argument("--log",   default="rag_log.jsonl",   help="Path to log file (jsonl)")
    return parser.parse_args()

ARGS = get_args()

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

ONTOLOGY_FILE = Path(ARGS.ontology)
INDEX_PATH     = Path(ARGS.index)
LOG_PATH       = Path(ARGS.log)

if INDEX_PATH.exists():
    with INDEX_PATH.open("rb") as f:
        data       = pickle.load(f)
        FACTS      = data["facts"]
        EMBEDDINGS = data["embeddings"]
else:
    g = Graph()
    if not ONTOLOGY_FILE.exists():
        raise FileNotFoundError(f"Ontology file '{ONTOLOGY_FILE}' not found")
    g.parse(ONTOLOGY_FILE, format="turtle")

    definition_prop = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
    fallback_prop   = RDFS.comment

    FACTS = []
    for s in g.subjects(RDFS.label, None):
        label       = g.value(s, RDFS.label)
        definition  = g.value(s, definition_prop) or g.value(s, fallback_prop)
        if isinstance(label, Literal) and isinstance(definition, Literal):
            node_id = str(s).split("/")[-1]
            FACTS.append(f"{label} ({node_id}): {definition}")

    subclass_facts = set()
    for a, _, b in g.triples((None, RDFS.subClassOf, None)):
        for _, _, c in g.triples((b, RDFS.subClassOf, None)):
            if a != c:
                subclass_facts.add(f"{a.split('#')[-1]} is a subclass of {c.split('#')[-1]}")

    FACTS.extend(subclass_facts)

    EMBEDDINGS = MODEL.encode(FACTS, show_progress_bar=True, dtype="float32")
    faiss.normalize_L2(EMBEDDINGS)

    with INDEX_PATH.open("wb") as f:
        pickle.dump({"facts": FACTS, "embeddings": EMBEDDINGS}, f)

index = faiss.IndexFlatIP(EMBEDDINGS.shape[1])
index.add(EMBEDDINGS)

def retrieve_facts(query: str, k: int = 5, threshold: float = 0.5):
    vec = MODEL.encode([query]).astype("float32")
    faiss.normalize_L2(vec)
    dists, idx = index.search(vec, k)
    results = [(FACTS[i], dists[0][j]) for j, i in enumerate(idx[0]) if dists[0][j] >= threshold]
    return [fact for fact, _ in results], [dist for _, dist in results]

def parse_verdict(eval_text: str) -> str:
    first = eval_text.splitlines()[0].strip().lower()
    mapping = {
        "ответ a лучше": "Answer A is better",
        "ответ b лучше": "Answer B is better",
        "ответы равнозначны": "Answers equivalent",
    }
    return mapping.get(first, "Unknown")

FILELOCK = FileLock(str(LOG_PATH) + ".lock")

def log_interaction(entry: dict):
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with FILELOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

groq_client = Groq()

def ask_with_rag_groq(query: str, k: int = 5):
    context_facts, distances = retrieve_facts(query, k)
    context = "\n".join(f"[{i+1}] {fact}" for i, fact in enumerate(context_facts)) + \
              "\n\nCited facts list:\n" + \
              "\n".join(f"[{i+1}] {fact}" for i, fact in enumerate(context_facts))

    prompt = f"""
You are a helpful assistant. Use the following ontology facts to answer the user's question.
Always cite facts explicitly by referring to [1], [2], etc.

Ontology facts:
{context}

User question:
{query}

Answer (with citations and cited fact text):
"""
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content, distances




def ask_without_rag_groq(query: str):
    prompt = f"""
You are a helpful assistant. Answer the following question concisely and factually:

{query}
"""
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content

def evaluate_answers_with_llm(question: str, answer_a: str, answer_b: str):
    prompt = f"""Вы — полезный оценщик. Сравните два ответа на один вопрос.

Вопрос:
{question}

Ответ A:
{answer_a}

Ответ B:
{answer_b}

Сначала выведите **ровно одну строку** ИЗ СПИСКА:
- «Ответ A лучше»
- «Ответ B лучше»
- «Ответы равнозначны»
Затем добавьте одну-две фразы пояснения."""
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content

def main():
    print("\U0001F9E0 Ask me something (type 'exit' to quit)")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        print("\n\U0001F539 Ответ без онтологии:")
        answer_no_rag = ask_without_rag_groq(q)
        print(answer_no_rag)

        print("\n\U0001F538 Ответ с онтологией (RAG):")
        answer_with_rag, distances = ask_with_rag_groq(q)
        print(answer_with_rag)

        print("\n\U0001F9EA \U0001F916 Автоматическая оценка ответа:")
        evaluation = evaluate_answers_with_llm(q, answer_no_rag, answer_with_rag)
        print(evaluation)

        verdict = parse_verdict(evaluation)

        log_interaction({
            "question": q,
            "answer_no_rag": answer_no_rag,
            "answer_with_rag": answer_with_rag,
            "llm_evaluation": evaluation,
            "verdict": verdict,
            "faiss_distances": [float(d) for d in distances],
            "top1_distance": round(float(distances[0]), 4) if distances else None,
            "mean_distance": round(float(np.mean(distances)), 4) if distances else None,
        })
        print("\n\u2705 Записано в лог.")

if __name__ == "__main__":
    main()
