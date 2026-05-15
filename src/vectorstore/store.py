import json

import chromadb

from src.core.config import settings
from src.vectorstore.embeddings import embed_text, embed_texts


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_embedding_text(food: dict) -> str:
    parts = [food["name_pt"]]
    parts.extend(food.get("synonyms", []))
    parts.append(food.get("category", ""))
    return " ".join(parts).strip()


def seed_foods(foods: list[dict]) -> int:
    collection = get_collection()

    ids = []
    documents = []
    metadatas = []

    texts_to_embed = []
    for food in foods:
        embedding_text = build_embedding_text(food)
        ids.append(food["id"])
        documents.append(embedding_text)
        texts_to_embed.append(embedding_text)
        metadatas.append({
            "name_pt": food["name_pt"],
            "category": food.get("category", ""),
            "nutrients_json": json.dumps(food["nutrients_per_100g"]),
            "portions_json": json.dumps(food.get("common_portions", [])),
            "synonyms_json": json.dumps(food.get("synonyms", [])),
        })

    embeddings = embed_texts(texts_to_embed)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def search_foods(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    foods = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance
        metadata = results["metadatas"][0][i]
        foods.append({
            "id": results["ids"][0][i],
            "name_pt": metadata["name_pt"],
            "category": metadata["category"],
            "nutrients_per_100g": json.loads(metadata["nutrients_json"]),
            "common_portions": json.loads(metadata["portions_json"]),
            "synonyms": json.loads(metadata["synonyms_json"]),
            "similarity": round(similarity, 4),
        })

    return foods
