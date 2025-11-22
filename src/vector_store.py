import chromadb
from chromadb.config import Settings


ChromaClient = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(
        is_persistent=True,
    ),
)


Collection = ChromaClient.get_or_create_collection(
    name="udio_embeddings",
    # embedding_function=VoyageEmbeddingFunction(api_key=settings.VOYAGE_API_KEY),
    metadata={"hnsw:space": "cosine"},
)


def add_embeddings(
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
):
    Collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_similar(
    query_embedding: list[float],
    n_results: int = 20,
):
    results = Collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results
