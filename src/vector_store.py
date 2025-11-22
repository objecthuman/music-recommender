from pathlib import Path
from typing import TypedDict

import chromadb
import torch
from chromadb.config import Settings
from laion_clap import CLAP_Module as ClapModel
from structlog import get_logger

from src.utils import (
    MusicMetadata,
    extract_metadata,
    load_checkpoint,
    save_checkpoint,
    clear_checkpoint,
)

logger = get_logger()


class EmbeddingResult(TypedDict):
    metadatas: list[MusicMetadata]
    ids: list[str]


class SimilarityResult(TypedDict):
    ids: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]


ChromaClient = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(
        is_persistent=True,
    ),
)


Collection = ChromaClient.get_or_create_collection(
    name="udio_embeddings",
    metadata={"hnsw:space": "cosine"},
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


clap_model = ClapModel(enable_fusion=False, device=device)
clap_model.load_ckpt()


def generate_and_upsert_embeddings(
    file_paths: list[str], batch_size: int = 32, folder_path: str | None = None
) -> EmbeddingResult:
    all_metadatas = []
    all_ids = []
    total_batches = (len(file_paths) + batch_size - 1) // batch_size
    processed_files = []

    start_batch = 1
    if folder_path:
        start_batch = load_checkpoint(folder_path, file_paths) + 1
        if start_batch > 1:
            logger.info(
                "Resuming from checkpoint",
                folder_path=folder_path,
                start_batch=start_batch,
                total_batches=total_batches,
            )
            processed_files = file_paths[: (start_batch - 1) * batch_size]

    for batch_num, i in enumerate(range(0, len(file_paths), batch_size), 1):
        if batch_num < start_batch:
            continue

        batch_paths = file_paths[i : i + batch_size]

        logger.info(
            "Processing batch",
            batch=batch_num,
            total_batches=total_batches,
            batch_size=len(batch_paths),
        )

        batch_metadatas = []
        batch_ids = []
        for file_path in batch_paths:
            metadata = extract_metadata(file_path)
            batch_metadatas.append(metadata)

            id_str = f"{metadata['artist']}_{metadata['song_name']}_{metadata['album_name']}_{metadata['genre']}_{Path(file_path).stem}".replace(
                " ", "_"
            )
            batch_ids.append(id_str)

        batch_embeddings = clap_model.get_audio_embedding_from_filelist(
            x=batch_paths,
            use_tensor=False,
        )

        batch_embeddings_list = batch_embeddings.tolist()

        Collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings_list,
            metadatas=batch_metadatas,
        )

        logger.info(
            "Batch upserted",
            batch=batch_num,
            total_batches=total_batches,
            upserted_count=len(batch_ids),
        )

        processed_files.extend(batch_paths)

        if folder_path:
            save_checkpoint(folder_path, batch_num, processed_files)

        all_metadatas.extend(batch_metadatas)
        all_ids.extend(batch_ids)

    if folder_path:
        clear_checkpoint(folder_path)
        logger.info("Checkpoint cleared", folder_path=folder_path)

    return EmbeddingResult(
        metadatas=all_metadatas,
        ids=all_ids,
    )


def get_similar_tracks(file_path: str, n_results: int = 20) -> SimilarityResult:
    embedding = clap_model.get_audio_embedding_from_filelist(
        x=[file_path],
        use_tensor=False,
    )

    embedding_list = embedding.tolist()[0]

    results = Collection.query(
        query_embeddings=[embedding_list],
        n_results=n_results,
    )

    return results
