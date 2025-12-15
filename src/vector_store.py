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
    file_paths: list[str], batch_size: int = 32
) -> EmbeddingResult:
    all_metadatas = []
    all_ids = []
    total_batches = (len(file_paths) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(file_paths), batch_size), 1):
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
            try:
                metadata = extract_metadata(file_path)
                batch_metadatas.append(metadata)

                id_str = f"{metadata['artist']}_{metadata['song_name']}_{metadata['album_name']}_{metadata['genre']}_{Path(file_path).stem}".replace(
                    " ", "_"
                )
                batch_ids.append(id_str)
            except FileNotFoundError as e:
                logger.error("Audio file not found", file_path=file_path, error=str(e))
                continue
            except PermissionError as e:
                logger.error("Permission denied reading file", file_path=file_path, error=str(e))
                continue
            except Exception as e:
                logger.error("Error extracting metadata", file_path=file_path, error=str(e), exc_info=True)
                continue

        if not batch_ids:
            logger.warning("No valid files in batch, skipping", batch=batch_num)
            continue

        try:
            batch_embeddings = clap_model.get_audio_embedding_from_filelist(
                x=batch_paths,
                use_tensor=False,
            )
            batch_embeddings_list = batch_embeddings.tolist()
        except RuntimeError as e:
            logger.error("Runtime error generating embeddings", batch=batch_num, error=str(e), exc_info=True)
            continue
        except Exception as e:
            logger.error("Error generating embeddings", batch=batch_num, error=str(e), exc_info=True)
            continue

        try:
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
        except ValueError as e:
            logger.error("Invalid data for database upsert", batch=batch_num, error=str(e), exc_info=True)
            continue
        except Exception as e:
            logger.error("Error upserting to database", batch=batch_num, error=str(e), exc_info=True)
            continue

        all_metadatas.extend(batch_metadatas)
        all_ids.extend(batch_ids)

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


def get_all_indexed_files() -> set[str]:
    try:
        results = Collection.get(include=["metadatas"])

        if not results or not results.get("metadatas"):
            logger.info("No indexed files found in database")
            return set()

        indexed_files = set()
        for metadata in results["metadatas"]:
            if metadata and "file_path" in metadata:
                indexed_files.add(metadata["file_path"])

        logger.info("Retrieved indexed files from database", count=len(indexed_files))
        return indexed_files

    except Exception as e:
        logger.error("Error retrieving indexed files from database", error=str(e), exc_info=True)
        return set()
