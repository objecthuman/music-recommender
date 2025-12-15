from pathlib import Path

from structlog import get_logger

from src.config import settings
from src.logger import setup_logging
from src.utils import get_audio_files
from src.vector_store import generate_and_upsert_embeddings, get_all_indexed_files

setup_logging("worker.log")
logger = get_logger()


def scan_and_index_new_music():
    if not settings.MUSIC_LIBRARIES:
        logger.warning("No music libraries configured, skipping scan")
        return

    logger.info(
        "Starting music library scan",
        libraries=settings.MUSIC_LIBRARIES,
        scan_interval=settings.SCAN_INTERVAL,
    )

    all_music_files = []
    for directory in settings.MUSIC_LIBRARIES:
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning("Music library does not exist", directory=directory)
                continue

            if not dir_path.is_dir():
                logger.warning("Path is not a directory", directory=directory)
                continue

            files = get_audio_files(folder_path=directory)
            all_music_files.extend(files)
            logger.info(
                "Found music files in library",
                directory=directory,
                count=len(files),
            )
        except FileNotFoundError as e:
            logger.error(
                "Directory not found",
                directory=directory,
                error=str(e),
            )
            continue
        except PermissionError as e:
            logger.error(
                "Permission denied accessing directory",
                directory=directory,
                error=str(e),
            )
            continue
        except Exception as e:
            logger.error(
                "Error scanning library",
                directory=directory,
                error=str(e),
                exc_info=True,
            )
            continue

    if not all_music_files:
        logger.info("No music files found in configured libraries")
        return

    logger.info("Total music files found", count=len(all_music_files))

    try:
        indexed_files = get_all_indexed_files()
        logger.info("Files already indexed", count=len(indexed_files))
    except Exception as e:
        logger.error(
            "Error retrieving indexed files from database",
            error=str(e),
            exc_info=True,
        )
        return

    new_files = list(set(all_music_files) - indexed_files)

    if not new_files:
        logger.info("No new files to index")
        return

    logger.info("New files to index", count=len(new_files))

    try:
        result = generate_and_upsert_embeddings(
            file_paths=new_files,
            batch_size=settings.BATCH_SIZE,
        )

        logger.info(
            "Successfully indexed new music files",
            count=len(result["ids"]),
        )
    except Exception as e:
        logger.error(
            "Error generating and upserting embeddings",
            error=str(e),
            exc_info=True,
        )


if __name__ == "__main__":
    logger.info("Music indexing worker started")
    scan_and_index_new_music()
    logger.info("Music indexing worker completed")
