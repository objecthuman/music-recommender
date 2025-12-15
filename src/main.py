import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from structlog import get_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import Logger, setup_logging
from src.vector_store import generate_and_upsert_embeddings, get_similar_tracks
from src.utils import get_audio_files
from src.middleware import LoggerMiddleWare


class EmbeddingRequest(BaseModel):
    file_paths: list[str] = Field(default=[])
    folder_path: str | None = Field(default=None)
    batch_size: int = Field(default=32, gt=0, le=128)


class EmbeddingResponse(BaseModel):
    success: bool
    count: int
    message: str


class SimilarityRequest(BaseModel):
    file_path: str
    top_k: int = Field(default=20, gt=0, le=100)


class SimilarTrack(BaseModel):
    id: str
    metadata: dict
    distance: float


class SimilarityResponse(BaseModel):
    query_file: str
    results: list[SimilarTrack]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


async def worker_process_manager(logger: Logger):
    while True:
        try:
            logger.info("Starting worker process", scan_interval=settings.SCAN_INTERVAL)

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "src.worker",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            await process.wait()

            if process.returncode == 0:
                logger.info("Worker process completed successfully")
            else:
                logger.error(
                    "Worker process failed",
                    return_code=process.returncode,
                )

        except Exception as e:
            logger.error("Error running worker process", error=str(e), exc_info=True)

        logger.info("Waiting for next scan", interval=settings.SCAN_INTERVAL)
        await asyncio.sleep(settings.SCAN_INTERVAL)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging("melodion.log")
    logger: Logger = get_logger()
    logger.info("Music recommender started.")

    worker_task = asyncio.create_task(worker_process_manager(logger))
    logger.info("Worker process manager started")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Worker process manager cancelled")
        pass


app = FastAPI(
    title="Melodion (Music Recommendation API)", version="0.1.0", lifespan=lifespan
)

app.add_middleware(LoggerMiddleWare)


@app.get("/v1/health", response_model=HealthResponse)
async def health_check(logger: Logger):
    logger.info("Health check", model_loaded=True)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


@app.post("/v1/music/similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest, logger: Logger):
    full_file_path = ""

    if Path(request.file_path).is_absolute() and Path(request.file_path).exists():
        full_file_path = request.file_path
    else:
        for library_path in settings.MUSIC_LIBRARIES:
            candidate_path = Path(library_path) / request.file_path
            if candidate_path.exists():
                full_file_path = str(candidate_path)
                break

    if not full_file_path:
        logger.warning("File not found in any library", file_path=request.file_path)
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {request.file_path}",
        )

    try:
        logger.info(
            "Finding similar tracks", file_path=full_file_path, top_k=request.top_k
        )

        results = get_similar_tracks(full_file_path, n_results=request.top_k)

        logger.info(
            "Similar tracks found",
            query_file=request.file_path,
            count=len(results),
        )

        similar_tracks = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i].copy()

            if "file_path" in metadata and settings.MUSIC_LIBRARIES:
                file_path = metadata["file_path"]
                for library_path in settings.MUSIC_LIBRARIES:
                    library_path_obj = Path(library_path)
                    file_path_obj = Path(file_path)
                    try:
                        relative_path = file_path_obj.relative_to(library_path_obj)
                        metadata["file_path"] = str(relative_path)
                        break
                    except ValueError:
                        continue

            similar_tracks.append(
                SimilarTrack(
                    id=results["ids"][0][i],
                    metadata=metadata,
                    distance=results["distances"][0][i],
                )
            )

        return SimilarityResponse(
            query_file=request.file_path,
            results=similar_tracks,
            count=len(similar_tracks),
        )

    except Exception as e:
        logger.error("Error finding similar tracks", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar tracks: {str(e)}",
        )
