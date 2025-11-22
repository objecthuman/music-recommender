from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from laion_clap import CLAP_Module as ClapModel


clap_model: ClapModel | None = None


class EmbeddingRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1)
    enable_fusion: bool = Field(default=False)


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    shape: tuple[int, int]
    files: list[str]
    device: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clap_model

    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU detected. Using CPU")

    print("Loading CLAP model...")
    clap_model = ClapModel(enable_fusion=False, device=device)
    clap_model.load_ckpt()
    print("✓ CLAP model loaded successfully")

    yield

    print("Shutting down...")


app = FastAPI(title="Music Recommendation API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy",
        model_loaded=clap_model is not None,
        device=device,
    )


@app.post("/generate-embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    if clap_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    invalid_files = []
    for file_path in request.file_paths:
        if not Path(file_path).exists():
            invalid_files.append(file_path)

    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Files not found: {', '.join(invalid_files)}",
        )

    try:
        embeddings = clap_model.get_audio_embedding_from_filelist(
            x=request.file_paths,
            use_tensor=False,
        )

        embeddings_list = embeddings.tolist()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return EmbeddingResponse(
            embeddings=embeddings_list,
            shape=embeddings.shape,
            files=request.file_paths,
            device=device,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}",
        )


