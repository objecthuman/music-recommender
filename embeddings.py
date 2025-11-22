from pathlib import Path
from typing import Annotated
import pickle

import cyclopts
import numpy as np
import torch
import laion_clap
from sklearn.metrics.pairwise import cosine_similarity


app = cyclopts.App()


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x / 32767.0).astype("float32")


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype("int16")


def get_audio_files(music_dir: Path) -> list[Path]:
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(music_dir.rglob(f"*{ext}"))

    return sorted(audio_files)


@app.command
def generate_embeddings(
    music_dir: Annotated[Path, cyclopts.Parameter(help="Directory containing music files")] = Path("music"),
    output_file: Annotated[Path, cyclopts.Parameter(help="Output pickle file path")] = Path("embeddings.pkl"),
    enable_fusion: Annotated[bool, cyclopts.Parameter(help="Enable fusion in CLAP model")] = False,
    batch_size: Annotated[int, cyclopts.Parameter(help="Number of files to process at once")] = 50,
) -> None:
    """Generate audio embeddings for all files in the music directory."""

    if not music_dir.exists():
        print(f"Error: Music directory '{music_dir}' does not exist")
        return

    print(f"Searching for audio files in {music_dir}...")
    audio_files = get_audio_files(music_dir)

    if not audio_files:
        print("No audio files found")
        return

    total_files = len(audio_files)
    print(f"Found {total_files} audio files\n")

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.2f} GB")
        print(f"Using device: GPU (CUDA)\n")
    else:
        device = "cpu"
        print("No GPU detected. Using CPU\n")

    # Load CLAP model
    print("Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
    model.load_ckpt()
    print("Model loaded successfully\n")

    # Generate embeddings with progress tracking
    print("=" * 80)
    print(f"Generating embeddings (batch size: {batch_size})...")
    print("=" * 80)

    all_embeddings: list[np.ndarray] = []
    failed_files: list[Path] = []

    # Process in batches
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = audio_files[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size

        print(f"\n{'='*80}")
        print(f"Batch {batch_num}/{total_batches} - Processing files {batch_start + 1}-{batch_end}/{total_files}")
        print(f"{'='*80}")

        # Show files in this batch
        total_batch_size = sum(f.stat().st_size for f in batch_files)
        print(f"Batch contains {len(batch_files)} files, total size: {format_size(total_batch_size)}")

        # Generate embeddings for batch
        try:
            batch_paths = [str(f) for f in batch_files]

            print(f"Processing batch on {device.upper()}...")
            embeddings = model.get_audio_embedding_from_filelist(
                x=batch_paths,
                use_tensor=False
            )

            all_embeddings.append(embeddings)

            # Show batch stats
            embedding_size = embeddings.nbytes
            print(f"✓ Batch complete!")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Size: {format_size(embedding_size)}")
            print(f"  Files/second: {len(batch_files):.1f}")

        except Exception as e:
            print(f"✗ Error processing batch: {e}")
            print(f"  Falling back to individual file processing...")

            # Fallback: process files individually
            for audio_file in batch_files:
                try:
                    embedding = model.get_audio_embedding_from_filelist(
                        x=[str(audio_file)],
                        use_tensor=False
                    )
                    all_embeddings.append(embedding)
                    print(f"  ✓ {audio_file.name}")
                except Exception as e:
                    print(f"  ✗ {audio_file.name}: {e}")
                    failed_files.append(audio_file)
                    continue

    if not all_embeddings:
        print("\nNo embeddings were generated successfully")
        return

    # Stack all embeddings
    print("\n" + "=" * 80)
    print("Finalizing embeddings...")
    embeddings = np.vstack(all_embeddings)

    # Calculate total size
    total_embedding_size = embeddings.nbytes

    print(f"  Total embeddings shape: {embeddings.shape}")
    print(f"  Total embeddings size: {format_size(total_embedding_size)}")
    print(f"  Embeddings per file: {embeddings.shape[1]} dimensions")
    print(f"  Data type: {embeddings.dtype}")

    # Only include successfully processed files
    successful_files = [f for f in audio_files if f not in failed_files]

    # Prepare data for saving
    embedding_data = {
        "files": [str(f.relative_to(music_dir)) for f in successful_files],
        "absolute_paths": [str(f) for f in successful_files],
        "embeddings": embeddings,
        "shape": embeddings.shape,
    }

    # Save to pickle
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(embedding_data, f)

    pickle_size = output_file.stat().st_size

    print("\n" + "=" * 80)
    print("✓ SUCCESS")
    print("=" * 80)
    print(f"  Processed files: {len(successful_files)}/{total_files}")
    if failed_files:
        print(f"  Failed files: {len(failed_files)}")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Output file: {output_file}")
    print(f"  Output file size: {format_size(pickle_size)}")
    print("=" * 80)


@app.command
def get_recommendation(
    song_path: Annotated[Path, cyclopts.Parameter(help="Path to the input song")],
    embeddings_file: Annotated[Path, cyclopts.Parameter(help="Path to embeddings pickle file")] = Path("embeddings.pkl"),
    top_k: Annotated[int, cyclopts.Parameter(help="Number of recommendations to return")] = 20,
    enable_fusion: Annotated[bool, cyclopts.Parameter(help="Enable fusion in CLAP model")] = False,
) -> None:
    """Get music recommendations based on similarity to an input song."""

    if not song_path.exists():
        print(f"Error: Song file '{song_path}' does not exist")
        return

    if not embeddings_file.exists():
        print(f"Error: Embeddings file '{embeddings_file}' does not exist")
        print("Run 'generate-embeddings' first to create the embeddings file")
        return

    print(f"Loading embeddings from {embeddings_file}...")
    with open(embeddings_file, "rb") as f:
        embedding_data = pickle.load(f)

    all_embeddings = embedding_data["embeddings"]
    all_files = embedding_data["files"]
    total_songs = len(all_files)

    print(f"Loaded {total_songs} song embeddings")

    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")

    # Load CLAP model
    print("Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
    model.load_ckpt()
    print("Model loaded successfully\n")

    # Generate embedding for input song
    print(f"Generating embedding for: {song_path.name}")
    try:
        input_embedding = model.get_audio_embedding_from_filelist(
            x=[str(song_path)],
            use_tensor=False
        )
    except Exception as e:
        print(f"Error generating embedding for input song: {e}")
        return

    print(f"Input embedding shape: {input_embedding.shape}\n")

    # Calculate cosine similarity
    print("Calculating similarities...")
    similarities = cosine_similarity(input_embedding, all_embeddings)[0]

    # Get top-k most similar songs
    # Sort by similarity (descending)
    top_indices = np.argsort(similarities)[::-1]

    # Check if input song is in the database
    input_song_name = str(song_path.name)
    filtered_indices = []

    for idx in top_indices:
        # Skip if it's the exact same file
        if all_files[idx].endswith(input_song_name):
            continue
        filtered_indices.append(idx)
        if len(filtered_indices) >= top_k:
            break

    # Display recommendations
    print("=" * 80)
    print(f"Top {top_k} Recommendations for: {song_path.name}")
    print("=" * 80)

    for rank, idx in enumerate(filtered_indices, start=1):
        similarity_score = similarities[idx]
        song_file = all_files[idx]

        print(f"\n{rank:2d}. {song_file}")
        # print(f"    Similarity: {similarity_score:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    app()
