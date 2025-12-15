import json
from pathlib import Path
from typing import TypedDict

from mutagen._file import File as MutagenFile


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

class MusicMetadata(TypedDict):
    artist: str
    genre: str
    song_name: str
    album_name: str
    file_path: str


def get_audio_files(
    file_paths: list[str] | None = None, folder_path: str | None = None
) -> list[str]:
    if file_paths:
        return file_paths

    if folder_path:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")

        files = [
            str(f) for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
        ]

        if not files:
            raise ValueError(f"No audio files found in folder: {folder_path}")

        return files

    raise ValueError("Either file_paths or folder_path must be provided")


def extract_metadata(file_path: str) -> MusicMetadata:
    try:
        audio = MutagenFile(file_path)
        if audio is None:
            return MusicMetadata(
                artist="Unknown",
                genre="Unknown",
                song_name=Path(file_path).stem,
                album_name="Unknown",
                file_path=file_path,
            )

        artist = "Unknown"
        genre = "Unknown"
        song_name = Path(file_path).stem
        album_name = "Unknown"

        if audio.tags:
            artist = (
                str(audio.tags.get("TPE1", ["Unknown"])[0])
                if "TPE1" in audio.tags
                else str(audio.tags.get("artist", ["Unknown"])[0])
                if "artist" in audio.tags
                else "Unknown"
            )
            genre = (
                str(audio.tags.get("TCON", ["Unknown"])[0])
                if "TCON" in audio.tags
                else str(audio.tags.get("genre", ["Unknown"])[0])
                if "genre" in audio.tags
                else "Unknown"
            )
            song_name = (
                str(audio.tags.get("TIT2", [Path(file_path).stem])[0])
                if "TIT2" in audio.tags
                else str(audio.tags.get("title", [Path(file_path).stem])[0])
                if "title" in audio.tags
                else Path(file_path).stem
            )
            album_name = (
                str(audio.tags.get("TALB", ["Unknown"])[0])
                if "TALB" in audio.tags
                else str(audio.tags.get("album", ["Unknown"])[0])
                if "album" in audio.tags
                else "Unknown"
            )

        return MusicMetadata(
            artist=artist,
            genre=genre,
            song_name=song_name,
            album_name=album_name,
            file_path=file_path,
        )
    except Exception:
        return MusicMetadata(
            artist="Unknown",
            genre="Unknown",
            song_name=Path(file_path).stem,
            album_name="Unknown",
            file_path=file_path,
        )

