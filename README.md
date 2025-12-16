# Melodion

A simple music clustering tool that groups music based on similarity.

## Overview

Melodion uses [CLAP](https://github.com/LAION-AI/CLAP) as the inference model to generate audio file embeddings. It can be used directly with [navidrome](https://github.com/objecthuman/navidrome) (our custom version) to queue similar music when playing.

## Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the server in development mode:
   ```bash
   uv run uvicorn src.main:app --reload
   ```

## Configuration

Set the following environment variables:

- **MUSIC_LIBRARIES**: List of music library directories to scan for music files
- **SCAN_INTERVAL**: Interval in seconds between automatic scans for new music files (default: 3600)

### Navidrome Integration

To make it work with navidrome, set `MUSIC_LIBRARIES` to the same value as you have in your navidrome config/library.
