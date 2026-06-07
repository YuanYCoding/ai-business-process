"""
Audio file management service.
Handles upload, URL download, listing, and deletion of audio files.
"""

import os
import re
import uuid
import time
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote

import requests
from fastapi import UploadFile

from models.schemas import AudioFile, AudioStatus, DownloadProgress

# Supported audio formats
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".wmv", ".amr", ".webm"}

# Thread-safe download progress tracking
_download_progress: Dict[str, DownloadProgress] = {}
_download_lock = threading.Lock()

# Cancellation flag
_download_cancelled = threading.Event()


def _sanitize_filename(url: str, content_disposition: Optional[str] = None) -> str:
    """Extract a clean filename from URL or Content-Disposition header."""
    if content_disposition:
        match = re.search(r'filename[^;=\n]*=((["\']).*?\2|[^;\n]*)', content_disposition, re.I)
        if match:
            fname = match.group(1).strip().strip('"\'')
            return unquote(fname)

    parsed = urlparse(url)
    path = unquote(parsed.path)
    name = os.path.basename(path)
    if name and any(name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        return name
    # Try to find filename in query params
    if not name or "." not in name:
        for ext in SUPPORTED_EXTENSIONS:
            pattern = re.compile(rf'[\w\-]+{re.escape(ext)}', re.I)
            match = pattern.search(url)
            if match:
                return match.group(0)
    if not name:
        name = f"audio_{uuid.uuid4().hex[:8]}.wav"
    return name


def _ensure_audio_ext(filename: str) -> str:
    """Ensure the filename has a supported audio extension."""
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return filename
    return filename + ".wav"


def get_audio_dir(base_dir: str = "./data/audio") -> str:
    """Get and ensure the audio directory exists."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_uploaded_files(files: List[UploadFile], base_dir: str = "./data/audio") -> List[AudioFile]:
    """Save uploaded files to the audio directory."""
    audio_dir = get_audio_dir(base_dir)
    results = []

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(AudioFile(
                filename=file.filename,
                path="",
                status=AudioStatus.ERROR,
                error_message=f"Unsupported format: {ext}"
            ))
            continue

        safe_name = _ensure_audio_ext(file.filename)
        filepath = os.path.join(audio_dir, safe_name)

        # Handle duplicate filenames
        counter = 1
        base, ext = os.path.splitext(safe_name)
        while os.path.exists(filepath):
            filepath = os.path.join(audio_dir, f"{base}_{counter}{ext}")
            counter += 1

        try:
            with open(filepath, "wb") as f:
                content = file.file.read()
                f.write(content)

            results.append(AudioFile(
                filename=os.path.basename(filepath),
                path=filepath,
                size_bytes=len(content),
                status=AudioStatus.UPLOADED
            ))
        except Exception as e:
            results.append(AudioFile(
                filename=file.filename,
                path="",
                status=AudioStatus.ERROR,
                error_message=str(e)
            ))

    return results


def download_audio_from_urls(
    urls: List[str],
    base_dir: str = "./data/audio",
    max_workers: int = 4,
    max_retries: int = 3,
    progress_callback: Optional[Callable] = None
) -> List[AudioFile]:
    """Download audio files from URLs using multi-threading with retry support."""
    global _download_progress, _download_cancelled
    audio_dir = get_audio_dir(base_dir)
    _download_cancelled.clear()

    # Initialize progress tracking
    with _download_lock:
        _download_progress.clear()
        for url in urls:
            _download_progress[url] = DownloadProgress(
                url=url,
                filename="",
                progress=0.0,
                status="pending"
            )

    results = []

    def _download_one(url: str) -> AudioFile:
        last_error = None
        for attempt in range(max_retries + 1):
            retry_suffix = f" (重试 {attempt}/{max_retries})" if attempt > 0 else ""
            with _download_lock:
                _download_progress[url].status = "downloading"
                _download_progress[url].progress = 5.0
                if attempt > 0:
                    _download_progress[url].error_message = f"重试中... ({attempt}/{max_retries})"

            try:
                if _download_cancelled.is_set():
                    return AudioFile(filename="", path="", status=AudioStatus.ERROR, error_message="Cancelled")

                # Stream download with progress
                response = requests.get(url, stream=True, timeout=300, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                response.raise_for_status()

                # Get filename
                content_disp = response.headers.get("Content-Disposition", "")
                filename = _sanitize_filename(url, content_disp)
                filename = _ensure_audio_ext(filename)

                # Handle duplicates
                filepath = os.path.join(audio_dir, filename)
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(filepath):
                    filepath = os.path.join(audio_dir, f"{base}_{counter}{ext}")
                    counter += 1

                final_filename = os.path.basename(filepath)

                with _download_lock:
                    _download_progress[url].filename = final_filename

                # Download with progress tracking
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if _download_cancelled.is_set():
                            f.close()
                            os.remove(filepath)
                            return AudioFile(filename=final_filename, path="", status=AudioStatus.ERROR, error_message="Cancelled")

                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = 5.0 + (downloaded / total_size) * 90.0
                                with _download_lock:
                                    _download_progress[url].progress = min(progress, 95.0)

                with _download_lock:
                    _download_progress[url].status = "completed"
                    _download_progress[url].progress = 100.0

                if progress_callback:
                    progress_callback(url, "completed", final_filename)

                return AudioFile(
                    filename=final_filename,
                    path=filepath,
                    size_bytes=os.path.getsize(filepath),
                    status=AudioStatus.DOWNLOADED
                )

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries and not _download_cancelled.is_set():
                    import time
                    wait_seconds = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    with _download_lock:
                        _download_progress[url].status = "retrying"
                        _download_progress[url].error_message = f"失败，{wait_seconds}秒后重试 ({attempt + 1}/{max_retries}): {last_error[:100]}"
                    time.sleep(wait_seconds)
                    continue
                break

        with _download_lock:
            _download_progress[url].status = "error"
            _download_progress[url].error_message = last_error or "下载失败"
        if progress_callback:
            progress_callback(url, "error", last_error or "下载失败")
        return AudioFile(filename="", path="", status=AudioStatus.ERROR, error_message=last_error or "下载失败")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_download_one, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result.filename:
                results.append(result)

    return results


def cancel_downloads():
    """Cancel all ongoing downloads."""
    _download_cancelled.set()


def get_download_progress(url: Optional[str] = None) -> dict:
    """Get download progress for all or a specific URL."""
    with _download_lock:
        if url:
            return _download_progress.get(url, None)
        return {k: v.model_dump() for k, v in _download_progress.items()}


def list_audio_files(base_dir: str = "./data/audio") -> List[AudioFile]:
    """List all audio files in the audio directory."""
    audio_dir = get_audio_dir(base_dir)
    files = []

    if not os.path.exists(audio_dir):
        return files

    for fname in sorted(os.listdir(audio_dir)):
        fpath = os.path.join(audio_dir, fname)
        if os.path.isfile(fpath):
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(AudioFile(
                    filename=fname,
                    path=fpath,
                    size_bytes=os.path.getsize(fpath),
                    status=AudioStatus.DOWNLOADED
                ))

    return files


def delete_audio_file(filename: str, base_dir: str = "./data/audio") -> bool:
    """Delete a specific audio file."""
    audio_dir = get_audio_dir(base_dir)
    filepath = os.path.join(audio_dir, os.path.basename(filename))

    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def delete_all_audio_files(base_dir: str = "./data/audio") -> int:
    """Delete all audio files. Returns count of deleted files."""
    audio_dir = get_audio_dir(base_dir)
    count = 0
    if os.path.exists(audio_dir):
        for fname in os.listdir(audio_dir):
            fpath = os.path.join(audio_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                count += 1
    return count


def get_audio_file_info(filename: str, base_dir: str = "./data/audio") -> Optional[AudioFile]:
    """Get metadata for a specific audio file."""
    audio_dir = get_audio_dir(base_dir)
    filepath = os.path.join(audio_dir, os.path.basename(filename))

    if not os.path.exists(filepath):
        return None

    return AudioFile(
        filename=filename,
        path=filepath,
        size_bytes=os.path.getsize(filepath),
        status=AudioStatus.DOWNLOADED
    )