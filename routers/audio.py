"""
Audio management API routes.
"""

import os
import json
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from models.schemas import (
    AudioFile, AudioListResponse, DownloadURLsRequest, DownloadProgress,
    MessageResponse, AudioStatus
)
from services import audio_service

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/upload", response_model=List[AudioFile])
async def upload_audio_files(files: List[UploadFile] = File(...)):
    """Upload audio files."""
    if not files:
        raise HTTPException(status_code=400, detail="请选择文件上传")

    results = audio_service.save_uploaded_files(files)
    return results


@router.post("/download-urls", response_model=MessageResponse)
async def download_audio_from_urls(request: DownloadURLsRequest):
    """Start downloading audio files from URLs (runs in background)."""
    import threading

    if not request.urls:
        raise HTTPException(status_code=400, detail="请输入至少一个URL")

    # Start download in background thread
    def _download_bg():
        audio_service.download_audio_from_urls(
            urls=request.urls,
            max_workers=request.max_workers
        )

    thread = threading.Thread(target=_download_bg, daemon=True)
    thread.start()

    return MessageResponse(
        success=True,
        message=f"已开始下载 {len(request.urls)} 个文件，使用 {request.max_workers} 个线程",
        data={"url_count": len(request.urls)}
    )


@router.get("/download-progress", response_model=dict)
async def get_download_progress():
    """Get current download progress."""
    return audio_service.get_download_progress()


@router.post("/cancel-downloads", response_model=MessageResponse)
async def cancel_downloads():
    """Cancel all ongoing downloads."""
    audio_service.cancel_downloads()
    return MessageResponse(success=True, message="已请求取消所有下载")


@router.get("/list", response_model=AudioListResponse)
async def list_audio_files():
    """List all audio files."""
    files = audio_service.list_audio_files()
    return AudioListResponse(
        files=files,
        total_count=len(files)
    )


@router.get("/{filename}", response_model=AudioFile)
async def get_audio_file(filename: str):
    """Get audio file metadata."""
    file_info = audio_service.get_audio_file_info(filename)
    if not file_info:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    return file_info


@router.delete("/{filename}", response_model=MessageResponse)
async def delete_audio_file(filename: str):
    """Delete an audio file."""
    success = audio_service.delete_audio_file(filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    return MessageResponse(success=True, message=f"已删除: {filename}")


@router.delete("/all/delete", response_model=MessageResponse)
async def delete_all_audio_files():
    """Delete all audio files."""
    count = audio_service.delete_all_audio_files()
    return MessageResponse(success=True, message=f"已删除 {count} 个文件")