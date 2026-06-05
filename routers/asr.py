"""
ASR processing API routes.
"""

import os
import json
import threading
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from models.schemas import (
    ASRConfig, ASRResult, ASRProgressMessage, MessageResponse
)
from services import asr_service

router = APIRouter(prefix="/api/asr", tags=["asr"])


@router.get("/config", response_model=ASRConfig)
async def get_asr_config():
    """Get current ASR configuration."""
    return asr_service.get_config()


@router.put("/config", response_model=ASRConfig)
async def update_asr_config(config: ASRConfig):
    """Update ASR configuration."""
    # Update the global processing state config
    with asr_service._state_lock:
        asr_service._processing_state["config"] = config.model_dump()
    return config


@router.post("/start", response_model=MessageResponse)
async def start_asr_processing(
    audio_dir: str = Query(default="./data/audio", description="Audio files directory"),
    specific_files: Optional[str] = Query(default=None, description="Comma-separated list of specific files to process")
):
    """Start ASR processing for all audio files."""
    state = asr_service.get_processing_state()
    if state["is_running"]:
        return MessageResponse(
            success=False,
            message="ASR处理正在进行中，请等待完成后再开始新的处理"
        )

    config = asr_service.get_config()
    config.output_dir = asr_service.get_config().output_dir or "./data/asr_output"

    # Parse specific files
    file_list = None
    if specific_files:
        file_list = [f.strip() for f in specific_files.split(",") if f.strip()]

    # Start processing in background thread
    def _process_bg():
        try:
            asr_service.process_batch(
                audio_dir=audio_dir,
                config=config,
                specific_files=file_list
            )
        except Exception as e:
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_process_bg, daemon=True)
    thread.start()

    return MessageResponse(
        success=True,
        message="ASR处理已启动",
        data={"audio_dir": audio_dir, "config": config.model_dump()}
    )


@router.get("/status", response_model=dict)
async def get_asr_status():
    """Get current ASR processing status."""
    state = asr_service.get_processing_state()
    return {
        "is_running": state["is_running"],
        "current": state["current"],
        "total": state["total"],
        "progress": round((state["current"] / max(state["total"], 1)) * 100, 1),
        "config": state["config"],
    }


@router.post("/cancel", response_model=MessageResponse)
async def cancel_asr_processing():
    """Cancel ongoing ASR processing."""
    asr_service.cancel_processing()
    return MessageResponse(success=True, message="已请求取消ASR处理")


@router.get("/results", response_model=List[ASRResult])
async def list_asr_results(
    output_dir: str = Query(default="./data/asr_output")
):
    """List all ASR results."""
    return asr_service.list_asr_results(output_dir)


@router.get("/results/{filename}", response_model=ASRResult)
async def get_asr_result(
    filename: str,
    output_dir: str = Query(default="./data/asr_output")
):
    """Get ASR result for a specific file."""
    result = asr_service.get_asr_result(filename, output_dir)
    if not result:
        raise HTTPException(status_code=404, detail=f"未找到ASR结果: {filename}")
    return result


@router.get("/view/{filename}", response_model=dict)
async def view_asr_text(
    filename: str,
    output_dir: str = Query(default="./data/asr_output")
):
    """Get ASR result as formatted text."""
    result = asr_service.get_asr_result(filename, output_dir)
    if not result:
        raise HTTPException(status_code=404, detail=f"未找到ASR结果: {filename}")

    # Format as display text
    lines = []
    for sent in result.sentences:
        speaker_prefix = f"[{sent.speaker}] " if sent.speaker else ""
        lines.append(f"第{sent.index}句话：{speaker_prefix}{sent.text}")

    return {
        "filename": result.filename,
        "status": result.status,
        "formatted_text": "\n".join(lines),
        "sentences": [s.model_dump() for s in result.sentences],
        "sentence_count": len(result.sentences),
    }