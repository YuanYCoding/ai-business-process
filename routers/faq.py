"""
FAQ extraction API routes.
"""

import os
import json
import threading
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

from models.schemas import (
    ASRResult, LLMConfig, MessageResponse
)
from services import asr_service, llm_service
from services.faq_service import extract_faqs, generate_faq_excel

router = APIRouter(prefix="/api/faq", tags=["faq"])

# FAQ storage
_faq_results = {}
_faq_lock = threading.Lock()


@router.post("/extract", response_model=MessageResponse)
async def extract_faq_from_asr(
    use_all_asr: bool = Query(default=True),
    asr_filenames: Optional[str] = Query(default=None, description="Comma-separated ASR output filenames"),
):
    """Extract FAQ from ASR results."""
    # Get ASR results
    if use_all_asr:
        all_results = asr_service.list_asr_results("./data/asr_output")
    elif asr_filenames:
        filenames = [f.strip() for f in asr_filenames.split(",") if f.strip()]
        all_results = []
        for fn in filenames:
            result = asr_service.get_asr_result(fn, "./data/asr_output")
            if result:
                all_results.append(result)
    else:
        all_results = asr_service.list_asr_results("./data/asr_output")

    if not all_results:
        return MessageResponse(success=False, message="没有找到ASR解析结果，请先进行ASR解析")

    # Convert to ASRResult objects
    asr_results = []
    for r in all_results:
        if isinstance(r, dict):
            asr_results.append(ASRResult(**r))
        else:
            asr_results.append(r)

    # Build LLM config
    config = LLMConfig(
        provider="deepseek",
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    def _extract_bg():
        try:
            faqs = extract_faqs(asr_results, config)
            with _faq_lock:
                _faq_results["latest"] = faqs
        except Exception as e:
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_extract_bg, daemon=True)
    thread.start()

    return MessageResponse(
        success=True,
        message=f"FAQ提取已启动，正在分析 {len(asr_results)} 个ASR结果",
        data={"audio_count": len(asr_results)}
    )


@router.get("/results", response_model=dict)
async def get_faq_results():
    """Get latest FAQ extraction results."""
    with _faq_lock:
        faqs = _faq_results.get("latest", [])
    return {"faqs": faqs, "count": len(faqs)}


@router.get("/download-excel")
async def download_faq_excel():
    """Download FAQ as Excel file."""
    with _faq_lock:
        faqs = _faq_results.get("latest", [])

    if not faqs:
        raise HTTPException(status_code=404, detail="暂无FAQ数据，请先提取FAQ")

    output_path = "./data/faq_output.xlsx"
    os.makedirs("./data", exist_ok=True)
    result_path = generate_faq_excel(faqs, output_path)

    if result_path.endswith('.csv'):
        return FileResponse(result_path, filename="FAQ.csv", media_type="text/csv")
    return FileResponse(result_path, filename="FAQ.xlsx",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")