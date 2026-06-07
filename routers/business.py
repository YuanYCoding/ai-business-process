"""
Business process generation API routes.
"""

import os
import json
import threading
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from models.schemas import (
    BusinessTaskStatus, BusinessProcessResult, BusinessProcessListItem,
    GenerateBusinessRequest, MessageResponse, ASRResult, LLMConfig
)
from services import business_service, asr_service, visualization_service, llm_service

router = APIRouter(prefix="/api/business", tags=["business"])


@router.post("/generate", response_model=MessageResponse)
async def generate_business_process(request: GenerateBusinessRequest):
    """Start generating business process from ASR results."""
    # Check if ASR results exist
    all_results = asr_service.list_asr_results("./data/asr_output")
    if not all_results:
        return MessageResponse(
            success=False,
            message="没有找到ASR解析结果，请先进行ASR解析"
        )

    # Select which ASR results to use
    if request.use_all_asr:
        selected_results = all_results
    elif request.asr_filenames:
        selected_results = [
            r for r in all_results
            if any(fn in r.filename for fn in request.asr_filenames)
        ]
    else:
        selected_results = all_results

    if not selected_results:
        return MessageResponse(
            success=False,
            message="没有找到匹配的ASR解析结果"
        )

    # Build LLM config
    config = request.llm_config or LLMConfig(
        provider="deepseek",
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    # Convert to ASRResult objects if needed
    if isinstance(selected_results[0], dict):
        asr_results = [ASRResult(**r) if isinstance(r, dict) else r for r in selected_results]
    else:
        asr_results = selected_results

    # Start generation in background thread
    def _generate_bg():
        try:
            result = business_service.generate_business_process(
                asr_results=asr_results,
                llm_config=config,
                output_dir="./data/business_output",
            )

            # Generate visualizations
            structured_path = result.visualization_paths.get("structured_json", "")
            structured_data = {}
            if structured_path and os.path.exists(structured_path):
                try:
                    with open(structured_path, "r", encoding="utf-8") as f:
                        structured_data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[Business] Failed to load structured data: {e}")
                    structured_data = {}

            # Ensure structured_data has a valid root
            if not isinstance(structured_data, dict) or not structured_data.get("root"):
                # Create a simple fallback structure from the markdown
                structured_data = {
                    "root": {
                        "title": result.markdown.strip().split('\n')[0].replace('# ', '') if result.markdown else "业务流程",
                        "type": "start",
                        "children": []
                    }
                }

            try:
                viz_paths = visualization_service.generate_all_visualizations(
                    structured_data=structured_data,
                    markdown=result.markdown,
                    output_dir="./data/business_output",
                    task_id=result.task_id
                )
                result.visualization_paths.update(viz_paths)

                # Also set mermaid_syntax for frontend
                if "mermaid_syntax" in viz_paths:
                    result.mermaid_syntax = viz_paths["mermaid_syntax"]
            except Exception as viz_err:
                print(f"[Business] Visualization generation failed: {viz_err}")
                import traceback
                traceback.print_exc()

            # Update result in storage
            with business_service._tasks_lock:
                business_service._results[result.task_id] = result

        except Exception as e:
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=_generate_bg, daemon=True)
    thread.start()

    # Wait a moment for task creation
    import time
    time.sleep(0.5)

    # Find the most recently created task
    tasks = business_service.list_tasks()
    if tasks:
        return MessageResponse(
            success=True,
            message=f"业务流程生成已启动，使用 {len(asr_results)} 个ASR结果",
            data={"task_id": tasks[0].task_id, "audio_count": len(asr_results)}
        )

    return MessageResponse(
        success=True,
        message="业务流程生成已启动",
        data={"audio_count": len(asr_results)}
    )


@router.get("/status/{task_id}", response_model=BusinessTaskStatus)
async def get_task_status(task_id: str):
    """Get business process generation task status."""
    status = business_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return status


@router.get("/result/{task_id}", response_model=BusinessProcessResult)
async def get_task_result(task_id: str):
    """Get business process generation result."""
    result = business_service.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"结果不存在: {task_id}")
    return result


@router.get("/download/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """Download a specific visualization file."""
    result = business_service.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"结果不存在: {task_id}")

    file_path = result.visualization_paths.get(file_type)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_type}")

    filename = os.path.basename(file_path)

    # Determine media type
    media_types = {
        "md": "text/markdown",
        "html": "text/html",
        "drawio": "application/xml",
        "json": "application/json",
        "xmind": "application/vnd.xmind.workbook",
    }

    ext = file_type.split(".")[-1] if "." in file_type else file_type
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


@router.get("/view/{task_id}/{file_type}")
async def view_file(task_id: str, file_type: str):
    """View a specific file content inline."""
    result = business_service.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"结果不存在: {task_id}")

    if file_type == "markdown":
        return {"content": result.markdown, "type": "text/markdown"}
    elif file_type == "mermaid":
        return {"content": result.mermaid_syntax, "type": "text/plain"}
    elif file_type == "echarts":
        path = result.visualization_paths.get("echarts")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return {"content": f.read(), "type": "text/html"}
    elif file_type == "summary":
        path = result.visualization_paths.get("summary")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return {"content": f.read(), "type": "text/html"}
    elif file_type == "drawio":
        path = result.visualization_paths.get("drawio")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return {"content": f.read(), "type": "text/xml"}
    elif file_type == "xmind":
        path = result.visualization_paths.get("xmind")
        if path and os.path.exists(path):
            return FileResponse(path, filename=os.path.basename(path),
                                media_type="application/vnd.xmind.workbook")

    raise HTTPException(status_code=404, detail=f"文件不存在: {file_type}")


@router.get("/list", response_model=List[BusinessProcessListItem])
async def list_business_processes():
    """List all generated business processes."""
    return business_service.list_tasks()


@router.get("/markdown/{task_id}")
async def get_markdown(task_id: str):
    """Get the markdown content of a business process."""
    result = business_service.get_task_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"结果不存在: {task_id}")
    return {"task_id": task_id, "markdown": result.markdown}