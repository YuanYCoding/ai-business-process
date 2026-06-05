"""
Business process analysis service.
Orchestrates LLM calls to analyze ASR transcripts and generate
structured business process documents following the reference format.
"""

import os
import re
import json
import time
import uuid
import threading
from typing import List, Dict, Optional, Callable

from models.schemas import (
    ASRResult, LLMConfig, BusinessTaskStatus, BusinessProcessResult,
    TaskStatus, BusinessProcessListItem
)
from services.llm_service import call_llm, call_llm_with_json_output, extract_json_from_text
from services.prompt_service import get_prompt

# Task storage
_tasks: Dict[str, BusinessTaskStatus] = {}
_results: Dict[str, BusinessProcessResult] = {}
_tasks_lock = threading.Lock()


def _create_task_id() -> str:
    """Create a unique task ID."""
    return f"bp_{uuid.uuid4().hex[:12]}"


def create_task(audio_count: int) -> BusinessTaskStatus:
    """Create a new business process generation task."""
    task_id = _create_task_id()
    task = BusinessTaskStatus(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        current_step="准备中...",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with _tasks_lock:
        _tasks[task_id] = task
    return task


def update_task_status(task_id: str, **kwargs):
    """Update task status."""
    with _tasks_lock:
        if task_id in _tasks:
            for key, value in kwargs.items():
                if hasattr(_tasks[task_id], key):
                    setattr(_tasks[task_id], key, value)


def get_task_status(task_id: str) -> Optional[BusinessTaskStatus]:
    """Get task status by ID."""
    with _tasks_lock:
        return _tasks.get(task_id)


def get_task_result(task_id: str) -> Optional[BusinessProcessResult]:
    """Get task result by ID."""
    with _tasks_lock:
        return _results.get(task_id)


def list_tasks() -> List[BusinessProcessListItem]:
    """List all business process tasks."""
    with _tasks_lock:
        items = []
        for task_id, task in _tasks.items():
            result = _results.get(task_id)
            items.append(BusinessProcessListItem(
                task_id=task_id,
                created_at=task.created_at,
                audio_count=0,  # Will be set from generation context
                status=task.status.value
            ))
        return sorted(items, key=lambda x: x.created_at, reverse=True)


def _build_combined_transcript(asr_results: List[ASRResult]) -> str:
    """Build a combined transcript from all ASR results."""
    parts = []
    for result in asr_results:
        audio_name = os.path.splitext(result.filename)[0]
        parts.append(f"\n=== 音频文件: {audio_name} ===\n")
        for sent in result.sentences:
            speaker_prefix = f"[{sent.speaker}] " if sent.speaker else ""
            parts.append(f"第{sent.index}句话：{speaker_prefix}{sent.text}")
        parts.append("")

    return "\n".join(parts)


def extract_markdown_from_text(text: str) -> str:
    """Extract markdown content from LLM response."""
    # Try to extract from markdown code block
    md_match = re.search(r'```(?:markdown|md)?\s*\n(.*?)\n```', text, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    # If the text starts with #, it's likely already markdown
    if text.strip().startswith('#'):
        return text.strip()

    return text.strip()


def generate_business_process(
    asr_results: List[ASRResult],
    llm_config: LLMConfig,
    output_dir: str = "./data/business_output",
    progress_callback: Optional[Callable] = None
) -> BusinessProcessResult:
    """
    Generate business process document from ASR results.
    This is the main orchestration function.
    """
    task_id = _create_task_id()
    os.makedirs(output_dir, exist_ok=True)

    # Initialize task
    task = BusinessTaskStatus(
        task_id=task_id,
        status=TaskStatus.RUNNING,
        progress=0.0,
        current_step="开始分析...",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with _tasks_lock:
        _tasks[task_id] = task

    try:
        # Step 1: Build combined transcript
        update_task_status(task_id, current_step="正在整合所有ASR文本...", progress=5.0)
        if progress_callback:
            progress_callback(task_id, "整合ASR文本", 5.0)

        combined_transcript = _build_combined_transcript(asr_results)

        # Step 2: Analyze business context
        update_task_status(task_id, current_step="正在分析业务上下文...", progress=15.0)
        if progress_callback:
            progress_callback(task_id, "分析业务上下文", 15.0)

        analysis_prompt = get_prompt("business_analysis")
        context_result = call_llm(
            messages=[{"role": "user", "content": analysis_prompt.format(
                transcripts=combined_transcript
            )}],
            config=llm_config
        )

        # Step 3: Generate business flow markdown
        update_task_status(task_id, current_step="正在生成业务流程文档...", progress=40.0)
        if progress_callback:
            progress_callback(task_id, "生成业务流程", 40.0)

        flow_prompt = get_prompt("business_flow")
        flow_result = call_llm(
            messages=[
                {"role": "user", "content": f"以下是业务上下文分析结果：\n\n{context_result}\n\n"}
            ],
            config=llm_config,
            system_prompt=flow_prompt
        )
        markdown_content = extract_markdown_from_text(flow_result)

        # Step 4: Generate structured data for visualization
        update_task_status(task_id, current_step="正在生成结构化数据...", progress=65.0)
        if progress_callback:
            progress_callback(task_id, "生成结构化数据", 65.0)

        viz_prompt = get_prompt("visualization")
        # Use replace instead of format to avoid KeyError from JSON curly braces in prompt
        viz_prompt_filled = viz_prompt.replace('{markdown}', markdown_content)
        structured_result = call_llm_with_json_output(
            messages=[{"role": "user", "content": viz_prompt_filled}],
            config=llm_config
        )
        structured_data = extract_json_from_text(structured_result) or {}

        # Step 5: Save markdown file
        update_task_status(task_id, current_step="正在保存文档...", progress=85.0)
        if progress_callback:
            progress_callback(task_id, "保存文档", 85.0)

        md_path = os.path.join(output_dir, f"{task_id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Save structured data
        json_path = os.path.join(output_dir, f"{task_id}_structured.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)

        # Build result
        result = BusinessProcessResult(
            task_id=task_id,
            markdown=markdown_content,
            visualization_paths={
                "markdown": md_path,
                "structured_json": json_path,
            }
        )

        with _tasks_lock:
            _results[task_id] = result

        update_task_status(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            current_step="完成！",
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        if progress_callback:
            progress_callback(task_id, "完成", 100.0)

        return result

    except Exception as e:
        update_task_status(
            task_id,
            status=TaskStatus.ERROR,
            error_message=str(e),
            current_step=f"错误: {str(e)}"
        )
        if progress_callback:
            progress_callback(task_id, f"错误: {str(e)}", -1)
        raise


def generate_structured_data_from_markdown(
    markdown_content: str,
    llm_config: LLMConfig
) -> Dict:
    """Generate structured JSON data from markdown for visualization."""
    viz_prompt = get_prompt("visualization")
    viz_prompt_filled = viz_prompt.replace('{markdown}', markdown_content)
    result = call_llm_with_json_output(
        messages=[{"role": "user", "content": viz_prompt_filled}],
        config=llm_config
    )
    return extract_json_from_text(result) or {}