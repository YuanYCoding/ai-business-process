"""
Prompt management API routes.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from models.schemas import PromptTemplate, MessageResponse
from services import prompt_service

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


class SavePromptRequest(BaseModel):
    content: str


@router.get("/list", response_model=List[PromptTemplate])
async def list_prompts():
    """List all prompt templates."""
    return prompt_service.list_prompts()


@router.get("/{name}", response_model=PromptTemplate)
async def get_prompt(name: str):
    """Get a specific prompt template."""
    info = prompt_service.get_prompt_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"提示词不存在: {name}")
    return info


@router.put("/{name}", response_model=MessageResponse)
async def save_prompt(name: str, request: SavePromptRequest):
    """Save/update a prompt template."""
    info = prompt_service.get_prompt_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"提示词不存在: {name}")

    success = prompt_service.save_prompt(name, request.content)
    if not success:
        raise HTTPException(status_code=500, detail="保存失败")

    return MessageResponse(
        success=True,
        message=f"提示词 '{info.display_name}' 已保存",
        data={"name": name, "display_name": info.display_name}
    )


@router.post("/{name}/reset", response_model=MessageResponse)
async def reset_prompt(name: str):
    """Reset a prompt to its default value."""
    info = prompt_service.get_prompt_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"提示词不存在: {name}")

    success = prompt_service.reset_prompt(name)
    if not success:
        raise HTTPException(status_code=500, detail="重置失败")

    return MessageResponse(
        success=True,
        message=f"提示词 '{info.display_name}' 已重置为默认值",
        data={"name": name, "content": prompt_service.get_prompt(name)}
    )