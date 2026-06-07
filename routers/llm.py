"""
LLM model configuration API routes.
"""

import os
from fastapi import APIRouter, HTTPException
from models.schemas import LLMConfig, LLMTestResult, LLMProvider, MessageResponse
from services import llm_service

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/providers", response_model=dict)
async def get_providers():
    """List all available LLM providers."""
    providers = llm_service.get_provider_presets()
    return {
        name: {
            "name": p.name,
            "display_name": p.display_name,
            "base_url": p.base_url,
            "models": p.models,
            "api_key_env": p.api_key_env,
        }
        for name, p in providers.items()
    }


@router.get("/config", response_model=LLMConfig)
async def get_llm_config():
    """Get default LLM configuration."""
    return LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
    )


@router.put("/config", response_model=MessageResponse)
async def update_llm_config(config: LLMConfig):
    """Update LLM configuration (stored in memory for this session)."""
    # In a real app, this would persist to a config file
    # For now, we store it as module-level state
    llm_service._current_config = config
    return MessageResponse(
        success=True,
        message="配置已更新",
        data=config.model_dump()
    )


@router.post("/test", response_model=LLMTestResult)
async def test_connection(config: LLMConfig):
    """Test LLM connection with the given configuration."""
    return llm_service.test_connection(config)


@router.post("/test-default", response_model=LLMTestResult)
async def test_default_connection():
    """Test connection with default DeepSeek configuration."""
    config = LLMConfig(
        provider="deepseek",
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    return llm_service.test_connection(config)