"""
LLM (Large Language Model) service.
Provides abstraction over multiple LLM providers with connection testing.
Reuses patterns from prompt_test_sample/main.py
"""

import os
import time
import json
from typing import List, Dict, Optional, AsyncGenerator, Any

from openai import OpenAI
from models.schemas import LLMConfig, LLMTestResult, LLMProvider

# ==================== Provider Presets ====================

PROVIDER_PRESETS: Dict[str, LLMProvider] = {
    "deepseek": LLMProvider(
        name="deepseek",
        display_name="DeepSeek",
        base_url="https://api.deepseek.com",
        models=["deepseek-v4-pro", "deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        api_key_env="DEEPSEEK_API_KEY"
    ),
    "openai": LLMProvider(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        api_key_env="OPENAI_API_KEY"
    ),
    "ollama": LLMProvider(
        name="ollama",
        display_name="Ollama (本地)",
        base_url="http://localhost:11434/v1",
        models=["qwen3:4b-instruct", "qwen2.5:7b", "llama3:8b", "deepseek-r1:8b"],
        api_key_env="OLLAMA_API_KEY"
    ),
    "siliconflow": LLMProvider(
        name="siliconflow",
        display_name="SiliconFlow (硅基流动)",
        base_url="https://api.siliconflow.cn/v1",
        models=["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct"],
        api_key_env="SILICONFLOW_API_KEY"
    ),
    "custom": LLMProvider(
        name="custom",
        display_name="自定义 (OpenAI兼容)",
        base_url="http://localhost:8080/v1",
        models=["custom-model"],
        api_key_env="CUSTOM_API_KEY"
    ),
}


def get_provider_presets() -> Dict[str, LLMProvider]:
    """Get all provider presets."""
    return PROVIDER_PRESETS


def get_provider(name: str) -> Optional[LLMProvider]:
    """Get a specific provider preset."""
    return PROVIDER_PRESETS.get(name)


def _build_client(config: LLMConfig) -> OpenAI:
    """Build an OpenAI-compatible client from config."""
    api_key = config.api_key
    if not api_key:
        # Try to get from environment
        provider = PROVIDER_PRESETS.get(config.provider)
        if provider:
            api_key = os.getenv(provider.api_key_env, "")
    if not api_key:
        api_key = os.getenv("LLM_API_KEY", "sk-placeholder")

    return OpenAI(
        api_key=api_key,
        base_url=config.base_url
    )


def test_connection(config: LLMConfig) -> LLMTestResult:
    """Test connection to an LLM provider."""
    try:
        client = _build_client(config)
        start_time = time.time()

        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "user", "content": "请回复：连接测试成功"}
            ],
            max_tokens=50,
            temperature=0.0
        )

        latency_ms = (time.time() - start_time) * 1000
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        return LLMTestResult(
            success=True,
            message=f"连接成功！回复: {content}",
            latency_ms=round(latency_ms, 2),
            model=config.model,
            tokens_used=tokens_used
        )

    except Exception as e:
        return LLMTestResult(
            success=False,
            message=f"连接失败: {str(e)}",
            model=config.model
        )


def call_llm(
    messages: List[Dict[str, str]],
    config: LLMConfig,
    system_prompt: Optional[str] = None
) -> str:
    """Call LLM with messages and return response text."""
    client = _build_client(config)

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    response = client.chat.completions.create(
        model=config.model,
        messages=full_messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    return response.choices[0].message.content or ""


def call_llm_with_json_output(
    messages: List[Dict[str, str]],
    config: LLMConfig,
    system_prompt: Optional[str] = None
) -> str:
    """Call LLM and request JSON output."""
    client = _build_client(config)

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    # Add JSON instruction to the last user message if not present
    if full_messages and full_messages[-1]["role"] == "user":
        if "json" not in full_messages[-1]["content"].lower():
            full_messages[-1]["content"] += "\n\n请以JSON格式输出结果。"

    response = client.chat.completions.create(
        model=config.model,
        messages=full_messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content or "{}"


async def call_llm_streaming(
    messages: List[Dict[str, str]],
    config: LLMConfig,
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Call LLM with streaming response."""
    client = _build_client(config)

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    stream = client.chat.completions.create(
        model=config.model,
        messages=full_messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    import re
    # ```json ... ```
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None