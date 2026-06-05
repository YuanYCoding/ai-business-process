"""
Prompt management service.
File-based prompt template storage with user customization support.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

from models.schemas import PromptTemplate, PromptCategory

# ==================== Default Prompts ====================

DEFAULT_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
USER_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prompts")

# Default prompt contents
DEFAULT_PROMPTS: Dict[str, Dict] = {
    "business_analysis": {
        "display_name": "业务上下文分析",
        "category": PromptCategory.BUSINESS_ANALYSIS,
        "description": "分析所有ASR转录文本，识别业务类型、对话角色、关键话题和流程",
        "content": """你是一个专业的业务分析专家。请仔细阅读以下电话录音的ASR转录文本，完成以下分析任务：

## 转录文本
{transcripts}

## 分析任务
请从以下维度进行分析：

1. **业务类型识别**：这是什么类型的业务电话？（如：车贷电话销售、保险回访、客服咨询等）
2. **对话角色识别**：对话中有哪些角色？（如：AI外呼机器人、客户、业务专员等）
3. **核心业务流程**：总结对话的主要流程阶段（如：开场白→需求确认→资质审核→产品介绍→转人工等）
4. **关键话术提取**：提取每个阶段的核心话术和标准问答
5. **分支逻辑**：识别对话中的条件分支（如：客户有意向/无意向，有车/无车等）
6. **变量识别**：识别话术中需要动态替换的变量（如：金额、利率、期限等）

请以结构化的方式输出分析结果，便于后续生成业务流程文档。"""
    },
    "business_flow": {
        "display_name": "业务流程生成",
        "category": PromptCategory.BUSINESS_FLOW,
        "description": "根据业务分析结果生成标准格式的业务流程Markdown文档",
        "content": """你是一个专业的业务流程文档撰写专家。请根据提供的业务上下文分析结果，生成标准格式的业务流程Markdown文档。

## 参考格式

请严格参照以下格式生成文档：

```markdown
# [业务名称]业务流程

## Q1：[开场白话术]

### 客户表达有意向（"关键词1"/"关键词2"/"关键词3"）

- Q2：话术：[产品介绍话术]

  - 客户确认[条件A]

    - Q3：话术：[进一步确认话术]

      - [分支条件1]

        - Q4-分支1：话术：[对应话术]

          - [子分支...]

      - [分支条件2]

        - Q4-分支2：话术：[对应话术]

### 客户直接询问[某信息]

- [信息类型1]相关询问

  - 话术：[回答话术]

    - （回到Q2流程）

- [信息类型2]相关询问

  - 话术：[回答话术]

### 客户表达[拒绝/负面情况]

- 话术：[处理话术]

### AI无法回答客户的复杂问题（兜底话术）

- 话术：[兜底话术]

  - 客户确认有意向

    - 话术：[转接话术]

      - 【系统执行转人工操作】

  - 客户无意愿

    - 话术：[结束话术]

### 客户情绪激动/明确拒绝转接/投诉倾向

- 话术：[结束话术]
```

## 格式要求

1. 使用层级标题（#, ##, ###, ####）和列表（-）构建树形结构
2. 每个节点包含：问题编号（Q1, Q2...）、话术内容、分支条件
3. 话术内容使用"话术："前缀
4. 条件分支使用"客户表达XXX"、"客户确认XXX"等形式
5. 支持流程引用，如"（回到Q2流程）"、"（转到Q6-转接确认流程）"
6. 系统操作使用【】标注，如【系统执行转人工操作】
7. 变量使用花括号，如{金额}、{期限}
8. 确保话术口语化、自然、符合电话沟通场景
9. 覆盖所有主要的对话分支路径
10. 包含兜底话术和异常处理流程

请根据提供的业务分析结果，生成完整的业务流程Markdown文档。"""
    },
    "script_generation": {
        "display_name": "话术脚本生成",
        "category": PromptCategory.SCRIPT_GENERATION,
        "description": "根据业务流程生成详细的话术脚本",
        "content": """你是一个专业的话术脚本撰写专家。请根据提供的业务流程，生成详细的话术脚本。

## 要求
1. 话术要自然、口语化，符合真实电话沟通场景
2. 每个节点的"话术："后面应该是一段完整的对话文本
3. 包含适当的过渡语和礼貌用语
4. 考虑不同客户反应的处理方式
5. 变量处使用{变量名}标记

请基于以下业务流程生成完整的话术脚本：
{flow_markdown}"""
    },
    "visualization": {
        "display_name": "可视化数据提取",
        "category": PromptCategory.VISUALIZATION,
        "description": "将业务流程Markdown转换为结构化JSON数据用于可视化",
        "content": """你是一个数据结构化专家。请将以下业务流程Markdown文档转换为结构化的JSON树形数据。

## 输入
{markdown}

## 输出格式
请输出以下JSON格式：

```json
{{
  "root": {{
    "title": "根节点标题（业务名称）",
    "type": "start",
    "children": [
      {{
        "title": "节点标题（如：Q1: 开场白话术内容前20字）",
        "type": "question|script|decision|action|end",
        "condition": "分支条件（可选）",
        "description": "完整话术或说明",
        "children": [...]
      }}
    ]
  }}
}}
```

## 节点类型说明
- "start": 起始节点（根节点）
- "question": 问题节点（AI提问）
- "script": 话术节点（包含话术内容）
- "decision": 决策节点（客户选择分支）
- "action": 动作节点（如转人工）
- "end": 结束节点

## 规则
1. 每个节点必须有title字段
2. type字段根据节点内容判断
3. 分支节点使用condition字段记录分支条件
4. 话术节点使用description字段记录完整话术
5. 保持Markdown文档的层级结构
6. 确保JSON格式正确，可以被直接解析

请输出完整的JSON结构。"""
    }
}


def _ensure_dirs():
    """Ensure prompt directories exist."""
    os.makedirs(DEFAULT_PROMPTS_DIR, exist_ok=True)
    os.makedirs(USER_PROMPTS_DIR, exist_ok=True)


def _init_default_prompts():
    """Initialize default prompt files if they don't exist."""
    _ensure_dirs()
    for name, info in DEFAULT_PROMPTS.items():
        path = os.path.join(DEFAULT_PROMPTS_DIR, f"{name}.txt")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(info["content"])


def get_prompt(name: str) -> str:
    """Get prompt content. Returns user-customized version if available, otherwise default."""
    _init_default_prompts()

    # Check user-customized version first
    user_path = os.path.join(USER_PROMPTS_DIR, f"{name}.txt")
    if os.path.exists(user_path):
        with open(user_path, "r", encoding="utf-8") as f:
            return f.read()

    # Fall back to default
    default_path = os.path.join(DEFAULT_PROMPTS_DIR, f"{name}.txt")
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            return f.read()

    # Fall back to hardcoded default
    if name in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[name]["content"]

    return ""


def save_prompt(name: str, content: str) -> bool:
    """Save user-customized prompt. Returns True if successful."""
    _ensure_dirs()
    user_path = os.path.join(USER_PROMPTS_DIR, f"{name}.txt")
    try:
        with open(user_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False


def reset_prompt(name: str) -> bool:
    """Reset a prompt to its default by deleting the user-customized version."""
    user_path = os.path.join(USER_PROMPTS_DIR, f"{name}.txt")
    try:
        if os.path.exists(user_path):
            os.remove(user_path)
        return True
    except Exception:
        return False


def list_prompts() -> List[PromptTemplate]:
    """List all available prompts."""
    _init_default_prompts()
    prompts = []

    for name, info in DEFAULT_PROMPTS.items():
        is_customized = os.path.exists(os.path.join(USER_PROMPTS_DIR, f"{name}.txt"))
        prompts.append(PromptTemplate(
            name=name,
            display_name=info["display_name"],
            category=info["category"],
            content=get_prompt(name),
            is_customized=is_customized,
            description=info["description"]
        ))

    return prompts


def get_prompt_info(name: str) -> Optional[PromptTemplate]:
    """Get info for a specific prompt."""
    if name not in DEFAULT_PROMPTS:
        return None

    info = DEFAULT_PROMPTS[name]
    is_customized = os.path.exists(os.path.join(USER_PROMPTS_DIR, f"{name}.txt"))

    return PromptTemplate(
        name=name,
        display_name=info["display_name"],
        category=info["category"],
        content=get_prompt(name),
        is_customized=is_customized,
        description=info["description"]
    )


# Initialize on import
_init_default_prompts()