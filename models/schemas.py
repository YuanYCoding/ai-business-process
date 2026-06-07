"""
Pydantic models for the AI Business Process Analysis Service.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ==================== Audio Models ====================

class AudioStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class AudioFile(BaseModel):
    filename: str
    path: str
    size_bytes: int = 0
    duration_seconds: Optional[float] = None
    status: AudioStatus = AudioStatus.PENDING
    error_message: Optional[str] = None


class AudioListResponse(BaseModel):
    files: List[AudioFile]
    total_count: int
    supported_formats: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"]


class DownloadURLsRequest(BaseModel):
    urls: List[str]
    max_workers: int = Field(default=4, ge=1, le=20)
    max_retries: int = Field(default=3, ge=0, le=10)


class DownloadProgress(BaseModel):
    url: str
    filename: str
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    status: str = "pending"  # pending, downloading, completed, error
    error_message: Optional[str] = None


# ==================== ASR Models ====================

class ASRConfig(BaseModel):
    language: str = Field(default="zh", description="ASR recognition language")
    output_dir: str = Field(default="./data/asr_output", description="ASR output directory")
    max_workers: int = Field(default=4, ge=1, le=20, description="Parallel processing workers")
    use_itn: bool = Field(default=True, description="Use inverse text normalization")
    batch_size_s: int = Field(default=60, ge=10, le=300, description="Batch size in seconds")


class ASRSentence(BaseModel):
    index: int
    text: str
    speaker: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None


class ASRResult(BaseModel):
    filename: str
    status: str = "pending"  # pending, processing, completed, error
    sentences: List[ASRSentence] = []
    full_text: str = ""
    error_message: Optional[str] = None
    processed_at: Optional[str] = None


class ASRProgressMessage(BaseModel):
    type: str  # "progress", "file_complete", "all_complete", "error", "model_downloading", "model_loaded"
    filename: Optional[str] = None
    current: int = 0
    total: int = 0
    message: str = ""
    result: Optional[ASRResult] = None
    model_progress: float = Field(default=0.0, ge=0.0, le=100.0)


# ==================== LLM Models ====================

class LLMProvider(BaseModel):
    name: str
    display_name: str
    base_url: str
    models: List[str]
    api_key_env: str


class LLMConfig(BaseModel):
    provider: str = Field(default="deepseek")
    model: str = Field(default="deepseek-v4-pro")
    api_key: str = ""
    base_url: str = Field(default="https://api.deepseek.com")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=65536)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class LLMTestResult(BaseModel):
    success: bool
    message: str
    latency_ms: float = 0.0
    model: str = ""
    tokens_used: int = 0


# ==================== Business Process Models ====================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class BusinessTaskStatus(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: str = ""
    error_message: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None


class BusinessProcessResult(BaseModel):
    task_id: str
    markdown: str = ""
    xmind_path: Optional[str] = None
    drawio_path: Optional[str] = None
    mermaid_syntax: Optional[str] = None
    echarts_html_path: Optional[str] = None
    summary_html_path: Optional[str] = None
    visualization_paths: Dict[str, str] = {}


class BusinessProcessListItem(BaseModel):
    task_id: str
    created_at: str
    audio_count: int
    status: str


# ==================== Prompt Models ====================

class PromptCategory(str, Enum):
    BUSINESS_ANALYSIS = "business_analysis"
    BUSINESS_FLOW = "business_flow"
    SCRIPT_GENERATION = "script_generation"
    VISUALIZATION = "visualization"


class PromptTemplate(BaseModel):
    name: str
    display_name: str
    category: PromptCategory
    content: str
    is_customized: bool = False
    description: str = ""


class PromptListResponse(BaseModel):
    prompts: List[PromptTemplate]


# ==================== Generic Models ====================

class MessageResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class GenerateBusinessRequest(BaseModel):
    use_all_asr: bool = True
    asr_filenames: Optional[List[str]] = None
    llm_config: Optional[LLMConfig] = None