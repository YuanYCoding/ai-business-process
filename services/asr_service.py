"""
ASR (Automatic Speech Recognition) service.
Uses SenseVoiceSmall via funasr for speech-to-text processing.
Reuses patterns from 鑫享车贷/录音/asr_batch.py
"""

import os
import re
import sys
import json
import time
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.schemas import ASRConfig, ASRResult, ASRSentence, ASRProgressMessage

# Supported audio formats
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".wmv", ".amr", ".webm"}

# ==================== Silent Logging Setup ====================
os.environ["MODELSCOPE_LOG_LEVEL"] = "40"

import logging
for _lib in ["funasr", "modelscope", "urllib3", "matplotlib", "PIL", "tensorflow",
              "transformers", "huggingface_hub", "sentence_transformers"]:
    logging.getLogger(_lib).setLevel(logging.ERROR)
logging.getLogger().handlers = [logging.NullHandler()]

import warnings
warnings.filterwarnings("ignore")


# Silent tqdm
class _SilentTqdm:
    def __init__(self, *args, **kwargs):
        self._it = iter(kwargs.get("iterable", args[0] if args else []))
        self.total = kwargs.get("total", 0)
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._it)
    def update(self, n=1):
        pass
    def close(self):
        pass
    def set_description(self, *a, **kw):
        pass
    @staticmethod
    def write(*a, **kw):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


# ==================== Global Model State ====================
_MODEL = None
_MODEL_LOCK = None
_MODEL_LOADING = False
_MODEL_LOADING_LOCK = threading.Lock()

# Model download progress
_model_download_state = {
    "is_downloading": False,
    "progress": 0.0,
    "message": "等待模型检查...",
    "start_time": 0.0,
}

# Processing state
_processing_state = {
    "is_running": False,
    "current": 0,
    "total": 0,
    "results": {},
    "config": None,
    "cancel_requested": False,
    "status_message": "等待开始...",
}
_state_lock = threading.Lock()


# Model cache path for SenseVoiceSmall (modelscope v1 format)
# Use a path without CJK characters to avoid encoding issues with C++ libs (sentencepiece)
_SAFE_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".model_cache")
os.environ["MODELSCOPE_CACHE"] = _SAFE_CACHE
MODEL_CACHE_DIR = os.path.join(_SAFE_CACHE, "hub/models/iic/SenseVoiceSmall")
MODEL_TEMP_DIR = os.path.join(_SAFE_CACHE, "hub/models/._____temp/iic/SenseVoiceSmall")
MODEL_EXPECTED_SIZE = 950 * 1024 * 1024  # ~950MB (model.pt ~900MB + campplus ~27MB + other files)


def _check_model_cached() -> bool:
    """Check if the SenseVoiceSmall model is already cached locally."""
    for check_dir in [MODEL_CACHE_DIR, MODEL_TEMP_DIR]:
        if os.path.exists(check_dir) and os.path.isdir(check_dir):
            try:
                for f in os.listdir(check_dir):
                    fp = os.path.join(check_dir, f)
                    if os.path.isfile(fp) and (f.endswith('.pt') or f == 'config.yaml' or f == 'configuration.json'):
                        return True
            except Exception:
                pass
    return False


def _walk_dir_size(base_dir: str) -> int:
    """Calculate total size of all files in a directory tree."""
    total = 0
    if not os.path.exists(base_dir):
        return 0
    try:
        for dirpath, dirnames, filenames in os.walk(base_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def clean_text(text: str) -> str:
    """Clean ASR output tags."""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_model(progress_callback=None):
    """Load the SenseVoiceSmall model (with download progress tracking)."""
    global _model_download_state

    # Check if model is already cached
    model_cached = _check_model_cached()

    if model_cached:
        # Model already cached - no download needed, just load into memory
        _model_download_state = {
            "is_downloading": False,
            "progress": 100.0,
            "message": "模型已缓存，正在加载到内存...",
            "start_time": time.time(),
        }
        if progress_callback:
            progress_callback(ASRProgressMessage(
                type="model_loaded",
                message="模型已缓存，正在加载到内存...",
                model_progress=100.0
            ))
    else:
        # Model needs to be downloaded - start progress monitor
        _model_download_state = {
            "is_downloading": True,
            "progress": 0.0,
            "message": "正在下载ASR模型（SenseVoiceSmall），首次使用约需下载900MB...",
            "start_time": time.time(),
        }
        if progress_callback:
            progress_callback(ASRProgressMessage(
                type="model_downloading",
                message="正在下载ASR模型（SenseVoiceSmall）...",
                model_progress=0.0
            ))

        # Monitor thread: watches cache directories for file growth
        def _monitor_download():
            last_size = 0
            stall_count = 0
            while _model_download_state["is_downloading"]:
                total_size = 0
                for check_path in [MODEL_CACHE_DIR, MODEL_TEMP_DIR]:
                    total_size += _walk_dir_size(check_path)

                elapsed = time.time() - _model_download_state["start_time"]

                if total_size > 0:
                    progress = min(total_size / MODEL_EXPECTED_SIZE * 100, 99.0)
                    _model_download_state["progress"] = progress
                    _model_download_state["message"] = (
                        f"正在下载模型... {total_size / (1024*1024):.0f}MB / ~900MB "
                        f"({progress:.0f}%, 已用时 {elapsed:.0f}s)"
                    )
                    if total_size > last_size:
                        last_size = total_size
                        stall_count = 0
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
                    _model_download_state["message"] = (
                        f"正在连接下载服务器... (已等待 {elapsed:.0f}s)"
                    )

                if progress_callback:
                    progress_callback(ASRProgressMessage(
                        type="model_downloading",
                        message=_model_download_state["message"],
                        model_progress=_model_download_state["progress"]
                    ))
                time.sleep(0.8)

        monitor_thread = threading.Thread(target=_monitor_download, daemon=True)
        monitor_thread.start()

    # Load model (download happens automatically if not cached)
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        from funasr import AutoModel
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            spk_model="cam++",
            disable_update=True,
        )
        return model
    finally:
        sys.stdout.close()
        sys.stdout, sys.stderr = _stdout, _stderr

        # Stop download monitoring
        _model_download_state["is_downloading"] = False
        _model_download_state["progress"] = 100.0
        _model_download_state["message"] = "模型加载完成"
        if progress_callback:
            progress_callback(ASRProgressMessage(
                type="model_loaded",
                message="模型加载完成",
                model_progress=100.0
            ))


def get_model(progress_callback=None):
    """Get or load the ASR model (singleton, thread-safe)."""
    global _MODEL, _MODEL_LOCK, _MODEL_LOADING

    if _MODEL is not None:
        return _MODEL, _MODEL_LOCK

    with _MODEL_LOADING_LOCK:
        if _MODEL is not None:
            return _MODEL, _MODEL_LOCK
        if _MODEL_LOADING:
            # Wait for model to finish loading
            while _MODEL_LOADING and _MODEL is None:
                time.sleep(0.1)
            return _MODEL, _MODEL_LOCK

        _MODEL_LOADING = True
        try:
            _MODEL = _load_model(progress_callback)
            _MODEL_LOCK = threading.Lock()
            return _MODEL, _MODEL_LOCK
        finally:
            _MODEL_LOADING = False


def process_single_audio(
    filepath: str,
    config: ASRConfig,
    speaker_map: Optional[Dict[int, str]] = None
) -> ASRResult:
    """Process a single audio file and return ASR result."""
    filename = os.path.basename(filepath)
    model, lock = get_model()

    with lock:
        _stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            result = model.generate(
                input=filepath,
                language=config.language,
                use_itn=config.use_itn,
                batch_size_s=config.batch_size_s,
            )
        except Exception as e:
            sys.stderr.close()
            sys.stderr = _stderr
            return ASRResult(
                filename=filename,
                status="error",
                error_message=str(e)
            )
        finally:
            if not sys.stderr.closed:
                sys.stderr.close()
            sys.stderr = _stderr

    sentences = []
    full_text_parts = []

    if result and len(result) > 0:
        if result[0].get("sentence_info"):
            for i, sent in enumerate(result[0]["sentence_info"]):
                text = clean_text(sent.get("text", "") or sent.get("sentence", ""))
                if text:
                    spk = None
                    if speaker_map and "spk" in sent:
                        spk = speaker_map.get(sent["spk"], f"说话人{sent['spk']}")
                    sentences.append(ASRSentence(
                        index=i + 1,
                        text=text,
                        speaker=spk,
                        start_ms=sent.get("start"),
                        end_ms=sent.get("end"),
                    ))
                    full_text_parts.append(text)
        elif result[0].get("text"):
            text = clean_text(result[0]["text"])
            if text:
                sentences.append(ASRSentence(index=1, text=text))
                full_text_parts.append(text)

    if not sentences:
        return ASRResult(
            filename=filename,
            status="completed",
            sentences=[],
            full_text="（无识别结果）"
        )

    return ASRResult(
        filename=filename,
        status="completed",
        sentences=sentences,
        full_text="\n".join(full_text_parts),
        processed_at=time.strftime("%Y-%m-%d %H:%M:%S")
    )


def _collect_audio_files(audio_dir: str) -> List[str]:
    """Collect all supported audio files from a directory."""
    files = []
    if not os.path.exists(audio_dir):
        return files

    for root, _, fs in os.walk(audio_dir):
        for f in sorted(fs):
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, f))
    return files


def process_batch(
    audio_dir: str,
    config: ASRConfig,
    progress_callback: Optional[Callable] = None,
    specific_files: Optional[List[str]] = None
) -> List[ASRResult]:
    """Process a batch of audio files with progress tracking."""
    global _processing_state

    # Determine files to process
    if specific_files:
        audio_files = [os.path.join(audio_dir, f) for f in specific_files
                       if os.path.exists(os.path.join(audio_dir, f))]
    else:
        audio_files = _collect_audio_files(audio_dir)

    if not audio_files:
        return []

    # Ensure output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize state - clear previous results
    with _state_lock:
        _processing_state = {
            "is_running": True,
            "current": 0,
            "total": len(audio_files),
            "results": {},
            "config": config.model_dump(),
            "cancel_requested": False,
            "status_message": "正在初始化...",
        }

    # Pre-load model (with progress callback for download tracking)
    with _state_lock:
        _processing_state["status_message"] = "正在加载ASR模型..."
    if progress_callback:
        progress_callback(ASRProgressMessage(
            type="progress",
            current=0,
            total=len(audio_files),
            message="正在加载ASR模型..."
        ))
    get_model(progress_callback)
    if _processing_state["cancel_requested"]:
        with _state_lock:
            _processing_state["is_running"] = False
        return []

    with _state_lock:
        _processing_state["status_message"] = "正在处理音频文件..."
    workers = min(config.max_workers, len(audio_files))
    results = []

    if workers <= 1:
        # Sequential processing
        for i, fp in enumerate(audio_files):
            if _processing_state["cancel_requested"]:
                break

            filename = os.path.basename(fp)
            with _state_lock:
                _processing_state["current"] = i + 1

            result = process_single_audio(fp, config)

            # Only save and track successfully completed results
            if result.status == "completed":
                _save_asr_result(result, config.output_dir)
                with _state_lock:
                    _processing_state["results"][filename] = result.model_dump()
                results.append(result)

                if progress_callback:
                    progress_callback(ASRProgressMessage(
                        type="file_complete",
                        filename=filename,
                        current=i + 1,
                        total=len(audio_files),
                        message=f"完成: {filename} ({len(result.sentences)} 句)",
                        result=result
                    ))
            else:
                if progress_callback:
                    progress_callback(ASRProgressMessage(
                        type="file_complete",
                        filename=filename,
                        current=i + 1,
                        total=len(audio_files),
                        message=f"失败: {filename} - {result.error_message}",
                        result=result
                    ))
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(process_single_audio, fp, config): fp
                              for fp in audio_files}

            completed = 0
            for future in as_completed(future_to_file):
                if _processing_state["cancel_requested"]:
                    for f in future_to_file.values():
                        f.cancel()
                    break

                fp = future_to_file[future]
                filename = os.path.basename(fp)
                completed += 1

                with _state_lock:
                    _processing_state["current"] = completed

                try:
                    result = future.result()
                except Exception as e:
                    result = ASRResult(filename=filename, status="error", error_message=str(e))

                # Only save and track successfully completed results
                if result.status == "completed":
                    _save_asr_result(result, config.output_dir)
                    with _state_lock:
                        _processing_state["results"][filename] = result.model_dump()

                results.append(result)

                if progress_callback:
                    progress_callback(ASRProgressMessage(
                        type="file_complete",
                        filename=filename,
                        current=completed,
                        total=len(audio_files),
                        message=f"完成: {filename} ({len(result.sentences)} 句)" if result.status == "completed" else f"失败: {filename} - {result.error_message}",
                        result=result
                    ))

    with _state_lock:
        _processing_state["is_running"] = False
        _processing_state["status_message"] = "处理完成" if not _processing_state["cancel_requested"] else "已取消"

    if progress_callback:
        progress_callback(ASRProgressMessage(
            type="all_complete",
            current=len(results),
            total=len(audio_files),
            message=f"全部完成！共 {len(results)} 个文件"
        ))

    return results


def _save_asr_result(result: ASRResult, output_dir: str):
    """Save ASR result to a text file in numbered format."""
    os.makedirs(output_dir, exist_ok=True)
    out_name = Path(result.filename).stem + ".txt"
    out_path = os.path.join(output_dir, out_name)

    lines = []
    for sent in result.sentences:
        speaker_prefix = f"[{sent.speaker}] " if sent.speaker else ""
        lines.append(f"part {sent.index}：{speaker_prefix}{sent.text}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_asr_result(filename: str, output_dir: str) -> Optional[ASRResult]:
    """Load ASR result from a saved text file."""
    out_name = Path(filename).stem + ".txt"
    out_path = os.path.join(output_dir, out_name)

    if not os.path.exists(out_path):
        return None

    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    sentences = []
    if content and content != "（无识别结果）":
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Parse "part X：speaker_prefix text"
            match = re.match(r'part (\d+)：(\[([^\]]+)\]\s*)?(.+)', line)
            if match:
                idx = int(match.group(1))
                speaker = match.group(3)
                text = match.group(4)
                sentences.append(ASRSentence(index=idx, text=text, speaker=speaker))

    return ASRResult(
        filename=filename,
        status="completed",
        sentences=sentences,
        full_text="\n".join([s.text for s in sentences]) if sentences else "（无识别结果）"
    )


def get_asr_result(filename: str, output_dir: str) -> Optional[ASRResult]:
    """Get ASR result for a specific file (from memory or disk)."""
    with _state_lock:
        if filename in _processing_state["results"]:
            data = _processing_state["results"][filename]
            return ASRResult(**data)

    return _load_asr_result(filename, output_dir)


def list_asr_results(output_dir: str) -> List[ASRResult]:
    """List all ASR results from the output directory (for history viewing)."""
    results = []
    if not os.path.exists(output_dir):
        return results

    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".txt"):
            result = _load_asr_result(fname, output_dir)
            if result:
                result.filename = fname
                results.append(result)

    return results


def get_current_session_results() -> List[ASRResult]:
    """Get only the current processing session's completed results (from memory)."""
    results = []
    with _state_lock:
        for filename, data in _processing_state["results"].items():
            if data.get("status") == "completed":
                results.append(ASRResult(**data))
    return results


def get_processing_state() -> dict:
    """Get current processing state."""
    with _state_lock:
        return dict(_processing_state)


def get_model_download_state() -> dict:
    """Get current model download state."""
    return dict(_model_download_state)


def cancel_processing():
    """Request cancellation of ongoing ASR processing."""
    with _state_lock:
        _processing_state["cancel_requested"] = True
        _processing_state["status_message"] = "正在取消..."


def get_config() -> ASRConfig:
    """Get the current ASR configuration from state or defaults."""
    with _state_lock:
        if _processing_state["config"]:
            return ASRConfig(**_processing_state["config"])
    return ASRConfig()