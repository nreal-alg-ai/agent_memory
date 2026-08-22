#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for import_root in (SRC_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from voice.voice_runtime import VoiceRuntime  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().with_name("test_nonstreaming_vad_asr_config.yaml")
DEFAULT_AUDIO = (
    ROOT
    / "test_data/ambient_transcript/Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav"
)
DEFAULT_RESULT_ROOT = ROOT / "tmp_result/nonstreaming_vad_asr"


@dataclass
class RuntimeConfig:
    audio: Path = DEFAULT_AUDIO
    result_root: Path = DEFAULT_RESULT_ROOT
    sample_rate: int = 16000
    language: str = "zh"
    device: Optional[str] = None
    max_duration_s: Optional[float] = None


@dataclass
class VADConfig:
    frame_ms: float = 32.0
    threshold: float = 0.5
    min_silence_ms: int = 100
    speech_pad_ms: int = 30


@dataclass
class ASRConfig:
    whisper_model_id: str = "openai/whisper-large-v3-turbo"
    whisper_chunk_length_s: Optional[float] = None
    whisper_condition_on_prev_tokens: bool = False
    whisper_max_new_tokens: Optional[int] = 64
    whisper_no_repeat_ngram_size: Optional[int] = None
    whisper_repetition_penalty: Optional[float] = None


@dataclass
class OutputConfig:
    simplify_chinese: bool = True
    print_empty: bool = False
    log_frames: bool = False
    output_jsonl: Optional[Path] = None


@dataclass
class AppConfig:
    config_path: Path
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class NonStreamingASREvent:
    segment_index: int
    start: float
    end: float
    duration: float
    frame_count: int
    speech_ratio: Optional[float]
    text: str


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "run.log"
        self.events_path = run_dir / "events.jsonl"
        self._log_file = self.log_path.open("a", encoding="utf-8")
        self._events_file = self.events_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def event(self, event: NonStreamingASREvent) -> None:
        self._events_file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._events_file.flush()

    def write_config(self, payload: Dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        self._log_file.close()
        self._events_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame-level VAD segmentation followed by non-streaming Whisper ASR."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional audio override for quick tests. Defaults to config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.audio is not None:
        config.runtime.audio = args.audio.expanduser().resolve()

    audio_path = config.runtime.audio.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    run_dir = _create_run_dir(config.runtime.result_root.expanduser().resolve(), audio_path)
    logger = RunLogger(run_dir)
    logger.write_config({"config": config, "audio": audio_path})
    logger.log(f"Run directory: {run_dir}")

    try:
        voice_runtime = VoiceRuntime(_voice_runtime_config(config))
        logger.log(
            "Non-streaming VAD+ASR test: "
            f"audio={audio_path}, frame={config.vad.frame_ms:.1f}ms, "
            f"sample_rate={config.runtime.sample_rate}, "
            f"model={config.asr.whisper_model_id}"
        )

        voice_report = voice_runtime.process_audio_file(audio_path)
        events = _events_from_voice_report(voice_report, config)
        for event in events:
            logger.event(event)
            if event.text or config.output.print_empty:
                logger.log(_format_asr_event(event))
        if config.output.output_jsonl is not None:
            _write_jsonl(config.output.output_jsonl.expanduser().resolve(), events)
        logger.log(f"Saved log to: {logger.log_path}")
        logger.log(f"Saved events to: {logger.events_path}")
        logger.log(f"Finished with {len(events)} ASR segments.")
    finally:
        if "voice_runtime" in locals():
            voice_runtime.close()
        logger.close()


def _voice_runtime_config(config: AppConfig) -> Dict[str, Any]:
    """Translate this script's config dataclasses to VoiceRuntime config."""
    return {
        "runtime": asdict(config.runtime),
        "vad": asdict(config.vad),
        "asr": asdict(config.asr),
        "output": asdict(config.output),
    }


def _events_from_voice_report(
    report: Dict[str, Any],
    config: AppConfig,
) -> List[NonStreamingASREvent]:
    """Preserve the script's event output format from VoiceRuntime segments."""
    events: List[NonStreamingASREvent] = []
    frame_ms = max(1.0, float(config.vad.frame_ms))
    for index, segment in enumerate(report.get("segments") or [], 1):
        metadata = segment.get("metadata") or {}
        start = float(metadata.get("audio_start_s") or 0.0)
        end = float(metadata.get("audio_end_s") or start)
        duration = max(0.0, end - start)
        events.append(
            NonStreamingASREvent(
                segment_index=int(segment.get("segment_index") or index),
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                frame_count=max(1, round(duration * 1000.0 / frame_ms)),
                speech_ratio=None,
                text=str(segment.get("text") or ""),
            )
        )
    return events


def load_config(config_path: Path) -> AppConfig:
    config_path = config_path.expanduser().resolve()
    config = AppConfig(config_path=config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config support requires PyYAML.") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in config file: {config_path}")
    base_dir = config_path.parent
    _load_group(config.runtime, payload, "runtime", base_dir)
    _load_group(config.vad, payload, "vad", base_dir)
    _load_group(config.asr, payload, "asr", base_dir)
    _load_group(config.output, payload, "output", base_dir)
    unknown_groups = sorted(set(payload) - {"runtime", "vad", "asr", "output"})
    if unknown_groups:
        raise ValueError(f"Unknown config group(s): {', '.join(unknown_groups)}")
    return config


def _load_group(target: Any, payload: Dict[str, Any], group_name: str, base_dir: Path) -> None:
    group = payload.get(group_name)
    if group is None:
        return
    if not isinstance(group, dict):
        raise ValueError(f"Expected mapping for config group {group_name!r}.")
    valid_fields = set(getattr(target, "__dataclass_fields__", {}).keys())
    for key, value in group.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown config key: {group_name}.{key}")
        setattr(target, key, _coerce_value(group_name, key, value, base_dir))


def _coerce_value(group_name: str, key: str, value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    if (group_name, key) in {
        ("runtime", "audio"),
        ("runtime", "result_root"),
        ("output", "output_jsonl"),
    }:
        return _resolve_path(value, base_dir)
    return value


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _format_asr_event(event: NonStreamingASREvent) -> str:
    speech_ratio = (
        f"{event.speech_ratio:.2f}"
        if event.speech_ratio is not None
        else "n/a"
    )
    return (
        f"[ASR SEGMENT {event.segment_index:04d}] "
        f"time={event.start:8.3f}-{event.end:8.3f}s "
        f"duration={event.duration:6.3f}s "
        f"frames={event.frame_count} speech_ratio={speech_ratio} "
        f"text={event.text}"
    )


def _write_jsonl(output_path: Path, events: List[NonStreamingASREvent]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    print(f"Saved JSONL events to: {output_path}", flush=True)


def _create_run_dir(result_root: Path, audio_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_name = _safe_name(audio_path.stem)
    run_dir = result_root / f"{timestamp}_{audio_name}"
    suffix = 1
    while run_dir.exists():
        run_dir = result_root / f"{timestamp}_{audio_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _safe_name(value: str) -> str:
    safe_chars = [
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in value
    ]
    return "".join(safe_chars).strip("._") or "audio"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
