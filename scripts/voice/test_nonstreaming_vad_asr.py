#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
for import_root in (SRC_ROOT, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from voice.asr import ASRBackend, WhisperASR  # noqa: E402
from voice.audio import load_audio_mono, slice_audio  # noqa: E402
from voice.text_utils import normalize_text  # noqa: E402
from voice.vad_detector import StreamingSileroVAD  # noqa: E402


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
    speech_ratio: float
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
        audio, sample_rate = load_audio_mono(audio_path, config.runtime.sample_rate)
        if config.runtime.max_duration_s is not None:
            audio = audio[: int(config.runtime.max_duration_s * sample_rate)]

        frame_samples = _seconds_to_samples(config.vad.frame_ms / 1000.0, sample_rate)
        if frame_samples != 512 and sample_rate == 16000:
            logger.log(
                "Warning: Silero streaming VAD commonly expects 512 samples at 16 kHz "
                f"(32 ms), current frame has {frame_samples} samples."
            )

        vad = StreamingSileroVAD(
            sample_rate=sample_rate,
            threshold=config.vad.threshold,
            min_silence_duration_ms=config.vad.min_silence_ms,
            speech_pad_ms=config.vad.speech_pad_ms,
        )
        asr = WhisperASR(_WhisperASRRuntimeConfig(config, sample_rate))
        logger.log(
            "Non-streaming VAD+ASR test: "
            f"audio={audio_path}, frame={config.vad.frame_ms:.1f}ms, "
            f"sample_rate={sample_rate}, model={config.asr.whisper_model_id}"
        )

        events = _run_nonstreaming_vad_asr(audio, sample_rate, config, vad, asr, logger)
        if config.output.output_jsonl is not None:
            _write_jsonl(config.output.output_jsonl.expanduser().resolve(), events)
        logger.log(f"Saved log to: {logger.log_path}")
        logger.log(f"Saved events to: {logger.events_path}")
        logger.log(f"Finished with {len(events)} ASR segments.")
    finally:
        logger.close()


class _WhisperASRRuntimeConfig:
    def __init__(self, config: AppConfig, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.language = config.runtime.language
        self.device = config.runtime.device
        self.whisper_model_id = config.asr.whisper_model_id
        self.whisper_chunk_length_s = config.asr.whisper_chunk_length_s
        self.whisper_condition_on_prev_tokens = config.asr.whisper_condition_on_prev_tokens
        self.whisper_max_new_tokens = config.asr.whisper_max_new_tokens
        self.whisper_no_repeat_ngram_size = config.asr.whisper_no_repeat_ngram_size
        self.whisper_repetition_penalty = config.asr.whisper_repetition_penalty


def _run_nonstreaming_vad_asr(
    audio: np.ndarray,
    sample_rate: int,
    config: AppConfig,
    vad: StreamingSileroVAD,
    asr: ASRBackend,
    logger: RunLogger,
) -> List[NonStreamingASREvent]:
    frame_samples = _seconds_to_samples(config.vad.frame_ms / 1000.0, sample_rate)
    total_frames = int(np.ceil(len(audio) / frame_samples))
    events: List[NonStreamingASREvent] = []
    speech_start: Optional[float] = None
    speech_frame_count = 0
    total_segment_frame_count = 0

    for frame_index in range(total_frames):
        frame_start_sample = frame_index * frame_samples
        frame_end_sample = min(len(audio), frame_start_sample + frame_samples)
        frame_start = frame_start_sample / sample_rate
        frame_end = min(frame_end_sample, len(audio)) / sample_rate
        frame = audio[frame_start_sample:frame_end_sample]
        if frame.size < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.size))

        vad_active = vad.accept_frame(frame, frame_start=frame_start, frame_end=frame_end)
        transition = vad.last_transition
        if config.output.log_frames:
            logger.log(_format_frame_log(frame_index, total_frames, frame_start, frame_end, vad_active))

        if vad_active:
            if speech_start is None:
                speech_start = (
                    transition.time
                    if transition is not None and transition.event_type == "start"
                    else frame_start
                )
                speech_frame_count = 0
                total_segment_frame_count = 0
                logger.log(f"[VAD speech_start] time={speech_start:.3f}s")
            speech_frame_count += 1
            total_segment_frame_count += 1
            continue

        if speech_start is None:
            continue

        speech_end = (
            transition.time
            if transition is not None and transition.event_type == "end"
            else frame_start
        )
        event = _transcribe_segment(
            segment_index=len(events) + 1,
            audio=audio,
            sample_rate=sample_rate,
            config=config,
            asr=asr,
            start=speech_start,
            end=speech_end,
            speech_frame_count=speech_frame_count,
            total_frame_count=max(1, total_segment_frame_count),
        )
        _record_event(event, config, logger, events)
        speech_start = None
        speech_frame_count = 0
        total_segment_frame_count = 0

    if speech_start is not None:
        audio_end = len(audio) / sample_rate
        event = _transcribe_segment(
            segment_index=len(events) + 1,
            audio=audio,
            sample_rate=sample_rate,
            config=config,
            asr=asr,
            start=speech_start,
            end=audio_end,
            speech_frame_count=speech_frame_count,
            total_frame_count=max(1, total_segment_frame_count),
        )
        _record_event(event, config, logger, events)

    return events


def _transcribe_segment(
    segment_index: int,
    audio: np.ndarray,
    sample_rate: int,
    config: AppConfig,
    asr: ASRBackend,
    start: float,
    end: float,
    speech_frame_count: int,
    total_frame_count: int,
) -> NonStreamingASREvent:
    chunk = slice_audio(audio, sample_rate, start, end)
    text = normalize_text(asr.transcribe_segment(chunk), config.output.simplify_chinese)
    speech_ratio = speech_frame_count / total_frame_count if total_frame_count else 0.0
    return NonStreamingASREvent(
        segment_index=segment_index,
        start=round(start, 3),
        end=round(end, 3),
        duration=round(max(0.0, end - start), 3),
        frame_count=total_frame_count,
        speech_ratio=round(speech_ratio, 4),
        text=text,
    )


def _record_event(
    event: NonStreamingASREvent,
    config: AppConfig,
    logger: RunLogger,
    events: List[NonStreamingASREvent],
) -> None:
    events.append(event)
    logger.event(event)
    if event.text or config.output.print_empty:
        logger.log(_format_asr_event(event))


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


def _format_frame_log(
    frame_index: int,
    total_frames: int,
    frame_start: float,
    frame_end: float,
    vad_active: bool,
) -> str:
    speech = "speech" if vad_active else "silence"
    return (
        f"[VAD frame={frame_index + 1:06d}/{total_frames:06d}] "
        f"time={frame_start:8.3f}-{frame_end:8.3f}s vad={speech}"
    )


def _format_asr_event(event: NonStreamingASREvent) -> str:
    return (
        f"[ASR SEGMENT {event.segment_index:04d}] "
        f"time={event.start:8.3f}-{event.end:8.3f}s "
        f"duration={event.duration:6.3f}s "
        f"frames={event.frame_count} speech_ratio={event.speech_ratio:.2f} "
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


def _seconds_to_samples(seconds: float, sample_rate: int) -> int:
    samples = round(seconds * sample_rate)
    if samples <= 0:
        raise ValueError("Time values must be positive.")
    return samples


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
