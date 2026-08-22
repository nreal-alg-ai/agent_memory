"""Voice ingestion runtime for audio-file based memory input.

This module owns the voice models and converts an audio file into normalized
transcript segments.  It deliberately does not know about the memory
database; callers can feed the returned segments into ``MemoryRuntime``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .text_utils import normalize_text


class VoiceRuntime:
    """Run VAD, speaker identification, and ASR over an audio file."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        logger: Optional[logging.Logger] = None,
        vad: Any = None,
        asr: Any = None,
        speaker_identifier: Any = None,
    ) -> None:
        self.config = dict(config or {})
        self.runtime_config = (
            self._section("runtime")
            if isinstance(self.config.get("runtime"), dict)
            else dict(self.config)
        )
        self.vad_config = self._section("vad")
        self.asr_config = self._section("asr") or self._section("nonstreaming_asr")
        if "asr_backend" not in self.asr_config:
            self.asr_config["asr_backend"] = (
                "paraformer-offline"
                if self.asr_config.get("paraformer_model_id")
                else "whisper"
            )
        self.config["asr"] = dict(self.asr_config)
        self.speaker_config = self._section("speaker_identification")
        self.logger = logger or logging.getLogger(__name__)

        # Imports are kept here so memory-only MCP clients do not need to load
        # the audio model dependencies before the voice runtime is requested.
        if vad is None:
            from .vad_detector import StreamingSileroVAD

            vad = StreamingSileroVAD(
                sample_rate=self._sample_rate(),
                threshold=float(self.vad_config.get("threshold", 0.5)),
                min_silence_duration_ms=int(
                    self.vad_config.get("min_silence_ms", 100)
                ),
                speech_pad_ms=int(self.vad_config.get("speech_pad_ms", 30)),
                window_frames=max(1, int(self.vad_config.get("window_frames", 5))),
                activate_ratio=float(self.vad_config.get("activate_ratio", 0.6)),
                deactivate_ratio=float(self.vad_config.get("deactivate_ratio", 0.4)),
            )
        self.vad = vad
        self.asr = asr or self._build_asr()
        self.speaker_identifier = (
            speaker_identifier
            if speaker_identifier is not None
            else self._build_speaker_identifier()
        )

    def process_audio_file(
        self,
        audio_path: Path | str,
        *,
        session_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert one audio file into timestamped transcript segments."""
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        from .audio import load_audio_mono, slice_audio

        audio, sample_rate = load_audio_mono(path, self._sample_rate())
        limit = self.runtime_config.get("max_duration_s")
        if limit is not None:
            audio = audio[: max(0, int(float(limit) * sample_rate))]

        recording_start = self._parse_session_start(session_start)
        spans = self._detect_speech_spans(audio, sample_rate)
        chunks = [slice_audio(audio, sample_rate, start, end) for start, end in spans]
        assignments = self._identify_speakers(chunks)
        segments: List[Dict[str, Any]] = []
        for index, ((start, end), chunk, assignment) in enumerate(
            zip(spans, chunks, assignments),
            1,
        ):
            text = normalize_text(
                self.asr.transcribe_segment(chunk),
                bool(
                    self.asr_config.get(
                        "simplify_chinese",
                        self.config.get("output", {}).get("simplify_chinese", True),
                    )
                ),
            )
            if not text:
                self.logger.debug(
                    "ASR returned empty text path=%s segment=%s start=%.3f end=%.3f",
                    path,
                    index,
                    start,
                    end,
                )
                continue
            speaker, speaker_metadata = self._speaker_label(assignment)
            segments.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "started_at": (recording_start + timedelta(seconds=start)).isoformat(),
                    "ended_at": (recording_start + timedelta(seconds=end)).isoformat(),
                    "segment_index": index,
                    "metadata": {
                        "audio_start_s": round(start, 3),
                        "audio_end_s": round(end, 3),
                        "sample_rate": sample_rate,
                        **speaker_metadata,
                    },
                }
            )

        return {
            "status": "ok",
            "audio_path": str(path),
            "sample_rate": sample_rate,
            "audio_duration_s": round(len(audio) / sample_rate, 3),
            "speech_segment_count": len(spans),
            "transcript_segment_count": len(segments),
            "segments": segments,
        }

    def close(self) -> None:
        """Reset streaming VAD state before the host releases this runtime."""
        reset = getattr(self.vad, "reset_stream_state", None)
        if callable(reset):
            reset()

    def _section(self, name: str) -> Dict[str, Any]:
        value = self.config.get(name)
        return dict(value) if isinstance(value, dict) else {}

    def _sample_rate(self) -> int:
        return max(1, int(self.runtime_config.get("sample_rate", 16000)))

    def _build_asr(self) -> Any:
        from .asr import create_asr

        config = dict(self.asr_config)
        config.setdefault("asr_backend", "whisper")
        config.setdefault("sample_rate", self._sample_rate())
        config.setdefault("language", self.runtime_config.get("language", "zh"))
        config.setdefault("device", self.runtime_config.get("device"))
        config.setdefault("whisper_model_id", "openai/whisper-large-v3-turbo")
        config.setdefault("whisper_chunk_length_s", None)
        config.setdefault("whisper_condition_on_prev_tokens", False)
        config.setdefault("whisper_max_new_tokens", 64)
        config.setdefault("whisper_no_repeat_ngram_size", None)
        config.setdefault("whisper_repetition_penalty", None)
        config.setdefault("parakeet_model_id", "nvidia/parakeet-tdt-0.6b-v3")
        config.setdefault("paraformer_streaming_model_id", "funasr/paraformer-zh-streaming")
        config.setdefault("paraformer_model_id", "funasr/paraformer-zh")
        config.setdefault("funasr_streaming_chunk_size", [60, 10, 5])
        config.setdefault("funasr_encoder_chunk_look_back", 4)
        config.setdefault("funasr_decoder_chunk_look_back", 1)
        config.setdefault("sherpa_onnx_tokens", "")
        config.setdefault("sherpa_onnx_encoder", "")
        config.setdefault("sherpa_onnx_decoder", "")
        config.setdefault("sherpa_onnx_joiner", "")
        config.setdefault("sherpa_onnx_num_threads", 1)
        config.setdefault("sherpa_onnx_provider", "cpu")
        config.setdefault("sherpa_onnx_feature_dim", 80)
        config.setdefault("sherpa_onnx_decoding_method", "greedy_search")
        config.setdefault("sherpa_onnx_max_active_paths", 4)
        config.setdefault("sherpa_onnx_hotwords_file", None)
        config.setdefault("sherpa_onnx_hotwords_score", 1.5)
        config.setdefault("sherpa_onnx_blank_penalty", 0.0)
        config.setdefault("sherpa_onnx_tail_padding_s", 0.66)
        return create_asr(SimpleNamespace(**config))

    def _build_speaker_identifier(self) -> Any:
        if not bool(self.speaker_config.get("enabled", False)):
            return None
        from .speaker_detector import SpeakerReferenceDetector

        reference_dir = self.speaker_config.get("user_reference_audio_dir")
        detector = SpeakerReferenceDetector(
            sample_rate=self._sample_rate(),
            device=self.runtime_config.get("device"),
            model_id=str(
                self.speaker_config.get(
                    "model_id",
                    "iic/speech_campplus_sv_zh-cn_16k-common",
                )
            ),
            similarity_threshold=float(
                self.speaker_config.get("similarity_threshold", 0.5)
            ),
            reference_memory_size=max(
                1,
                int(self.speaker_config.get("reference_memory_size", 10)),
            ),
            min_new_reference_segments=max(
                1,
                int(self.speaker_config.get("min_new_reference_segments", 2)),
            ),
            min_new_reference_duration_s=max(
                0.0,
                float(self.speaker_config.get("min_new_reference_duration_s", 3.0)),
            ),
            user_similarity_threshold=self.speaker_config.get(
                "user_similarity_threshold"
            ),
            user_reference_audio_dir=(
                Path(reference_dir).expanduser() if reference_dir else None
            ),
            speaker_assignment_method=str(
                self.speaker_config.get("speaker_assignment_method", "reference")
            ),
        )
        reference_audio = self.speaker_config.get("user_reference_audio")
        if isinstance(reference_audio, list) and reference_audio:
            detector.load_user_reference_audios(
                [Path(item).expanduser().resolve() for item in reference_audio]
            )
        return detector

    def _detect_speech_spans(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float]]:
        self.vad.reset_stream_state()
        frame_samples = max(
            1,
            round(
                float(self.vad_config.get("frame_ms", 32.0))
                / 1000.0
                * sample_rate
            ),
        )
        spans: List[Tuple[float, float]] = []
        speech_start: Optional[float] = None
        total_frames = int(np.ceil(len(audio) / frame_samples)) if len(audio) else 0
        for frame_index in range(total_frames):
            frame_start_sample = frame_index * frame_samples
            frame_end_sample = min(len(audio), frame_start_sample + frame_samples)
            frame_start = frame_start_sample / sample_rate
            frame_end = frame_end_sample / sample_rate
            frame = audio[frame_start_sample:frame_end_sample]
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))
            active = self.vad.accept_frame(
                frame,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            transition = self.vad.last_transition
            if active and speech_start is None:
                speech_start = max(
                    0.0,
                    transition.time
                    if transition is not None and transition.event_type == "start"
                    else frame_start,
                )
            elif not active and speech_start is not None:
                speech_end = (
                    transition.time
                    if transition is not None and transition.event_type == "end"
                    else frame_start
                )
                if speech_end > speech_start:
                    spans.append((speech_start, min(speech_end, len(audio) / sample_rate)))
                speech_start = None
        if speech_start is not None:
            audio_end = len(audio) / sample_rate
            if audio_end > speech_start:
                spans.append((speech_start, audio_end))
        self.logger.info(
            "Voice VAD completed frames=%s speech_segments=%s duration_s=%.3f",
            total_frames,
            len(spans),
            len(audio) / sample_rate if sample_rate else 0.0,
        )
        return spans

    def _identify_speakers(self, chunks: Sequence[np.ndarray]) -> List[Any]:
        if not chunks or self.speaker_identifier is None:
            return [None] * len(chunks)
        try:
            assignments = self.speaker_identifier.identify_batch_segments_speaker(
                list(chunks)
            )
            if len(assignments) == len(chunks):
                return list(assignments)
            self.logger.warning(
                "Speaker identification returned %s assignments for %s chunks; "
                "using unknown labels",
                len(assignments),
                len(chunks),
            )
        except Exception:
            self.logger.exception("Speaker identification failed; using unknown labels")
        return [None] * len(chunks)

    @staticmethod
    def _speaker_label(assignment: Any) -> Tuple[str, Dict[str, Any]]:
        if assignment is None:
            return "unknown_speaker", {}
        speaker_id = int(getattr(assignment, "speaker_id", -1))
        if speaker_id == 0:
            label = "user"
        elif speaker_id > 0:
            label = f"speaker_{speaker_id}"
        else:
            label = "unknown_speaker"
        return label, {
            "speaker_id": speaker_id,
            "speaker_similarity": getattr(assignment, "similarity", None),
            "speaker_assignment_reason": str(
                getattr(assignment, "reason", "unknown")
            ),
        }

    @staticmethod
    def _parse_session_start(value: Optional[str]) -> datetime:
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Invalid session_start timestamp: {value}") from exc
        return datetime.now().astimezone()


__all__ = ["VoiceRuntime"]
