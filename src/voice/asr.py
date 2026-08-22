from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASRBackend(Protocol):
    def transcribe_segment(self, audio: np.ndarray) -> str:
        ...


class ASRRuntimeConfig(Protocol):
    sample_rate: int
    language: str
    device: Optional[str]


class WhisperASRConfig(ASRRuntimeConfig, Protocol):
    whisper_model_id: str
    whisper_chunk_length_s: Optional[float]
    whisper_condition_on_prev_tokens: Optional[bool]
    whisper_max_new_tokens: Optional[int]
    whisper_no_repeat_ngram_size: Optional[int]
    whisper_repetition_penalty: Optional[float]


class ParakeetTDTASRConfig(ASRRuntimeConfig, Protocol):
    parakeet_model_id: str


class ParaformerStreamingASRConfig(ASRRuntimeConfig, Protocol):
    paraformer_streaming_model_id: str
    funasr_streaming_chunk_size: Sequence[int]
    funasr_encoder_chunk_look_back: int
    funasr_decoder_chunk_look_back: int


class ParaformerASRConfig(ASRRuntimeConfig, Protocol):
    paraformer_model_id: str


class SherpaOnnxStreamingZipformerASRConfig(ASRRuntimeConfig, Protocol):
    sherpa_onnx_tokens: str
    sherpa_onnx_encoder: str
    sherpa_onnx_decoder: str
    sherpa_onnx_joiner: str
    sherpa_onnx_num_threads: int
    sherpa_onnx_provider: str
    sherpa_onnx_feature_dim: int
    sherpa_onnx_decoding_method: str
    sherpa_onnx_max_active_paths: int
    sherpa_onnx_hotwords_file: Optional[str]
    sherpa_onnx_hotwords_score: float
    sherpa_onnx_blank_penalty: float
    sherpa_onnx_tail_padding_s: float


class ASRFactoryConfig(
    WhisperASRConfig,
    ParakeetTDTASRConfig,
    ParaformerStreamingASRConfig,
    ParaformerASRConfig,
    SherpaOnnxStreamingZipformerASRConfig,
    Protocol,
):
    asr_backend: str


class BaseASR:
    def __init__(self, config: ASRRuntimeConfig) -> None:
        self.config = config
        self.device = self._resolve_device()

    def _resolve_device(self) -> str:
        if self.config.device:
            return self.config.device
        return "cuda:0" if torch.cuda.is_available() else "cpu"


class WhisperASR(BaseASR):
    def __init__(self, config: WhisperASRConfig) -> None:
        super().__init__(config)
        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(config.whisper_model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config.whisper_model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        pipeline_device = 0 if self.device.startswith("cuda") else -1
        pipe_kwargs: Dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.model,
            "tokenizer": self.processor.tokenizer,
            "feature_extractor": self.processor.feature_extractor,
            "torch_dtype": self.torch_dtype,
            "device": pipeline_device,
        }
        chunk_length_s = getattr(config, "whisper_chunk_length_s", None)
        if chunk_length_s is not None and float(chunk_length_s) > 0:
            pipe_kwargs["chunk_length_s"] = float(chunk_length_s)
        self.pipe = pipeline(**pipe_kwargs)

    def transcribe_segment(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        generate_kwargs = {
            "task": "transcribe",
            "condition_on_prev_tokens": bool(
                getattr(self.config, "whisper_condition_on_prev_tokens", False)
            ),
        }
        if self.config.language:
            generate_kwargs["language"] = self.config.language
        max_new_tokens = getattr(self.config, "whisper_max_new_tokens", None)
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = int(max_new_tokens)
        no_repeat_ngram_size = getattr(self.config, "whisper_no_repeat_ngram_size", None)
        if no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        repetition_penalty = getattr(self.config, "whisper_repetition_penalty", None)
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = float(repetition_penalty)
        result = self.pipe(
            {"array": audio, "sampling_rate": self.config.sample_rate},
            generate_kwargs=generate_kwargs,
        )
        return result["text"].strip()


class ParakeetTDTASR(BaseASR):
    """NVIDIA Parakeet TDT ASR backend.

    Note: nvidia/parakeet-tdt-0.6b-v3 is multilingual but does not list Chinese
    among its supported languages, so it is mainly useful for non-Chinese tests.
    """

    _CHINESE_LANGUAGE_HINTS = {"zh", "zh-cn", "zh-tw", "cmn", "yue", "cn"}

    def __init__(self, config: ParakeetTDTASRConfig, model_id: Optional[str] = None) -> None:
        super().__init__(config)
        self.model_id = model_id or config.parakeet_model_id
        if config.language and config.language.lower() in self._CHINESE_LANGUAGE_HINTS:
            warnings.warn(
                f"{self.model_id} does not advertise Chinese support; "
                "Chinese transcription quality may be poor.",
                RuntimeWarning,
                stacklevel=2,
            )

        pipe_kwargs: Dict[str, Any] = {
            "task": "automatic-speech-recognition",
            "model": self.model_id,
            "device": _pipeline_device(self.device),
        }
        if self.device.startswith("cuda"):
            pipe_kwargs["torch_dtype"] = torch.float16
        self.pipe = pipeline(**pipe_kwargs)

    def transcribe_segment(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        result = self.pipe(
            {
                "array": _as_float32_mono(audio),
                "sampling_rate": self.config.sample_rate,
            }
        )
        return _extract_text(result)


class FunASRParaformerStreamingASR(BaseASR):
    """FunASR Paraformer streaming backend.

    `transcribe_segment` runs a segment through the streaming model with the
    default chunk settings from the model card. `transcribe_chunks` is provided
    for callers that already have chunked audio and want to preserve cache state.
    """

    def __init__(
        self,
        config: ParaformerStreamingASRConfig,
        model_id: Optional[str] = None,
        chunk_size: Optional[Sequence[int]] = None,
        encoder_chunk_look_back: Optional[int] = None,
        decoder_chunk_look_back: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "FunASR Paraformer requires FunASR. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        self.model_id = model_id or config.paraformer_streaming_model_id
        self.chunk_size = list(chunk_size or config.funasr_streaming_chunk_size)
        self.encoder_chunk_look_back = (
            encoder_chunk_look_back
            if encoder_chunk_look_back is not None
            else config.funasr_encoder_chunk_look_back
        )
        self.decoder_chunk_look_back = (
            decoder_chunk_look_back
            if decoder_chunk_look_back is not None
            else config.funasr_decoder_chunk_look_back
        )
        self.model = AutoModel(
            model=self.model_id,
            hub="hf",
            device=_funasr_device(self.device),
        )
        self.reset_asr_stream()

    def transcribe_segment(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        # FunASR's file path input path is the most stable across releases.
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, _as_float32_mono(audio), self.config.sample_rate)
            result = self.model.generate(
                input=temp_file.name,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
        return _extract_text(result)

    def reset_asr_stream(self) -> None:
        self.realtime_asr_cache: Dict[str, Any] = {}
        self.realtime_text_parts: List[str] = []

    def transcribe_with_cache(
        self,
        audio: np.ndarray,
        cache: Dict[str, Any],
        text_parts: List[str],
        is_final: bool,
    ) -> str:
        if audio.size:
            result = self.model.generate(
                input=_as_float32_mono(audio),
                cache=cache,
                is_final=is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            text_delta = _extract_text(result)
            if text_delta:
                text_parts.append(text_delta)
        return "".join(part for part in text_parts if part).strip()

    def transcribe_realtime_chunk(self, audio: np.ndarray, is_final: bool) -> str:
        text = self.transcribe_with_cache(
            audio,
            self.realtime_asr_cache,
            self.realtime_text_parts,
            is_final=is_final,
        )
        if is_final:
            self.reset_asr_stream()
        return text

    def transcribe_full_utterance(self, audio: np.ndarray) -> str:
        return self.transcribe_segment(audio).strip()

    def transcribe_chunks(self, chunks: Iterable[np.ndarray]) -> List[str]:
        cache: Dict[str, Any] = {}
        chunk_list = list(chunks)
        texts: List[str] = []
        for index, chunk in enumerate(chunk_list):
            if chunk.size == 0:
                texts.append("")
                continue
            result = self.model.generate(
                input=_as_float32_mono(chunk),
                cache=cache,
                is_final=index == len(chunk_list) - 1,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            texts.append(_extract_text(result))
        return texts


class FunASRParaformerASR(BaseASR):
    """FunASR offline Paraformer backend for full-segment transcription."""

    def __init__(
        self,
        config: ParaformerASRConfig,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "FunASR Paraformer requires FunASR. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc

        self.model_id = model_id or config.paraformer_model_id
        self.model = AutoModel(
            model=self.model_id,
            hub="hf",
            device=_funasr_device(self.device),
        )

    def transcribe_segment(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        # File input keeps FunASR behavior consistent across releases.
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, _as_float32_mono(audio), self.config.sample_rate)
            result = self.model.generate(input=temp_file.name)
        return _extract_text(result)

    def transcribe_full_utterance(self, audio: np.ndarray) -> str:
        return self.transcribe_segment(audio).strip()


class SherpaOnnxStreamingZipformerASR(BaseASR):
    """Sherpa-ONNX streaming Zipformer transducer backend."""

    def __init__(self, config: SherpaOnnxStreamingZipformerASRConfig) -> None:
        super().__init__(config)
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "Sherpa-ONNX streaming Zipformer ASR requires sherpa_onnx. "
                "Install it with: pip install sherpa-onnx"
            ) from exc

        self.sherpa_onnx = sherpa_onnx
        self.tokens = _required_file(config.sherpa_onnx_tokens, "sherpa_onnx_tokens")
        self.encoder = _required_file(config.sherpa_onnx_encoder, "sherpa_onnx_encoder")
        self.decoder = _required_file(config.sherpa_onnx_decoder, "sherpa_onnx_decoder")
        self.joiner = _required_file(config.sherpa_onnx_joiner, "sherpa_onnx_joiner")
        self.tail_padding_s = float(getattr(config, "sherpa_onnx_tail_padding_s", 0.66))

        hotwords_file = getattr(config, "sherpa_onnx_hotwords_file", None)
        if hotwords_file:
            hotwords_file = str(_required_file(hotwords_file, "sherpa_onnx_hotwords_file"))
        else:
            hotwords_file = ""

        recognizer_kwargs: Dict[str, Any] = {
            "tokens": str(self.tokens),
            "encoder": str(self.encoder),
            "decoder": str(self.decoder),
            "joiner": str(self.joiner),
            "num_threads": int(getattr(config, "sherpa_onnx_num_threads", 1)),
            "sample_rate": self.config.sample_rate,
            "feature_dim": int(getattr(config, "sherpa_onnx_feature_dim", 80)),
            "decoding_method": getattr(config, "sherpa_onnx_decoding_method", "greedy_search"),
            "max_active_paths": int(getattr(config, "sherpa_onnx_max_active_paths", 4)),
            "provider": getattr(config, "sherpa_onnx_provider", "cpu"),
            "hotwords_file": hotwords_file,
            "hotwords_score": float(getattr(config, "sherpa_onnx_hotwords_score", 1.5)),
            "blank_penalty": float(getattr(config, "sherpa_onnx_blank_penalty", 0.0)),
        }
        self.recognizer = self._create_recognizer(recognizer_kwargs)
        self.reset_asr_stream()

    def transcribe_segment(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.config.sample_rate, _as_float32_mono(audio))
        self._finish_stream(stream)
        return self._get_stream_text(stream)

    def reset_asr_stream(self) -> None:
        self.stream = self.recognizer.create_stream()

    def transcribe_realtime_chunk(self, audio: np.ndarray, is_final: bool) -> str:
        if audio.size:
            self.stream.accept_waveform(self.config.sample_rate, _as_float32_mono(audio))
            self._decode_ready_stream(self.stream)
        if is_final:
            self._finish_stream(self.stream)
        text = self._get_stream_text(self.stream)
        if is_final:
            self.reset_asr_stream()
        return text.strip()

    def transcribe_full_utterance(self, audio: np.ndarray) -> str:
        return self.transcribe_segment(audio).strip()

    def transcribe_chunks(self, chunks: Iterable[np.ndarray]) -> List[str]:
        self.reset_asr_stream()
        chunk_list = list(chunks)
        texts: List[str] = []
        for index, chunk in enumerate(chunk_list):
            texts.append(self.transcribe_realtime_chunk(chunk, is_final=index == len(chunk_list) - 1))
        return texts

    def _create_recognizer(self, recognizer_kwargs: Dict[str, Any]) -> Any:
        try:
            return self.sherpa_onnx.OnlineRecognizer.from_transducer(**recognizer_kwargs)
        except TypeError:
            # Older sherpa-onnx wheels do not expose every optional argument.
            fallback_kwargs = dict(recognizer_kwargs)
            for key in ("hotwords_file", "hotwords_score", "blank_penalty"):
                fallback_kwargs.pop(key, None)
            return self.sherpa_onnx.OnlineRecognizer.from_transducer(**fallback_kwargs)

    def _decode_ready_stream(self, stream: Any) -> None:
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

    def _finish_stream(self, stream: Any) -> None:
        if self.tail_padding_s > 0:
            tail_padding = np.zeros(
                int(round(self.tail_padding_s * self.config.sample_rate)),
                dtype=np.float32,
            )
            stream.accept_waveform(self.config.sample_rate, tail_padding)
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        self._decode_ready_stream(stream)

    def _get_stream_text(self, stream: Any) -> str:
        return _sherpa_result_text(self.recognizer.get_result(stream))


def create_asr(config: ASRFactoryConfig) -> ASRBackend:
    backend = config.asr_backend.lower().replace("_", "-")
    if backend in {"whisper", "openai-whisper"}:
        return WhisperASR(config)
    if backend in {"parakeet", "parakeet-tdt", "nvidia-parakeet"}:
        return ParakeetTDTASR(config)
    if backend in {"paraformer", "paraformer-streaming", "funasr-paraformer"}:
        return FunASRParaformerStreamingASR(config)
    if backend in {"paraformer-offline", "paraformer-nonstreaming", "funasr-paraformer-offline"}:
        return FunASRParaformerASR(config)
    if backend in {"sherpa-onnx-zipformer", "sherpa-zipformer", "zipformer-streaming"}:
        return SherpaOnnxStreamingZipformerASR(config)
    raise ValueError(
        f"Unsupported ASR backend: {config.asr_backend!r}. "
        "Expected one of: whisper, parakeet, paraformer-streaming, "
        "paraformer-offline, sherpa-onnx-zipformer."
    )


def _pipeline_device(device: str) -> Any:
    if device.startswith("cuda"):
        return int(device.split(":", 1)[1]) if ":" in device else 0
    if device == "cpu":
        return -1
    return device


def _funasr_device(device: str) -> str:
    if device.startswith("cuda"):
        return device
    if device == "mps":
        warnings.warn("FunASR may not support MPS; falling back to CPU.", RuntimeWarning, stacklevel=2)
        return "cpu"
    return "cpu"


def _as_float32_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32)


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    if isinstance(result, list):
        parts = [_extract_text(item) for item in result]
        return "".join(part for part in parts if part).strip()
    return str(result).strip()


def _required_file(path: Any, field_name: str) -> Path:
    if path is None or not str(path).strip():
        raise ValueError(f"{field_name} is required for Sherpa-ONNX streaming Zipformer ASR.")
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved}")
    return resolved


def _sherpa_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()
