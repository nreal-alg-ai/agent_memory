from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad


@dataclass
class VADFrameState:
    raw_speech: bool
    frame_start: float
    frame_end: float


@dataclass
class VADTransition:
    event_type: str
    time: float
    frame_start: float
    frame_end: float
    speech_ratio: float


class StreamingSileroVAD:
    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        window_frames: int = 5,
        activate_ratio: float = 0.6,
        deactivate_ratio: float = 0.4,
    ) -> None:
        self.iterator = VADIterator(
            load_silero_vad(),
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        if window_frames <= 0:
            raise ValueError("window_frames must be greater than 0.")
        if not 0.0 <= deactivate_ratio <= activate_ratio <= 1.0:
            raise ValueError("Expected 0 <= deactivate_ratio <= activate_ratio <= 1.")
        self.sample_rate = sample_rate
        self.window_frames = window_frames
        self.activate_ratio = activate_ratio
        self.deactivate_ratio = deactivate_ratio
        self._recent_frames: Deque[VADFrameState] = deque(maxlen=window_frames)
        self._raw_triggered = False
        self._speech_ratio = 0.0
        self._stable_triggered = False
        self._last_transition: Optional[VADTransition] = None
        self._next_frame_start = 0.0

    def accept_frame(
        self,
        frame: np.ndarray,
        frame_start: Optional[float] = None,
        frame_end: Optional[float] = None,
    ) -> bool:
        if frame_start is None:
            frame_start = self._next_frame_start
        if frame_end is None:
            frame_end = frame_start + (len(frame) / self.sample_rate)
        self._next_frame_start = frame_end
        self._last_transition = None

        event = self.iterator(torch.tensor(frame, dtype=torch.float32), return_seconds=True)
        raw_speech = bool(self.iterator.triggered)
        if event and "start" in event:
            raw_speech = True
        elif event and "end" in event:
            raw_speech = False

        self._raw_triggered = raw_speech
        self._recent_frames.append(
            VADFrameState(
                raw_speech=raw_speech,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        )
        self._speech_ratio = (
            sum(state.raw_speech for state in self._recent_frames) / len(self._recent_frames)
        )
        was_triggered = self._stable_triggered
        if not was_triggered and self._speech_ratio >= self.activate_ratio:
            self._stable_triggered = True
            self._last_transition = VADTransition(
                event_type="start",
                time=self._activation_time(frame_start),
                frame_start=frame_start,
                frame_end=frame_end,
                speech_ratio=self._speech_ratio,
            )
        elif was_triggered and self._speech_ratio <= self.deactivate_ratio:
            self._stable_triggered = False
            self._last_transition = VADTransition(
                event_type="end",
                time=self._deactivation_time(frame_start),
                frame_start=frame_start,
                frame_end=frame_end,
                speech_ratio=self._speech_ratio,
            )
        return self._stable_triggered

    def reset_stream_state(self) -> None:
        """Reset stream state while keeping the loaded VAD model resident."""
        self.iterator.reset_states()
        self._recent_frames.clear()
        self._raw_triggered = False
        self._speech_ratio = 0.0
        self._stable_triggered = False
        self._last_transition = None
        self._next_frame_start = 0.0

    def _activation_time(self, fallback_time: float) -> float:
        for state in self._recent_frames:
            if state.raw_speech:
                return state.frame_start
        return fallback_time

    def _deactivation_time(self, fallback_time: float) -> float:
        for state in reversed(self._recent_frames):
            if state.raw_speech:
                return state.frame_end
        return fallback_time

    @property
    def raw_triggered(self) -> bool:
        return self._raw_triggered

    @property
    def speech_ratio(self) -> float:
        return self._speech_ratio

    @property
    def stable_triggered(self) -> bool:
        return self._stable_triggered

    @property
    def last_transition(self) -> Optional[VADTransition]:
        return self._last_transition
