"""Run-local speaker labels for Feishu test replay, not production import."""

import re
from pathlib import Path
from typing import Any, Dict, Sequence

from memory.feishu_transcript import OWNER_SPEAKER


ANONYMOUS_SPEAKER_RE = re.compile(
    r"^(?:说话人|匿名|未知|未识别|unknown(?:[\s_-]|$)|speaker(?:[\s_-]*\d|$))",
    re.IGNORECASE,
)


def remap_meeting_speakers(meetings: Sequence[Dict[str, Any]]) -> None:
    """Number selected speech in replay order, keeping raw labels in reports.

    Call once after file selection, chronological parsing and segment limits.
    Named labels share IDs across meetings, ignoring explicit room/device
    suffixes; anonymous labels are scoped to their source file. This is label
    normalization, not inferred speaker identification or text anonymization.
    """
    assigned: Dict[tuple[str, str], str] = {}
    for meeting in meetings:
        source = str(Path(meeting["source_file"]).resolve())
        mapping: Dict[str, str] = {}
        for segment in meeting["segments"]:
            original = segment["metadata"]["original_speaker"]
            if segment["speaker"] == OWNER_SPEAKER:
                label = OWNER_SPEAKER
            else:
                if ANONYMOUS_SPEAKER_RE.match(original):
                    key = (source, original)
                else:
                    # "张超 在 <room>" and "张超 用 @某人 的设备" still
                    # explicitly name 张超; never identify an anonymous user
                    # from the room name or the owner of a shared device.
                    name = original.split(" 在 ", 1)[0].split(" 用 @", 1)[0].strip()
                    key = ("", name)
                if key not in assigned:
                    assigned[key] = f"speaker_{len(assigned) + 1}"
                label = assigned[key]
            segment["speaker"] = label
            mapping[original] = label
        meeting["speaker_mapping"] = mapping
