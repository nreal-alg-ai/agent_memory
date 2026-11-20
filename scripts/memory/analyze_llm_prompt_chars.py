#!/usr/bin/env python3
"""Summarize prompt character counts by stage without loading PWM or its config.

Example:
    python scripts/memory/analyze_llm_prompt_chars.py path/to/llm_client.log
    python scripts/memory/analyze_llm_prompt_chars.py path/to/llm_client.log* --include-retries

Only timestamped INFO ``LLM request started`` records are counted. DEBUG prompt
bodies and completion records are ignored. Failed requests are still counted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple


REQUEST_START = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} INFO "
    r"(?P<logger>\S+): LLM request started\s+(?P<fields>.*)$"
)


def summarize_prompt_chars(
    paths: Iterable[Path], *, include_retries: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Return stage -> (request count, total characters), streaming log files."""
    totals: Dict[str, Tuple[int, int]] = {}
    seen_calls = set()
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                match = REQUEST_START.match(line.rstrip("\r\n"))
                if match is None:
                    continue
                fields = dict(re.findall(r"(\w+)=(\S+)", match["fields"]))
                stage = fields.get("stage_name")
                chars = fields.get("prompt_chars", "")
                if not stage or not chars.isascii() or not chars.isdigit():
                    continue
                call_id = fields.get("call_id")
                if call_id and not include_retries:
                    key = (match["logger"], stage, call_id)
                    if key in seen_calls:
                        continue
                    seen_calls.add(key)
                count, total = totals.get(stage, (0, 0))
                totals[stage] = (count + 1, total + int(chars))
    return totals


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按 stage 统计 LLM prompt_chars 平均值（字符数，非 token 数）。")
    parser.add_argument("logs", nargs="+", type=Path, help="日志路径；可传入多个文件，包括轮转日志。")
    parser.add_argument("--include-retries", action="store_true", help="计入每次重试；默认同一 call_id 只统计一次。")
    args = parser.parse_args(argv)
    try:
        totals = summarize_prompt_chars(args.logs, include_retries=args.include_retries)
    except OSError as exc:
        parser.error(str(exc))
    if not totals:
        print("未找到有效的 LLM request started 日志。", file=sys.stderr)
        return 0
    print("stage_name\trequest_count\tavg_prompt_chars")
    for stage, (count, total) in sorted(totals.items()):
        print(f"{stage}\t{count}\t{total / count:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
