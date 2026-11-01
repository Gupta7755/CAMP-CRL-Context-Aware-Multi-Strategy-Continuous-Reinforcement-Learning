"""In-memory ring buffer of recent log records, with basic secret redaction."""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from datetime import datetime

_SECRET = re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S+")


def redact(text: str) -> str:
    return _SECRET.sub(lambda m: f"{m.group(1)}=***", text)


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self._records: deque[dict] = deque(maxlen=capacity)
        self._lock_rb = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = redact(record.getMessage())
        except Exception:
            msg = "<unformattable log message>"
        with self._lock_rb:
            self._seq += 1
            self._records.append({
                "seq": self._seq,
                "time": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name, "message": msg,
            })

    def recent(self, limit: int = 200, min_level: str = "DEBUG") -> list[dict]:
        threshold = logging.getLevelName(min_level.upper())
        threshold = threshold if isinstance(threshold, int) else logging.DEBUG
        with self._lock_rb:
            rows = [r for r in self._records if logging.getLevelName(r["level"]) >= threshold]
        return rows[-limit:]
