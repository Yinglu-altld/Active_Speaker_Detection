from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SegmentDoa:
    azimuth_deg: float
    conf: float
    spread_deg: float


@dataclass(frozen=True)
class SegmentRecord:
    seg_id: str
    t_start_ms: int
    t_end_ms: int
    doa: SegmentDoa | None
    doa_scores: list[dict[str, Any]]


class SegmentLogger:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: SegmentRecord) -> None:
        payload = asdict(record)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

