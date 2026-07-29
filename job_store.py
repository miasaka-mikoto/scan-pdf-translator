from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobEvent:
    sequence: int
    stage: str
    message: str
    progress: float
    page: int | None = None
    total_pages: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class JobControl:
    def __init__(self) -> None:
        self.pause_requested = threading.Event()
        self.cancel_requested = threading.Event()

    def checkpoint(self) -> None:
        if self.cancel_requested.is_set():
            raise RuntimeError("任务已取消")
        while self.pause_requested.is_set():
            time.sleep(0.25)
            if self.cancel_requested.is_set():
                raise RuntimeError("任务已取消")


class JobStore:
    """Disk-backed events and checkpoints. API keys never enter this class."""

    def __init__(self, root: Path, job_id: str) -> None:
        self.root = root / job_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.status_path = self.root / "status.json"
        # Continue monotonically after a restart; otherwise a resumed task's
        # events become invisible to clients that already consumed earlier IDs.
        self._sequence = 0
        if self.events_path.exists():
            try:
                self._sequence = max((json.loads(line).get("sequence", 0) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()), default=0)
            except (OSError, ValueError, json.JSONDecodeError):
                self._sequence = 0
        self._lock = threading.Lock()
        self.control = JobControl()

    @property
    def pages_dir(self) -> Path:
        path = self.root / "pages"
        path.mkdir(exist_ok=True)
        return path

    def page_dir(self, page_number: int) -> Path:
        path = self.pages_dir / f"page-{page_number:04d}"
        path.mkdir(exist_ok=True)
        return path

    def emit(
        self,
        stage: str,
        message: str,
        progress: float,
        page: int | None = None,
        total_pages: int | None = None,
        **detail: Any,
    ) -> JobEvent:
        with self._lock:
            self._sequence += 1
            event = JobEvent(self._sequence, stage, message, max(0.0, min(1.0, progress)), page, total_pages, detail)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            return event

    def save_status(self, **status: Any) -> None:
        # Status updates occur independently (pause, completion, error). Merge
        # them instead of losing start time, preflight data or output paths.
        previous: dict[str, Any] = {}
        if self.status_path.exists():
            try:
                previous = json.loads(self.status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        payload = {**previous, "updated_at": time.time(), **status}
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.status_path)

    def save_page_state(self, page_number: int, **state: Any) -> None:
        path = self.page_dir(page_number) / "state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def read_events(self, after: int = 0) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["sequence"] > after:
                events.append(item)
        return events
