"""Per-transcript trace: one event per pipeline step, with inputs, outputs, timing, status and attempt."""
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Event:
    stage: str
    inputs: dict
    outputs: dict = field(default_factory=dict)
    seconds: float = 0.0
    status: str = "ok"            # ok | retry | error
    error: str | None = None
    attempt: int = 1


@dataclass
class Trace:
    run_id: str = ""
    encounter_id: str = ""
    events: list[Event] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, attempt: int = 1, **inputs):
        """Record one step. The caller fills `event.outputs` (and may set `event.status`) inside the block.
        An exception marks the event `error`, records the message, and is re-raised."""
        event = Event(stage=name, inputs=inputs, attempt=attempt)
        start = time.perf_counter()
        try:
            yield event
        except Exception as e:
            event.status, event.error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            event.seconds = round(time.perf_counter() - start, 3)
            self.events.append(event)

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "encounter_id": self.encounter_id,
                "events": [asdict(e) for e in self.events]}

    def save(self, path: Path) -> Path:
        """Write the trace as JSON; non-JSON values (e.g. Paths) are stringified."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path
