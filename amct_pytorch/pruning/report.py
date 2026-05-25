from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class StageEvent:
    stage: str
    method: str
    module: str
    detail: str


@dataclass(slots=True)
class PruneReport:
    backend: str
    params_before: int
    params_after: int = 0
    events: List[StageEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add(self, stage: str, method: str, module: str, detail: str) -> None:
        self.events.append(StageEvent(stage=stage, method=method, module=module, detail=detail))

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "events": [
                {"stage": e.stage, "method": e.method, "module": e.module, "detail": e.detail}
                for e in self.events
            ],
            "warnings": list(self.warnings),
        }
