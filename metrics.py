"""Basic metrics tracking utilities."""
from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class MetricRecord:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)


class MetricsTracker:
    """Collects simple numeric metrics."""

    def __init__(self) -> None:
        self.records: Dict[str, List[MetricRecord]] = {}

    def log(self, name: str, value: float) -> None:
        self.records.setdefault(name, []).append(MetricRecord(name, value))

    def summary(self) -> Dict[str, float]:
        return {
            name: sum(r.value for r in recs) / len(recs)
            for name, recs in self.records.items()
            if recs
        }
