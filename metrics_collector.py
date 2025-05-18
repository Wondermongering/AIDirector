import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


class MetricsCollector:
    """Collects response times, token usage and error counts."""

    def __init__(self) -> None:
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.token_usage: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)

    def record_response(self, model: str, response_time: float, tokens: int) -> None:
        self.response_times[model].append(response_time)
        self.token_usage[model] += tokens

    def record_error(self, model: str) -> None:
        self.error_counts[model] += 1

    def summary(self) -> Dict[str, Any]:
        models = set(self.response_times) | set(self.token_usage) | set(self.error_counts)
        result: Dict[str, Any] = {}
        for model in models:
            times = self.response_times.get(model, [])
            avg_time = sum(times) / len(times) if times else 0.0
            result[model] = {
                "avg_response_time": avg_time,
                "total_tokens": self.token_usage.get(model, 0),
                "error_count": self.error_counts.get(model, 0),
            }
        return result

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)

