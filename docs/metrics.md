# Metrics Module

The `metrics.py` module offers a very small metrics tracker.
It stores numeric values using `MetricsTracker.log()` and provides averages via `summary()`.

```
from metrics import MetricsTracker
tracker = MetricsTracker()
tracker.log("tokens", 42)
print(tracker.summary())
```

You can enable or disable metrics tracking with the `metrics_tracking` option in `config.yaml`.
