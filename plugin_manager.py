"""Plugin loading and hook execution utilities."""
import logging
import importlib
from importlib import metadata
from typing import Any, List

from model_registry import ConfigurationManager

logger = logging.getLogger(__name__)


class PluginManager:
    """Load optional plugins and execute hook functions."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.plugins: List[Any] = []
        self._load_from_config(config_manager)
        self._load_from_entry_points()

    def _load_from_config(self, config_manager: ConfigurationManager) -> None:
        for module_path in getattr(config_manager, "get_plugins", lambda: [])():
            try:
                module = importlib.import_module(module_path)
                self.plugins.append(module)
                logger.info("Loaded plugin %s from config", module_path)
            except Exception as e:  # pragma: no cover - logging only
                logger.warning("Failed to load plugin %s: %s", module_path, e)

    def _load_from_entry_points(self) -> None:
        try:
            try:
                eps = metadata.entry_points(group="aidirector.plugins")
            except TypeError:  # pragma: no cover - older Python
                eps = metadata.entry_points().get("aidirector.plugins", [])
        except Exception:  # pragma: no cover - metadata failure
            eps = []
        for ep in eps:
            try:
                plugin = ep.load()
                self.plugins.append(plugin)
                logger.info("Loaded plugin %s from entry point", ep.name)
            except Exception as e:  # pragma: no cover - logging only
                logger.warning("Failed to load entry point %s: %s", ep.name, e)

    def run_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """Execute a hook across all loaded plugins."""
        results: List[Any] = []
        for plugin in self.plugins:
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                try:
                    results.append(hook(*args, **kwargs))
                except Exception as e:  # pragma: no cover - runtime errors
                    logger.error(
                        "Error in plugin %s during hook %s: %s", plugin, hook_name, e
                    )
        return results
