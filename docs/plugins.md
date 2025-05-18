# Plugin System

AI Orchestrator can be extended with custom plugins. A plugin is a regular Python module that exposes hook functions. The `PluginManager` loads plugins and calls hooks during a run.

## Creating a Plugin

Create a module and define functions matching the hooks you want to implement:

```python
# my_package/my_plugin.py

def on_init(orchestrator):
    print("orchestrator initialised")

def message(actor, message):
    # inspect or modify messages
    pass
```

## Enabling Plugins

Plugins may be loaded in two ways:

1. **Configuration** – list module paths under the `plugins` key in `config.yaml`:

```yaml
plugins:
  - my_package.my_plugin
```

2. **Entry points** – install a package that registers an entry point in the `aidirector.plugins` group. Example using `pyproject.toml`:

```toml
[project.entry-points."aidirector.plugins"]
my-plugin = "my_package.my_plugin"
```

After enabling, hooks can be triggered using `PluginManager.run_hook(<hook_name>, *args, **kwargs)`.
