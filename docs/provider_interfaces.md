# Provider Interfaces

The provider interface layer allows you to extend the orchestrator with new AI services.
All providers implement `ProviderInterface` from `provider_interfaces.py`.

```python
from provider_interfaces import ProviderInterface

class MyProvider(ProviderInterface):
    async def generate(self, messages, **kwargs):
        # connect to your API and return a response
        pass
```

`PluginProvider` is a simple example that would send messages to a custom HTTP endpoint.
