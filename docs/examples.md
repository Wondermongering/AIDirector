# Step-by-Step Examples

## Custom Roles

1. Open `config.yaml` and add your model with `role: CUSTOM` and a `system_prompt`.
2. Run the orchestrator with your custom model name:
   ```bash
   python main.py --models my-custom-model other-model --turns 3
   ```

## Token Strategies

1. Choose a strategy (`TRUNCATE_OLDEST`, `SUMMARIZE`, `SLIDING_WINDOW`) in the model config under `token_limit_strategy`.
2. Start a conversation and watch the memory module apply the strategy automatically.

## Plugin Provider Usage

1. Define a model that uses `CLI` provider or your own provider implementing `ProviderInterface`.
2. Set `WORLD_INTERFACE_KEY` (or your plugin API key) in your environment:
   ```bash
   export WORLD_INTERFACE_KEY=my-secret-key
   ```
3. Run the orchestrator with the plugin model included:
   ```bash
   python main.py --models plugin-model gpt-4
   ```
