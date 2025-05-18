# Configuration Guide

The AI Orchestrator is configured with a YAML file. The configuration is divided into sections described below.

## Models
Define each model under the `models` key. Common fields for a model configuration are:

- `provider` – one of `OPENAI`, `ANTHROPIC`, `CLI`, or `HUMAN`.
- `api_name` – the model identifier used by the provider.
- `role` – conversation role such as `PRIMARY`, `CRITIC`, `SUMMARIZER`, `MEDIATOR`, `CREATIVE`, `LOGICAL`, or `CUSTOM`.
- `temperature` – randomness of the responses.
- `max_response_tokens` – maximum tokens to generate for each response.
- `max_context_tokens` – total tokens allowed in the conversation history for the model.
- `token_limit_strategy` – how to handle long conversations; options are `TRUNCATE_OLDEST`, `SUMMARIZE`, or `SLIDING_WINDOW`.
- `system_prompt` – default system prompt for the model.
- `custom_parameters` – provider specific options passed directly to the API.

## Orchestration
Settings related to the conversation as a whole are placed under `orchestration`:

- `default_max_turns` – number of turns before the orchestrator stops.
- `auto_summarize` – whether to periodically summarize the conversation.
- `summary_interval` – number of turns between summaries.
- `allow_human_intervention` – if a human participant can interject.
- `intervention_probability` – probability of human intervention when enabled.
- `default_models` – list of model names used when no models are specified on the command line.

## Logging
Control log output under `logging`:

- `log_level` – logging verbosity (`INFO`, `DEBUG`, etc.).
- `console_output` – enable colored console output.
- `file_output` – write plain text logs to disk.
- `rich_formatting` – store logs with rich formatting.
- `save_media` – preserve any media attachments.
- `metrics_tracking` – record basic metrics about the run.
- `anonymize_data` – remove personal information from logs.

Use the example configuration file (`config.example.yaml`) as a starting point and modify the fields that apply to your setup.
