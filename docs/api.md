# API Overview

This project exposes a small set of classes that coordinate multi-model conversations.

## AIOrchestrator
High level interface that manages the conversation loop. It loads configuration, initializes models, and coordinates response generation and logging.

## ModelRegistry
Loads model definitions from the configuration file and initializes API clients for each provider.

## ResponseGenerator
Handles calling the underlying model APIs (OpenAI, Anthropic, CLI or human input) to generate responses based on the conversation history.

## ConversationMemory
Stores messages exchanged during the conversation and applies token limit strategies such as truncation or summarization when the history grows too large.

## ConfigurationManager
Reads the YAML configuration, providing access to model, orchestration and logging sections.

## ConversationLogger
Writes conversation transcripts to plain text and JSON log files for later analysis.

These classes are defined in `main.py` and are intended to be composed by `AIOrchestrator` when running a chat session.

## Provider Interfaces

`provider_interfaces.py` defines `ProviderInterface` and `PluginProvider` which
allow you to build additional model backends or plugins. Implement
`generate()` to connect to your API and return a response.

## Semantic Memory

`semantic_memory.py` contains a simple `SemanticMemory` class for storing text
embeddings. It can be expanded to integrate a vector database for long term
recall.

## Metrics

Use `MetricsTracker` from `metrics.py` to record numeric metrics during a run.
Enable this behavior with the `metrics_tracking` option in your configuration.
