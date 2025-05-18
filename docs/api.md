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

## MetricsCollector
Aggregates response times, token usage and error counts and can export a summary to JSON.

These classes are defined in `main.py` and are intended to be composed by `AIOrchestrator` when running a chat session.
