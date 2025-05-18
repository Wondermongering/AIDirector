# AI Orchestrator

AI Orchestrator is a powerful framework for conducting multi-agent conversations between various AI models and human participants. It enables seamless interaction between different language models (like OpenAI's GPT and Anthropic's Claude), allowing them to collaborate, critique, and complement each other's responses in structured conversations.

## Features

- **Multi-Model Orchestration**: Create conversations between different AI models from OpenAI, Anthropic, and other providers
- **Role-Based System**: Assign specialized roles to models (Primary, Critic, Mediator, etc.)
- **Human Participation**: Seamlessly integrate human participants into AI conversations
- **Token Management**: Sophisticated token tracking with multiple strategies to handle context limits
- **Rich Configuration**: YAML-based configuration for models, roles, and conversation parameters
- **Asynchronous Architecture**: Non-blocking operation with async/await pattern
- **Comprehensive Logging**: Both human-readable and structured JSON logs of conversations
- **Beautiful Console Interface**: Rich console output with color coding and formatting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-orchestrator.git
cd ai-orchestrator

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Requirements

- Python 3.9+
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)

## Quick Start

1. Create a configuration file by copying the included example:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml to customize your models and orchestration settings
```

2. Run a simple conversation:

```bash
python orchestrator.py --models gpt-4-turbo claude-sonnet --turns 5
```

3. Include a human in the conversation:

```bash
python orchestrator.py --models gpt-4-turbo claude-sonnet human-participant --turns 3
```

## Configuration

AI Orchestrator uses YAML configuration files to define models, roles, and conversation parameters. Here's a simple example:

```yaml
models:
  gpt-4-turbo:
    provider: OPENAI
    api_name: gpt-4-turbo-preview
    role: PRIMARY
    temperature: 0.8
    system_prompt: "You are a helpful, harmless, and honest AI assistant."

  claude-sonnet:
    provider: ANTHROPIC
    api_name: claude-3-sonnet-20240229
    role: CRITIC
    temperature: 0.7
    system_prompt: "You are a thoughtful critic and analyst."

orchestration:
  default_max_turns: 10
  allow_human_intervention: true
```

See the [configuration documentation](docs/configuration.md) for complete details.

## Role System

The role system allows you to create specialized AI participants:

- **PRIMARY**: General-purpose assistants that lead the conversation
- **CRITIC**: Evaluates responses and provides constructive criticism
- **SUMMARIZER**: Condenses conversation history periodically
- **MEDIATOR**: Helps resolve conflicting viewpoints
- **CREATIVE**: Focuses on generating novel ideas and solutions
- **LOGICAL**: Emphasizes analytical thinking and factual accuracy
- **CUSTOM**: Define your own role with custom system prompts

## Advanced Usage

### Token Management Strategies

```bash
# Use summarization for managing long conversations
python orchestrator.py --models gpt-4-turbo claude-sonnet --token-strategy SUMMARIZE
```

### Custom Roles

```bash
# Define and use models with custom roles
python orchestrator.py --models creative-gpt logical-claude mediator-claude
```

### Saving and Analyzing Conversations

```bash
# Specify a custom log folder
python orchestrator.py --log-folder ./my_conversations
```

## API Reference

The AI Orchestrator exposes several key classes:

- `AIOrchestrator`: Main class for setting up and running conversations
- `ModelRegistry`: Handles model initialization and configuration
- `ResponseGenerator`: Generates responses from models
- `ConversationMemory`: Manages conversation history and token usage

For full API documentation, see the [API reference](docs/api.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the GPT API
- Anthropic for the Claude API
- All contributors and testers

---

Built with ❤️ for the AI research community
