# Maverick AI

A fully local, personality-customizable voice assistant - a self-hostable alternative to Siri that anyone can reskin (custom name, custom voice, custom personality, any language). No cloud dependency, no device/filesystem/network access for the assistant itself - it's a sandboxed conversational layer only.

## Features

- Local speech-to-text using faster-whisper
- Local LLM via Ollama (Llama 3.1 8B or Mistral 7B)
- Local text-to-speech using Piper TTS
- Personality customization via YAML config
- Text-only mode for testing
- Sandboxed design: no filesystem, network, or OS access

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- [uv](https://github.com/astral-sh/uv) for Python package management

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   uv sync
   ```

### Model Setup

Pull the required Ollama models (choose one):

```bash
# Llama 3.1 8B
ollama pull llama3.1:8b

# Or Mistral 7B
ollama pull mistral
```

### Web Search Setup (Optional)

To enable real-time web search capabilities, you need to obtain a free API key from Tavily:

1. Go to https://tavily.com and sign up for a free account
2. Create an API key from the dashboard
3. Set the API key as an environment variable:
   ```bash
   export TAVILY_API_KEY="your_api_key_here"
   ```
   Or add it to a `.env` file in the project root (remember `.env` is ignored by git):
   ```
   TAVILY_API_KEY=your_api_key_here
   ```

If the API key is not set, the assistant will honestly inform you that search is not configured and will rely on its internal knowledge only.

### Piper TTS Voices

Piper TTS voices are downloaded automatically on first use. To pre-download a voice:

```bash
pip install piper-tts
# Then download a voice (example: en_US-lessac-medium)
piper --download_voice en_US-lessac-medium
```

## Usage

### Voice Mode (default)

```bash
uv run mavcli
```

### Text-only Mode

```bash
uv run mavcli --text-only
```

### Specify a Persona

```bash
uv run mavcli --persona my_persona.yaml
```

## Project Structure

- `core/` - Orchestration logic
- `persona/` - Configuration files for different personalities
- `stt/` - Speech-to-text module
- `tts/` - Text-to-speech module
- `cli/` - Command-line interface entry point
- `tests/` - Test scripts

## Design Principles

### Sandboxed Operation

The Maverick AI assistant operates under strict sandboxing:
- **No filesystem access**: Cannot read/write files outside of configured voice/model paths
- **Limited network access**: Only the web_search tool can call the Tavily API for real-time information; no other external network calls are made.
- **No OS-level access**: Cannot execute shell commands or access system resources
- **Conversation-only**: Only sees conversation text and produces conversation text/audio

This design ensures privacy and security by default.

## Customization

Create a new persona YAML file in the `persona/` directory:

```yaml
name: "Assistant"
language: "en"
personality: "helpful, friendly, and professional"
tone: "clear, concise, and supportive"
voice_model: "en_US-lessac-medium"
```

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Adding New Features

The architecture is modular:
- STT, LLM, and TTS components are isolated
- Persona configuration drives behavior
- Core orchestrates the pipeline without knowing implementation details

## License

MIT