# Maverick AI - MVP Scaffold Summary

## What's Runnable Right Now (After Setup)

The core scaffolding is complete. After installing dependencies and setting up external components, you can run:

```bash
# Install Python dependencies
uv sync

# Pull required Ollama model (example)
ollama pull llama3.1:8b

# Ensure Piper TTS voice is available (will download on first use)
# Or pre-download: piper --download_voice en_US-lessac-medium

# Run the assistant in text-only mode (for testing)
uv run mavcli --text-only

# Run with voice (requires microphone and speakers)
uv run mavcli
```

The following components are implemented and functional:
- **Project structure**: `/core`, `/persona`, `/stt`, `/tts`, `/cli`, `/tests`
- **Dependency management**: UV with `pyproject.toml` and entry point
- **STT module**: Faster-Whisper integration (`stt/engine.py`)
- **LLM module**: Ollama integration (`core/llm_engine.py`)
- **TTS module**: Piper TTS integration (`tts/engine.py`)
- **Persona system**: YAML-based configuration with system prompt generation (`core/persona_handler.py`)
- **CLI interface**: Text-only and voice modes with conversation loop (`cli/main.py`)
- **Default persona**: `persona/default.yaml`
- **Test scaffold**: `tests/test_conversation.py` (demonstrates mocking approach)

## What Still Needs Manual Setup

### 1. System Dependencies
- **Ollama**: Install from https://ollama.com/ and ensure the server is running
- **Audio system**: 
  - Linux: `libportaudio2` or equivalent for sounddevice
  - macOS: Built-in audio should work
  - Windows: May need additional drivers
- **Piper TTS**: The Python package is installed via UV, but native libraries may need system dependencies

### 2. Model Setup
- **LLM**: Pull at least one model via Ollama:
  ```bash
  ollama pull llama3.1:8b   # or
  ollama pull mistral
  ```
- **TTS Voices**: Piper voices download automatically on first use, but can be pre-downloaded:
  ```bash
  pip install piper-tts  # if not already installed via UV
  piper --download_voice en_US-lessac-medium
  ```

### 3. Permissions & Hardware
- **Microphone**: Ensure your system has a working microphone and permissions are granted
- **Speakers/Headphones**: Audio output device must be available
- **Storage**: Sufficient disk space for models (LLM models can be several GB each)

### 4. Environment
- **Python 3.11+**: Verify correct version is used
- **UV**: Ensure UV is installed and in PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Next Milestone Recommendations

After verifying the MVP works in text-only mode, we recommend:

### Milestone 2: Audio Pipeline Refinement
- Implement proper audio buffering and voice activity detection (VAD)
- Add wake-word detection (e.g., Porcupine) for hands-free operation
- Improve error handling for audio device failures
- Add configurable audio parameters (sample rate, channels, etc.)

### Milestone 3: Enhanced Persona System
- Add conversation memory (short-term context window persistence)
- Implement persona inheritance and overrides
- Add emotional state tracking that influences tone
- Support multi-language switching mid-conversation

### Milestone 4: Safety & Sandboxing Hardening
- Implement actual sandboxing (seccomp, namespaces, or similar) to enforce no filesystem/network access
- Add runtime checks for accidental system calls
- Create secure defaults for all external interfaces
- Document threat model and mitigation strategies

### Milestone 5: Packaging & Distribution
- Create platform-specific installers (Homebrew, Chocolatey, etc.)
- Add Docker container option for easy deployment
- Generate comprehensive documentation with examples
- Implement update mechanism for models and personas

### Milestone 6: Advanced Features
- Allow user-trained voice cloning (opt-in, with privacy controls)
- Add support for multiple LLMs with automatic fallback
- Implement conversation logging and analytics (opt-in)
- Add plugin system for extending functionality

## Current Limitations (Known & Documented)

1. **Sandboxing**: The assistant currently relies on documentation and lack of explicit system calls for sandboxing. True enforcement requires OS-level restrictions.
2. **Memory**: Conversation context is limited to the session window; no persistent memory.
3. **Audio Quality**: Dependent on microphone quality and environment; no noise cancellation.
4. **Model Performance**: CPU-only inference may be slow for larger models; GPU acceleration requires additional setup.
5. **Error Recovery**: Limited recovery from external service failures (Ollama not running, etc.)

## Immediate Next Steps

1. Install system dependencies for audio
2. Install and start Ollama
3. Pull an LLM model
4. Run `uv run mavcli --text-only` to verify the conversation pipeline
5. If successful, try voice mode with `uv run mavcli`

Enjoy building your personalized voice assistant!