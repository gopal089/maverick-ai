"""Test script for Maverick AI conversation pipeline."""

import logging
import sys
from unittest.mock import patch, MagicMock

import numpy as np

# Add the project root to the path so we can import modules
sys.path.insert(0, '.')

from cli.main import main
from core.llm_engine import LLMEngine
from core.persona_handler import load_persona, create_system_prompt
from stt.engine import STTEngine
from tts.engine import TTSEngine

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mock_stt_transcribe(self, audio, sample_rate=16000):
    """Mock STT transcription that returns predefined responses."""
    # In a real test, we might have a queue of responses
    return "Hello, how are you?"

def mock_llm_generate(self, prompt, system=None, context=None, stream=False):
    """Mock LLM generation that returns predefined responses."""
    if "hello" in prompt.lower() or "hi" in prompt.lower():
        return "I'm doing well, thank you! How can I assist you today?"
    elif "bye" in prompt.lower():
        return "Goodbye! Have a great day!"
    else:
        return "That's interesting. Tell me more."

def mock_tts_synthesize(self, text):
    """Mock TTS synthesis that returns dummy audio."""
    # Return 1 second of silence at 22050 Hz
    return np.zeros(22050, dtype=np.int16)

def run_test_conversation():
    """Run a test conversation in text-only mode with mocked components."""
    logger.info("Starting test conversation...")

    # Patch the STT transcribe method
    with patch.object(STTEngine, 'transcribe', mock_stt_transcribe):
        # Patch the LLM generate method
        with patch.object(LLMEngine, 'generate', mock_llm_generate):
            # Patch the TTS synthesize method
            with patch.object(TTSEngine, 'synthesize', mock_tts_synthesize):
                # Also patch sounddevice play to avoid actually playing audio
                with patch('sounddevice.play'), patch('sounddevice.wait'):
                    # Run the CLI in text-only mode
                    # We'll simulate the main function but we need to avoid the infinite loop
                    # Instead, we'll call the main components directly for testing

                    # Load persona
                    persona_config = load_persona('persona/default.yaml')
                    system_prompt = create_system_prompt(persona_config)

                    # Initialize engines
                    stt_engine = STTEngine()
                    llm_engine = LLMEngine()
                    tts_engine = TTSEngine()

                    conversation_history = []

                    # Run 5 test conversations
                    test_inputs = [
                        "Hello",
                        "What's the weather like today?",
                        "Tell me a joke",
                        "Thanks for the chat",
                        "Bye"
                    ]

                    for i, user_input in enumerate(test_inputs, 1):
                        logger.info(f"Test conversation {i}/5")
                        logger.info(f"User: {user_input}")

                        # Process user input (STT is mocked, so we just use the input)
                        # In text-only mode, we skip actual STT

                        # Add to conversation history
                        conversation_history.append({"role": "user", "content": user_input})

                        # Generate response
                        messages = [{"role": "system", "content": system_prompt}] + conversation_history
                        llm_response = llm_engine.chat(messages)
                        logger.info(f"Assistant: {llm_response}")

                        # Add response to history
                        conversation_history.append({"role": "assistant", "content": llm_response})

                        # Synthesize audio (mocked)
                        audio_response = tts_engine.synthesize(llm_response)
                        # In text-only mode, we don't play audio

                        # Check if we should exit
                        if user_input.lower() in ['bye', 'exit', 'quit']:
                            break

                    logger.info("Test conversation completed successfully")
                    return True

if __name__ == '__main__':
    try:
        success = run_test_conversation()
        if success:
            logger.info("All tests passed!")
            sys.exit(0)
        else:
            logger.error("Tests failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        sys.exit(1)