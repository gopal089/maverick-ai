"""Main entry point for Maverick AI CLI."""

import logging
import sys
from pathlib import Path

import click
import numpy as np
import sounddevice as sd

from mavcli.core.llm_engine import LLMEngine
from mavcli.core.persona_handler import load_persona, create_system_prompt
from mavcli.stt.engine import STTEngine
from mavcli.tts.engine import TTSEngine
from mavcli.core.web_search import web_search

import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def needs_web_search(text: str) -> bool:
    """
    Heuristic to determine if a web search is needed for the given text.
    Returns True if the text appears to be a factual question requiring current information.
    """
    text_lower = text.lower().strip()

    # Question words that often indicate factual questions
    question_words = ['who', 'what', 'when', 'where', 'why', 'how']

    # Keywords that indicate need for current/factual information
    factual_keywords = [
        'stock', 'weather', 'news', 'score', 'won', 'lost', 'price',
        'exchange rate', 'forecast', 'today', 'yesterday', 'this week',
        'last month', '2024', '2025', 'last', 'recent', 'current',
        'score', 'match', 'game', 'tournament', 'championship', 'election',
        'president', 'prime minister', 'celebrity', 'movie', 'release date',
        'price', 'bitcoin', 'crypto', 'dollar', 'rupee', 'euro'
    ]

    # Check for question words
    has_question_word = any(word in text_lower for word in question_words)

    # Check for factual keywords
    has_factual_keyword = any(keyword in text_lower for keyword in factual_keywords)

    # If it has either a question word or factual keyword, we consider it needs search
    # This is biased towards recall (catching more potential factual questions)
    return has_question_word or has_factual_keyword

# Determine the default persona path
try:
    # If the package is installed, we can use importlib.resources
    import importlib.resources as pkg_resources
    DEFAULT_PERSONA_PATH = str(pkg_resources.files('mavcli') / 'persona' / 'default.yaml')
except Exception:
    # Fallback for development
    DEFAULT_PERSONA_PATH = 'src/mavcli/persona/default.yaml'


@click.command()
@click.option('--text-only', is_flag=True, help='Run in text-only mode (no audio)')
@click.option('--persona', default=DEFAULT_PERSONA_PATH, help='Path to persona YAML file')
@click.option('--model', default='qwen2.5:7b', help='Ollama model name')
@click.option('--stt-model', default='medium', help='Whisper model size')
@click.option('--tts-model', default='en_US-lessac-medium', help='Piper TTS voice model')
@click.option('--log-level', default='INFO', help='Logging level')
def main(text_only: bool, persona: str, model: str, stt_model: str, tts_model: str, log_level: str):
    """Maverick AI - A fully local, personality-customizable voice assistant."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting Maverick AI")

    load_dotenv()
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        logger.info("Tavily API key: loaded")
    else:
        logger.info("Tavily API key: NOT FOUND")

    # Load persona
    try:
        persona_config = load_persona(persona)
    except FileNotFoundError:
        logger.error(f"Persona file {persona} not found. Please check the path.")
        sys.exit(1)

    # Debug: show loaded persona
    logger.info(f"Loaded persona config: {persona_config}")

    # Determine LLM model: from persona config, else CLI option
    llm_model = persona_config.get('llm_model', model)

    # Initialize components
    logger.info("Initializing LLM engine...")
    llm_engine = LLMEngine(model_name=llm_model)

    # Initialize STT and TTS only if not in text-only mode
    if not text_only:
        logger.info("Initializing STT engine...")
        stt_engine = STTEngine(model_size=stt_model, language=persona_config.get('language') if persona_config else None)

        logger.info("Initializing TTS engine...")
        tts_engine = TTSEngine(voice_model=tts_model)

    # Create system prompt from persona
    if persona_config:
        system_prompt = create_system_prompt(persona_config)
    else:
        # Fallback to a generic system prompt
        system_prompt = "You are a helpful AI assistant. Keep responses concise and conversational."

    logger.info("Starting conversation loop...")
    conversation_history = []  # List of dicts for Ollama chat format

    try:
        while True:
            # Get user input
            if text_only:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    logger.info("User exited conversation")
                    break
                if not user_input:
                    continue
            else:
                # Listening stage
                print("\rListening...", end='', flush=True)
                # Record audio from microphone
                duration = 5  # seconds
                sample_rate = 16000
                audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()  # Wait for recording to finish
                audio = audio.flatten()  # Convert to 1D array

                # Transcribe audio
                print("\rThinking...", end='', flush=True)
                user_input = stt_engine.transcribe(audio, sample_rate)
                logger.info(f"You said: {user_input}")
                if not user_input:
                    logger.info("No speech detected, continuing...")
                    # Go back to listening for the next turn
                    print("\rListening...", end='', flush=True)
                    continue
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    logger.info("User exited conversation")
                    break

            # Check if we need to perform a web search for current information
            search_results = None
            if needs_web_search(user_input):
                try:
                    search_results = web_search(user_input)
                    logger.info(f"Web search triggered for: {user_input}")
                except Exception as e:
                    logger.error(f"Web search failed: {e}")
                    # Continue without search results

            # Build system prompt with search results if available
            if search_results:
                # Truncate search results to avoid overly long prompts
                max_search_length = 1500
                if len(search_results) > max_search_length:
                    search_results = search_results[:max_search_length] + "..."
                augmented_system_prompt = f"{system_prompt}\n\nYou have access to the following current information from a web search. Use this information to answer the user's question. If the information is not sufficient, say so. Do not make up information.\n\nSearch results:\n{search_results}"
            else:
                augmented_system_prompt = system_prompt

            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Generate LLM response with system prompt (augmented with search results if applicable)
            if text_only:
                print("Thinking...")
            # Prepare messages for Ollama chat: system prompt + conversation history
            messages = [{"role": "system", "content": augmented_system_prompt}] + conversation_history
            llm_response = llm_engine.chat(messages)
            logger.info(f"Assistant: {llm_response}")

            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": llm_response})

            # Output response
            if text_only:
                print(f"\nAssistant: {llm_response}")
            else:
                # Speaking stage
                print("\rSpeaking...", end='', flush=True)
                audio_response = tts_engine.synthesize(llm_response)
                # Play audio (assuming 22050 Hz for Piper)
                sd.play(audio_response, samplerate=22050)
                sd.wait()
                # After speaking, we will go back to listening for the next turn
                # The next iteration will start with Listening... so we don't print it here.

    except KeyboardInterrupt:
        logger.info("Conversation interrupted by user")
    except Exception as e:
        logger.error(f"Error in conversation loop: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()