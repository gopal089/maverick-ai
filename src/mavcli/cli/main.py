"""Main entry point for Maverick AI CLI."""

import logging
import sys
import time
from pathlib import Path

import click
import numpy as np
import sounddevice as sd
import webrtcvad

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
@click.option('--wakeword', default='hey jarvis', help='Wake word to listen for (default: hey jarvis)')
@click.option('--audio_device', type=int, default=None, help='Audio device index for wake word detection (default: system default)')
@click.option('--sensitivity', type=float, default=0.5, help='Wake word detection sensitivity [0,1] (default: 0.5)')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--verbose', is_flag=True, help='Enable verbose logging to terminal')
def main(text_only: bool, persona: str, model: str, stt_model: str, tts_model: str, wakeword: str, audio_device: int, sensitivity: float, log_level: str, verbose: bool):
    """Maverick AI - A fully local, personality-customizable voice assistant."""
    # Setup logging
    log_level = getattr(logging, log_level.upper())

    # Create logger (root logger)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # File handler: always on
    log_file = "logs/mavcli.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # If verbose, add a stream handler
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

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

        # Initialize wake word detection
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            logger.error("openwakeword package not installed. Please install it with 'pip install openwakeword'")
            sys.exit(1)
        # Download pretrained models (one-time operation)
        logger.info("Downloading pretrained wake word models (one-time setup)...")
        openwakeword.utils.download_models()
        # Store wake word config for later use
        wake_word_config = {
            'wakeword': wakeword,
            'audio_device': audio_device,
            'sensitivity': sensitivity
        }
        logger.info(f"Wake word detection initialized: {wakeword}")

    # Create system prompt from persona
    if persona_config:
        system_prompt = create_system_prompt(persona_config)
    else:
        # Fallback to a generic system prompt
        system_prompt = "You are a helpful AI assistant. Keep responses concise and conversational."

    # Initialize VAD for voice activity detection
    vad = webrtcvad.Vad(mode=0)  # mode 0-3, 0 being least aggressive (more false positives, fewer false negatives)

    logger.info("Starting conversation loop...")
    conversation_history = []  # List of dicts for Ollama chat format

    # Define helper functions for wake word detection and VAD input
    def wait_for_wake_word():
        nonlocal wake_word_config
        # Import openwakeword inside the function to ensure it's available
        import openwakeword
        from openwakeword.model import Model
        # Create model for the specific wakeword
        model = Model(wakeword_models=[wake_word_config['wakeword']])
        # Audio parameters - openWakeWord expects 16kHz 16-bit PCM
        SAMPLE_RATE = 16000
        FRAME_SIZE = 512  # Must be multiple of 160? openWakeWord expects frame size multiple of 160? We'll use 512 as in test.
        # Debounce variables
        last_detection_time = None
        COOLDOWN_SECONDS = 1.5  # Minimum time between detections
        # Flag to indicate detection
        detected = False

        def audio_callback(indata, frames, time_info, status):
            nonlocal last_detection_time, detected
            if status:
                logger.debug(f"Audio callback status: {status}")
            # Convert float32 audio from sounddevice to 16-bit PCM for openWakeWord
            audio_float = indata[:, 0]  # Get mono channel
            audio_int16 = (audio_float * 32767).astype(np.int16)
            # Process frame with openWakeWord
            prediction = model.predict(audio_int16)
            # Get current score for our wake word (0 if not in prediction)
            current_score = prediction.get(wake_word_config['wakeword'], 0.0)
            current_time = time.time()
            # Debounce logic:
            # - Only trigger detection when score crosses ABOVE threshold
            # - Prevent re-triggering while score remains above threshold
            # - Require score to drop BELOW threshold before allowing next detection
            # - Additionally enforce minimum cooldown time between detections
            if current_score >= wake_word_config['sensitivity'] and last_detection_time is not None:
                # Score is above threshold - check if we should ignore due to debounce/cooldown
                time_since_last_detection = current_time - last_detection_time
                if time_since_last_detection < COOLDOWN_SECONDS:
                    # Still in cooldown period, ignore this detection
                    return
            if current_score >= wake_word_config['sensitivity']:
                # Score is at or above threshold - trigger detection
                logger.info(f"Wake word '{wake_word_config['wakeword']}' detected! (score: {current_score:.3f})")
                detected = True
                last_detection_time = current_time
                # We'll stop the stream by raising an exception or returning a flag? We'll set detected and the outer loop will break.
                # Since we cannot break the stream from within the callback easily, we'll set detected and let the outer loop check.
                # The outer loop will break when detected becomes True.
            # else: score is below threshold, waiting for next wake word utterance

        try:
            audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=FRAME_SIZE,
                callback=audio_callback,
                device=wake_word_config['audio_device']
            )
            audio_stream.start()
            # Wait until detection occurs
            while not detected:
                time.sleep(0.1)
            audio_stream.stop()
            audio_stream.close()
            return True
        except KeyboardInterrupt:
            logger.info("Wake word detection interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return False

    def get_vad_input():
        nonlocal vad, stt_engine, logger
        # Record audio from microphone using VAD
        sample_rate = 16000
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        max_recording_seconds = 30
        silence_threshold_seconds = 1.5

        # Audio buffer for recorded speech
        recorded_frames = []
        triggered = False
        speech_start_time = None
        silence_start_time = None
        stop_requested = False
        started_at = time.time()

        def audio_callback(indata, frames, time_info, status):
            nonlocal triggered, speech_start_time, silence_start_time, recorded_frames, stop_requested
            if status:
                logger.debug(f"Audio callback status: {status}")
            if stop_requested:
                return
            # Convert to mono int16 PCM
            audio_float = indata[:, 0].copy()
            audio_int16 = (audio_float * 32767).astype(np.int16)
            # Ensure we have the right number of frames (should be frame_size)
            if len(audio_int16) != frame_size:
                # If not, we need to handle it (e.g., by buffering)
                # For simplicity, we'll assume the blocksize is set correctly.
                return

            # Voice activity detection
            try:
                is_speech = vad.is_speech(audio_int16.tobytes(), sample_rate)
            except Exception as e:
                logger.error(f"VAD error: {e}")
                return

            current_time = time.time()

            if is_speech:
                if not triggered:
                    # Speech started
                    triggered = True
                    speech_start_time = current_time
                    # We already printed "Listening..." and will keep it until done
                    silence_start_time = None
                # Add frame to recording
                recorded_frames.append(audio_float.copy())
                # Reset silence timer since we have speech
                silence_start_time = None
            else:
                if triggered:
                    # We are in speech, check if silence has persisted long enough
                    if silence_start_time is None:
                        silence_start_time = current_time
                    elif current_time - silence_start_time >= silence_threshold_seconds:
                        # Silence threshold reached, request stop
                        if not stop_requested:  # Guard against multiple triggers
                            stop_requested = True
                            speech_end_time = current_time
                            # Log for debugging
                            logger.debug(f"Speech ended at {speech_end_time - started_at:.2f}s, silence duration: {speech_end_time - silence_start_time:.2f}s")
                        # Do NOT append this frame (the one that triggered the threshold))
                # If not triggered, we are waiting for speech, do nothing

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=frame_size, callback=audio_callback, device=3):
                while not stop_requested:
                    time.sleep(0.1)
                    # Check max duration
                    if time.time() - started_at > max_recording_seconds:
                        print("Max duration reached")
                        break
        except Exception as e:
            logger.error(f"VAD recording error: {e}")
        # Combine recorded frames
        if recorded_frames:
            audio = np.concatenate(recorded_frames, axis=0)
        else:
            audio = np.array([], dtype=np.float32)

        # Transcribe audio
        print("Thinking...")
        user_input = stt_engine.transcribe(audio, sample_rate)
        logger.info(f"You said: {user_input}")
        if not user_input:
            logger.info("No speech detected, continuing...")
            return None
        return user_input

    try:
        if text_only:
            # Keep the original text_only loop (unchanged)
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    logger.info("User exited conversation")
                    break
                if not user_input:
                    continue
                print("Thinking...", end='', flush=True)
                # Check if we need to perform a web search for current information
                search_results = None
                if needs_web_search(user_input):
                    try:
                        # Get search provider from persona config, default to tavily
                        search_provider = persona_config.get('search_provider', 'tavily') if persona_config else 'tavily'
                        search_results = web_search(user_input, search_provider=search_provider)
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
                messages = [{"role": "system", "content": augmented_system_prompt}] + conversation_history
                llm_response = llm_engine.chat(messages)
                logger.info(f"Assistant: {llm_response}")

                # Add assistant response to conversation history
                conversation_history.append({"role": "assistant", "content": llm_response})

                # Output response
                print()  # to end the thinking line
                print(f"{persona_config.get('name', 'Maverick')}: {llm_response}")
        else:
            # State machine for wake word and VAD
            state = 'IDLE'  # Start in idle state waiting for wake word
            while True:
                if state == 'IDLE':
                    print("Say the wake word...")
                    if wait_for_wake_word():
                        state = 'LISTENING'
                elif state == 'LISTENING':
                    user_input = get_vad_input()
                    if user_input is None:
                        # No speech detected, go back to idle
                        state = 'IDLE'
                        continue
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        logger.info("User exited conversation")
                        break
                    # Process the user input (search, LLM, TTS, output)
                    # Check if we need to perform a web search for current information
                    search_results = None
                    if needs_web_search(user_input):
                        try:
                            # Get search provider from persona config, default to tavily
                            search_provider = persona_config.get('search_provider', 'tavily') if persona_config else 'tavily'
                            search_results = web_search(user_input, search_provider=search_provider)
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
                    messages = [{"role": "system", "content": augmented_system_prompt}] + conversation_history
                    llm_response = llm_engine.chat(messages)
                    logger.info(f"Assistant: {llm_response}")

                    # Add assistant response to conversation history
                    conversation_history.append({"role": "assistant", "content": llm_response})

                    # Output response
                    print(f"You: {user_input}")
                    print(f"{persona_config.get('name', 'Maverick')}: {llm_response}")
                    print("Speaking...")
                    # Speaking stage
                    audio_response = tts_engine.synthesize(llm_response)
                    # Play audio (assuming 22050 Hz for Piper)
                    sd.play(audio_response, samplerate=22050)
                    sd.wait()
                    # After speaking, we will go back to listening for the next turn
                    # The next iteration will start with Listening... so we don't print it here.
                    state = 'IDLE'

    except KeyboardInterrupt:
        logger.info("Conversation interrupted by user")
    except Exception as e:
        logger.error(f"Error in conversation loop: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()