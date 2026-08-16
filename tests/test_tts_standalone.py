#!/usr/bin/env python3
"""
Standalone TTS test for Maverick AI.
Synthesizes text to audio using Piper TTS and plays it.
Usage:
  python tests/test_tts_standalone.py --text "Hello world"
  python tests/test_tts_standalone.py --text "Hello" --voice en_US-lessac-medium --save-to output.wav
"""

import argparse
import sys
import os
import wave
import numpy as np
import sounddevice as sd
from mavcli.tts.engine import TTSEngine

def main():
    parser = argparse.ArgumentParser(description='Standalone TTS test')
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--voice', type=str, default='en_US-lessac-medium', help='Piper voice model name or path to .onnx file (default: en_US-lessac-medium)')
    parser.add_argument('--save-to', type=str, help='If provided, save audio to this WAV file path')
    parser.add_argument('--play', action='store_true', help='Play audio through speakers (default: True if neither --save-to nor --play given?)')
    parser.add_argument('--no-play', dest='play', action='store_false', help='Do not play audio')
    parser.set_defaults(play=True)  # default to play unless overridden by --no-play
    args = parser.parse_args()

    # Initialize TTS engine
    try:
        tts_engine = TTSEngine(voice_model=args.voice)
    except Exception as e:
        print(f"Failed to initialize TTS engine: {e}", file=sys.stderr)
        sys.exit(1)

    # Synthesize
    try:
        audio = tts_engine.synthesize(args.text)
    except Exception as e:
        print(f"Text synthesis failed: {e}", file=sys.stderr)
        sys.exit(2)

    # Determine sample rate (Piper voices typically 22050 Hz, but we can try to get from voice config)
    # For simplicity, assume 22050; we could try to read from tts_engine.voice.config.sample_rate if available.
    sample_rate = getattr(tts_engine.voice, 'config', None)
    if sample_rate and hasattr(sample_rate, 'sample_rate'):
        sample_rate = sample_rate.sample_rate
    else:
        sample_rate = 22050  # fallback

    # Save to wav if requested
    if args.save_to:
        try:
            # Ensure audio is int16 (as returned by synthesize)
            if audio.dtype != np.int16:
                # Convert float to int16 if needed (should not happen with Piper)
                audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
            with wave.open(args.save_to, 'wb') as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())
            print(f"Audio saved to {args.save_to}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to write WAV file: {e}", file=sys.stderr)
            sys.exit(3)

    # Play audio if requested
    if args.play:
        try:
            # Use the engine's play method (which uses sounddevice)
            tts_engine.play(audio)
        except Exception as e:
            print(f"Audio playback failed: {e}", file=sys.stderr)
            sys.exit(4)

    # If neither save nor play, we could still output something; but default is to play.
    # If user explicitly disabled both, just exit silently.
    if not args.save_to and not args.play:
        # No output requested; we could print a message.
        pass

if __name__ == '__main__':
    main()