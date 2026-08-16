#!/usr/bin/env python3
"""
Standalone STT test for Maverick AI.
Transcribes audio from a file or microphone using faster-whisper.
Usage:
  python tests/test_stt_standalone.py --file path/to/test.wav
  python tests/test_stt_standalone.py --record --duration 5
  python tests/test_stt_standalone.py --record --language es
"""

import argparse
import sys
import wave
import numpy as np
import sounddevice as sd
from mavcli.stt.engine import STTEngine

def load_wav(file_path):
    """Load a wav file and return mono float32 audio at native sample rate."""
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()

        # Read all frames
        frames = wf.readframes(n_frames)
        # Convert to numpy array
        if sample_width == 1:
            dtype = np.uint8  # 8-bit unsigned
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 3:
            dtype = np.int32  # we'll handle 24-bit as 32-bit then shift
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        audio = np.frombuffer(frames, dtype=dtype)

        # Reshape to channels
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        # Convert to float32 in [-1, 1]
        if sample_width == 1:
            audio = audio.astype(np.float32)
            audio = (audio - 128) / 128.0
        elif sample_width == 2:
            audio = audio.astype(np.float32) / 32768.0
        elif sample_width == 3:
            # 24-bit: store in 32-bit int and then divide by 2^23
            audio = audio.astype(np.int32)
            # We'll assume the audio data is in the lower 3 bytes of each 32-bit chunk (little endian).
            # Actually, the way we read, the bytes are in the order they appear in the file.
            # To keep it simple, we'll treat the array as if it were 24-bit aligned in 32-bit ints
            # by shifting left 8 bits and then interpreting as a signed 32-bit integer.
            audio = audio << 8  # now the 24-bit data is in the upper 24 bits, with sign bit at bit 31
            # Sign-extend to 32-bit: if the sign bit (bit 31) is set, we want to keep it as negative.
            # In Python, shifting left and then interpreting as signed is tricky because we have unlimited precision.
            # Instead, we'll convert to float by dividing by 2^23, but we have to account for the sign.
            # We'll do: if the original 24-bit value had its sign bit set, then the 32-bit integer after shifting
            # will be negative? Actually, shifting left by 8 of a 24-bit signed integer does not preserve the sign.
            # Let's do a different approach: we'll convert each 3-byte sample to a signed integer manually.
            # Given time constraints, we'll assume 16-bit or 32-bit wav files for now and note that 24-bit is not supported.
            # For simplicity, we'll raise an error for 24-bit.
            raise NotImplementedError("24-bit wav files are not supported in this test.")
        elif sample_width == 4:
            audio = audio.astype(np.float32) / 2147483648.0  # 2^31

        # If stereo, convert to mono by averaging
        if n_channels > 1:
            audio = np.mean(audio, axis=1)

        return audio, frame_rate

def main():
    parser = argparse.ArgumentParser(description='Standalone STT test')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Path to wav file')
    group.add_argument('--record', action='store_true', help='Record from microphone')
    parser.add_argument('--duration', type=float, default=5.0, help='Recording duration in seconds (default: 5)')
    parser.add_argument('--language', type=str, default='en', help='Language code (default: en)')
    parser.add_argument('--model', type=str, default='medium', help='Whisper model size (default: medium)')

    args = parser.parse_args()

    # Initialize STT engine
    try:
        stt_engine = STTEngine(model_size=args.model, language=args.language)
    except Exception as e:
        print(f"Failed to initialize STT engine: {e}", file=sys.stderr)
        sys.exit(1)

    if args.file:
        try:
            audio, sample_rate = load_wav(args.file)
        except Exception as e:
            print(f"Failed to load wav file: {e}", file=sys.stderr)
            sys.exit(2)

        # Check sample rate
        if sample_rate != 16000:
            print(f"Warning: Sample rate is {sample_rate} Hz, expected 16000 Hz. Whisper may still work but accuracy could be affected.", file=sys.stderr)

        # Transcribe
        try:
            text = stt_engine.transcribe(audio, sample_rate)
            print(text)
        except Exception as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
            sys.exit(3)

    elif args.record:
        print(f"Recording for {args.duration} seconds...", file=sys.stderr)
        try:
            # Record audio
            audio = sd.rec(int(args.duration * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            audio = audio.flatten()  # Ensure 1D array
        except Exception as e:
            print(f"Recording failed: {e}", file=sys.stderr)
            sys.exit(4)

        # Transcribe
        try:
            text = stt_engine.transcribe(audio, 16000)
            print(text)
        except Exception as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
            sys.exit(3)

if __name__ == '__main__':
    main()