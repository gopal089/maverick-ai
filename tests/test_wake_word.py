#!/usr/bin/env python3
"""
Standalone wake word test for Maverick AI.
Tests openWakeWord wake word detection in isolation using pretrained models.
Usage:
  python tests/test_wake_word.py --wakeword "hey jarvis"
  python tests/test_wake_word.py --wakeword "hey jarvis" --audio_device 0
  python tests/test_wake_word.py --wakeword "hey snowball" --sensitivity 0.7
"""

import argparse
import sys
import time
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model

def main():
    parser = argparse.ArgumentParser(description='Standalone wake word test')
    parser.add_argument('--wakeword', type=str, default='hey jarvis',
                        help='Wake word to detect (default: hey jarvis)')
    parser.add_argument('--audio_device', type=int, default=None,
                        help='Audio device index to use (default: system default)')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Wake word detection threshold [0, 1] (default: 0.5)')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Timeout in seconds (0 = no timeout, default: 0)')

    args = parser.parse_args()

    # Download pretrained models (one-time operation)
    print("Downloading pretrained models (one-time setup)...")
    openwakeword.utils.download_models()

    # Initialize the model
    print(f"Initializing openWakeWord model for wake word: {args.wakeword}")
    model = Model(wakeword_models=[args.wakeword])

    # Debounce variables
    last_detection_time = None
    COOLDOWN_SECONDS = 1.5  # Minimum time between detections

    # Audio parameters - openWakeWord expects 16kHz 16-bit PCM
    SAMPLE_RATE = 16000
    # Frame size should be multiple of 80ms at 16kHz = 1280 samples (80ms)
    # But let's use a reasonable size - 512 samples (~32ms) should work
    FRAME_SIZE = 512

    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Frame size: {FRAME_SIZE} samples ({FRAME_SIZE/SAMPLE_RATE*1000:.1f} ms)")
    print(f"Listening for '{args.wakeword}'... (speak clearly into microphone)")
    print(f"Detection threshold: {args.sensitivity}")
    if args.timeout > 0:
        print(f"Timeout: {args.timeout} seconds")
    print("Press Ctrl+C to exit\n")

    start_time = time.time()

    # Initialize audio stream
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        # Convert float32 audio from sounddevice to 16-bit PCM for openWakeWord
        # sounddevice provides float32 in range [-1, 1]
        # openWakeWord expects 16-bit PCM
        audio_float = indata[:, 0]  # Get mono channel
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Process frame with openWakeWord
        prediction = model.predict(audio_int16)

        # Get current score for our wake word (0 if not in prediction)
        current_score = prediction.get(args.wakeword, 0.0)
        current_time = time.time()

        # Debounce logic: only trigger if:
        # 1. Score is above threshold, AND
        # 2. Either we've never detected before OR score dropped below threshold since last detection
        # 3. AND minimum cooldown period has passed (1 second)
        if current_score >= args.sensitivity:
            # Check if we should allow a new detection
            should_detect = False
            if last_detection_time is None:
                # First detection ever
                should_detect = True
            elif current_score < args.sensitivity:
                # Score dropped below threshold - reset for next detection
                should_detect = False  # Actually, we'll handle this in the else clause below
            elif (current_time - last_detection_time) >= COOLDOWN_SECONDS:
                # Enough time has passed since last detection
                should_detect = True

            if should_detect:
                print("\n" + "="*60)
                print(f"🎉 Wake word '{args.wakeword}' detected! "
                      f"(score: {current_score:.3f})")
                print("="*60 + "\n")
                last_detection_time = current_time
                # In a real implementation, this would trigger the VAD-based listening
                # For this test, we just continue listening
        else:
            # Score dropped below threshold - we're ready for next detection
            # (no action needed, last_detection_time will be checked on next frame)
            pass

    try:
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=FRAME_SIZE,
            callback=audio_callback,
            device=args.audio_device
        )

        audio_stream.start()

        # Keep the script running until interrupted or timeout
        while True:
            time.sleep(0.1)

            # Check timeout if specified
            if args.timeout > 0 and (time.time() - start_time) >= args.timeout:
                print(f"\nTimeout reached ({args.timeout} seconds)")
                break

    except KeyboardInterrupt:
        print("\nStopping wake word detection...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up resources
        if 'audio_stream' in locals():
            audio_stream.stop()
            audio_stream.close()

if __name__ == '__main__':
    main()