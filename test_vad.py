#!/usr/bin/env python3
"""
Standalone test for WebRTC VAD.
Records audio from microphone and uses WebRTC VAD to detect speech.
Reports when speech starts and stops, and total recording duration.
"""

import webrtcvad
import numpy as np
import sounddevice as sd
import time
import sys

# Parameters
SAMPLE_RATE = 16000  # Hz, must be 8000, 16000, 32000, or 48000 for webrtcvad
FRAME_DURATION_MS = 30  # ms, must be 10, 20, or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
MAX_RECORDING_SECONDS = 30
SILENCE_THRESHOLD_SECONDS = 1.5  # seconds of silence to stop recording
VAD_MODE = 0  # 0-3, 0 being least aggressive (more false positives, fewer false negatives)

def main():
    print("Initializing WebRTC VAD (mode {})...".format(VAD_MODE))
    vad = webrtcvad.Vad(VAD_MODE)

    print("Starting VAD test. Speak into the microphone...")
    print("Will stop after max recording duration or silence threshold.")

    # Audio buffer to store recorded speech
    recorded_audio = []  # list of numpy arrays (each frame)
    # State tracking
    triggered = False  # Whether we have detected speech and are recording
    speech_start_time = None
    silence_start_time = None
    stop_requested = False
    overall_start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, speech_start_time, silence_start_time, recorded_audio, stop_requested
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        if stop_requested:
            return
        # Convert to mono float32 numpy array, then to int16 for webrtcvad
        # webrtcvad expects 16-bit PCM audio
        audio_float = indata[:, 0].copy()  # assuming single channel
        # Convert float32 in range [-1, 1] to int16
        audio_int16 = (audio_float * 32767).astype(np.int16)
        # Ensure we have the right number of frames (should be FRAME_SIZE)
        if len(audio_int16) != FRAME_SIZE:
            # If not, we need to handle it (e.g., by buffering)
            # For simplicity, we'll assume the blocksize is set correctly.
            return

        # Voice activity detection
        try:
            is_speech = vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
        except Exception as e:
            print(f"VAD error: {e}", file=sys.stderr)
            return

        current_time = time.time()

        if is_speech:
            if not triggered:
                # Speech started
                triggered = True
                speech_start_time = current_time
                print(f"\n[Speech started at {speech_start_time - overall_start_time:.2f}s]")
                silence_start_time = None  # Reset silence timer
            # Add frame to recording
            recorded_audio.append(audio_float.copy())  # keep float32 for later use
            # Reset silence timer since we have speech
            silence_start_time = None
        else:
            if triggered:
                # We are in speech, check if silence has persisted long enough
                if silence_start_time is None:
                    silence_start_time = current_time
                elif current_time - silence_start_time >= SILENCE_THRESHOLD_SECONDS:
                    # Silence threshold reached, stop recording
                    if not stop_requested:  # Guard against multiple triggers
                        stop_requested = True
                        speech_end_time = current_time
                        print(f"\n[Speech ended at {speech_end_time - overall_start_time:.2f}s]")
                        print(f"Silence duration: {speech_end_time - silence_start_time:.2f}s")
                        # Do NOT append this frame (the one that triggered the threshold))
            # If not triggered, we are waiting for speech, do nothing

    try:
        # Start audio stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=FRAME_SIZE,
            callback=audio_callback,
            device=3  # Use MacBook Air Microphone
        ):
            print("\nListening... (speak now)")
            # Stream will stop when callback requests stop or after max duration
            while not stop_requested:
                time.sleep(0.1)
                # Check max duration
                if time.time() - overall_start_time > MAX_RECORDING_SECONDS:
                    print("\n[Max duration reached]")
                    break
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError in stream: {e}", file=sys.stderr)

    # After stopping, process recorded audio
    overall_end_time = time.time()
    if recorded_audio:
        audio_data = np.concatenate(recorded_audio, axis=0)
        duration = len(audio_data) / SAMPLE_RATE
        print(f"\nRecorded {len(recorded_audio)} frames ({duration:.2f} seconds of audio)")
        print(f"Total time from start to stop: {overall_end_time - overall_start_time:.2f} seconds")
    else:
        print("\nNo speech recorded.")

if __name__ == "__main__":
    main()