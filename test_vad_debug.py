#!/usr/bin/env python3
"""
Standalone test for WebRTC VAD with debug prints.
"""
import webrtcvad
import numpy as np
import sounddevice as sd
import time
import sys

# Parameters
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
MAX_RECORDING_SECONDS = 30
SILENCE_THRESHOLD_SECONDS = 1.5
VAD_MODE = 2

def main():
    print("Initializing WebRTC VAD (mode {})...".format(VAD_MODE))
    vad = webrtcvad.Vad(VAD_MODE)

    print("Starting VAD test. Speak into the microphone...")
    print("Will stop after max recording duration or silence threshold.")

    recorded_audio = []
    triggered = False
    speech_start_time = None
    silence_start_time = None
    stop_requested = False
    overall_start_time = time.time()
    callback_count = 0
    speech_frame_count = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal triggered, speech_start_time, silence_start_time, recorded_audio, stop_requested, callback_count, speech_frame_count
        callback_count += 1
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        if stop_requested:
            return
        audio_float = indata[:, 0].copy()
        audio_int16 = (audio_float * 32767).astype(np.int16)
        if len(audio_int16) != FRAME_SIZE:
            print(f"Warning: frame size mismatch: expected {FRAME_SIZE}, got {len(audio_int16)}", file=sys.stderr)
            return
        try:
            is_speech = vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
        except Exception as e:
            print(f"VAD error: {e}", file=sys.stderr)
            return
        current_time = time.time()
        if is_speech:
            if not triggered:
                triggered = True
                speech_start_time = current_time
                print(f"\n[Speech started at {speech_start_time - overall_start_time:.2f}s]")
                silence_start_time = None
            recorded_audio.append(audio_float.copy())
            speech_frame_count += 1
            silence_start_time = None
        else:
            if triggered:
                if silence_start_time is None:
                    silence_start_time = current_time
                elif current_time - silence_start_time >= SILENCE_THRESHOLD_SECONDS:
                    if not stop_requested:
                        stop_requested = True
                        speech_end_time = current_time
                        print(f"\n[Speech ended at {speech_end_time - overall_start_time:.2f}s]")
                        print(f"Silence duration: {speech_end_time - silence_start_time:.2f}s")
            # If not triggered, we are waiting for speech, do nothing

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=FRAME_SIZE,
            callback=audio_callback
        ):
            print("\nListening... (speak now)")
            while not stop_requested:
                time.sleep(0.1)
                if time.time() - overall_start_time > MAX_RECORDING_SECONDS:
                    print("\n[Max duration reached]")
                    break
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError in stream: {e}", file=sys.stderr)

    overall_end_time = time.time()
    if recorded_audio:
        audio_data = np.concatenate(recorded_audio, axis=0)
        duration = len(audio_data) / SAMPLE_RATE
        print(f"\nRecorded {len(recorded_audio)} frames ({duration:.2f} seconds of audio)")
        print(f"Total callbacks: {callback_count}")
        print(f"Speech frames appended: {speech_frame_count}")
        print(f"Total time from start to stop: {overall_end_time - overall_start_time:.2f} seconds")
    else:
        print("\nNo speech recorded.")

if __name__ == "__main__":
    main()