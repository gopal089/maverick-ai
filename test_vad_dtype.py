#!/usr/bin/env python3
"""
Check the dtype and shape of indata from sounddevice.
"""
import sounddevice as sd
import numpy as np
import time

# Parameters
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

def main():
    print("Checking indata dtype and shape...")
    callback_count = 0

    def audio_callback(indata, frames, time_info, status):
        nonlocal callback_count
        callback_count += 1
        if callback_count <= 5:
            print(f"Callback {callback_count}:")
            print(f"  indata type: {type(indata)}")
            print(f"  indata dtype: {indata.dtype}")
            print(f"  indata shape: {indata.shape}")
            print(f"  frames argument: {frames}")
            print(f"  first 5 samples: {indata[:5, 0] if indata.ndim == 2 else indata[:5]}")
        if callback_count >= 10:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=FRAME_SIZE,
            callback=audio_callback
        ):
            print("Listening for 10 callbacks...")
            while callback_count < 10:
                time.sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()