#!/usr/bin/env python3
"""
Check default audio device.
"""
import sounddevice as sd

def main():
    print("Default device (input, output):", sd.default.device)
    print("Device info:")
    print(sd.query_devices(sd.default.device[0]))

if __name__ == "__main__":
    main()