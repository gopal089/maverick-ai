"""Text-to-speech engine using Piper TTS."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from piper import PiperVoice

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, voice_model: str = "en_US-lessac-medium",
                 use_cuda: bool = False):
        """
        Initialize the TTS engine.

        Args:
            voice_model: Path to the Piper voice model or voice name
            use_cuda: Whether to use CUDA for inference (if available)
        """
        self.voice_model = voice_model
        self.use_cuda = use_cuda
        self.voice = None
        self._load_voice()

    def _load_voice(self):
        """Load the Piper voice model."""
        logger.info(f"Loading Piper voice: {self.voice_model}")
        try:
            # If voice_model is a path to a .onnx file, load it directly
            if os.path.isfile(self.voice_model) and self.voice_model.endswith('.onnx'):
                self.voice = PiperVoice.load(self.voice_model, use_cuda=self.use_cuda)
            else:
                # Assume it's a voice name; look for the model in ~/.local/share/piper/voices
                voice_path = Path.home() / ".local" / "share" / "piper" / "voices" / self.voice_model
                onnx_path = voice_path.with_suffix('.onnx')
                if onnx_path.is_file():
                    self.voice = PiperVoice.load(str(onnx_path), use_cuda=self.use_cuda)
                else:
                    # Voice not found; provide helpful error message
                    raise FileNotFoundError(
                        f"Piper voice model not found for '{self.voice_model}'. "
                        f"Expected to find '{onnx_path}'. "
                        f"Please download the voice model first (e.g., using `piper --download_voice {self.voice_model}`) "
                        f"or provide a valid path to a .onnx file."
                    )
            logger.info("Piper voice loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Piper voice: {e}")
            raise

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as numpy array (int16, 22050 Hz for most Piper voices)
        """
        if self.voice is None:
            raise RuntimeError("TTS voice not loaded")

        # Piper synthesizes to an iterable of AudioChunk objects
        audio_chunks = self.voice.synthesize(text)
        # Concatenate the audio bytes from each chunk
        audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
        # Convert to numpy array (assuming 16-bit PCM)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_np

    def play(self, audio: np.ndarray, sample_rate: int = 22050):
        """
        Play audio through the default output device.

        Args:
            audio: Audio data as numpy array (int16)
            sample_rate: Sample rate of the audio
        """
        sd.play(audio, samplerate=sample_rate)
        sd.wait()  # Wait until audio is finished playing

# Convenience function for simple usage
def synthesize_text(text: str, voice_model: str = "en_US-lessac-medium",
                    use_cuda: bool = False) -> np.ndarray:
    """
    Synthesize text to audio using a temporary TTS engine.

    Args:
        text: Text to synthesize
        voice_model: Voice model name or path
        use_cuda: Whether to use CUDA

    Returns:
        Audio data as numpy array
    """
    engine = TTSEngine(voice_model=voice_model, use_cuda=use_cuda)
    return engine.synthesize(text)

def speak_text(text: str, voice_model: str = "en_US-lessac-medium",
               use_cuda: bool = False):
    """
    Synthesize and play text through speakers.

    Args:
        text: Text to speak
        voice_model: Voice model name or path
        use_cuda: Whether to use CUDA
    """
    audio = synthesize_text(text, voice_model, use_cuda)
    engine = TTSEngine(voice_model=voice_model, use_cuda=use_cuda)
    engine.play(audio)