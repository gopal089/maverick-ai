"""Speech-to-text engine using faster-whisper."""

import logging
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class STTEngine:
    def __init__(self, model_size: str = "medium", language: Optional[str] = None):
        """
        Initialize the STT engine.

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            language: Language code for transcription (e.g., 'en', 'es'). If None, auto-detect.
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        logger.info(f"Loading Whisper model: {self.model_size}")
        self.model = WhisperModel(
            self.model_size,
            device="cpu",  # For MVP, use CPU; can be changed to cuda if available
            compute_type="int8",  # Use int8 for faster CPU inference
        )
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio (should be 16000 for Whisper)

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("STT model not loaded")

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Whisper expects 16kHz mono audio
        if sample_rate != 16000:
            # TODO: Implement resampling if needed
            logger.warning(f"Sample rate {sample_rate} may not be optimal for Whisper")

        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            best_of=5,
            patience=1.0,
            length_penalty=1.0,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
        )

        # Collect all segments
        text = " ".join([segment.text for segment in segments])
        return text.strip()

# Convenience function for simple usage
def transcribe_audio(audio: np.ndarray, sample_rate: int = 16000,
                     model_size: str = "base", language: Optional[str] = None) -> str:
    """
    Transcribe audio to text using a temporary STT engine.

    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        model_size: Whisper model size
        language: Language code

    Returns:
        Transcribed text
    """
    engine = STTEngine(model_size=model_size, language=language)
    return engine.transcribe(audio, sample_rate)