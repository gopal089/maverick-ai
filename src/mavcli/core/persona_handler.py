"""Persona handling for Maverick AI."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def load_persona(persona_path: str) -> Dict:
    """
    Load persona configuration from YAML file.

    Args:
        persona_path: Path to the persona YAML file

    Returns:
        Dictionary containing persona configuration
    """
    persona_file = Path(persona_path)
    if not persona_file.exists():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")

    with open(persona_file, 'r') as f:
        persona = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['name', 'language', 'personality', 'tone', 'voice_model']
    for field in required_fields:
        if field not in persona:
            raise ValueError(f"Missing required field '{field}' in persona")

    logger.info(f"Loaded persona: {persona['name']}")
    return persona

def create_system_prompt(persona: Dict) -> str:
    """
    Create a system prompt from persona configuration.

    Args:
        persona: Persona configuration dictionary

    Returns:
        Formatted system prompt string
    """
    system_prompt = f"""You are {persona['name']}, a helpful AI assistant.
Your personality is: {persona['personality']}
Your tone should be: {persona['tone']}
You respond in {persona['language']} language.
Keep your responses concise and conversational.
Do not mention that you are an AI unless asked.
{persona.get('introduction', '')}
{persona.get('limitations', '')}
"""
    return system_prompt.strip()

def get_default_persona() -> Dict:
    """
    Return a default persona configuration.

    Returns:
        Default persona dictionary
    """
    return {
        "name": "Assistant",
        "language": "en",
        "personality": "helpful, friendly, and professional",
        "tone": "clear, concise, and supportive",
        "voice_model": "en_US-lessac-medium"
    }