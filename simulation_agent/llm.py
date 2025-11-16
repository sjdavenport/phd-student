"""
LLM Backend abstraction for the simulation agent.
"""

from abc import ABC, abstractmethod
from typing import Optional
import json


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Generate a JSON response from the LLM."""
        pass


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from Claude."""
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Generate a JSON response from Claude."""
        json_prompt = prompt + "\n\nRespond with valid JSON only, no other text."
        response_text = self.generate(json_prompt, system_prompt)

        # Try to extract JSON from response
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from GPT."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096
        )
        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Generate a JSON response from GPT."""
        json_prompt = prompt + "\n\nRespond with valid JSON only, no other text."
        response_text = self.generate(json_prompt, system_prompt)

        # Try to extract JSON from response
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())
