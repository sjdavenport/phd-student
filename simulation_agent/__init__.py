"""
Simulation Agent - An LLM-powered agent for creating and running statistical simulations.
"""

from .agent import SimulationAgent
from .llm import LLMBackend, AnthropicBackend, OpenAIBackend

__version__ = "0.1.0"
__all__ = ["SimulationAgent", "LLMBackend", "AnthropicBackend", "OpenAIBackend"]
