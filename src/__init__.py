"""
Pacote principal do Chatbot Especialista Genérico.
"""

from src.preprocessor import normalizar
from src.loader import load_base
from src.engine import ChatbotEngine

__all__ = ["normalizar", "load_base", "ChatbotEngine"]
