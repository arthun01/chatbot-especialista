"""
Módulo de pré-processamento e normalização textual.

Fornece funções para padronizar textos de entrada removendo acentos, pontuação,
espaços excedentes e convertendo caracteres para minúsculas.
"""

import re
import unicodedata


def normalizar(texto: str) -> str:
    """
    Normaliza uma string para facilitar casamento de padrões e vetorização NLP:
    1. Converte para minúsculas.
    2. Remove acentuações e diacríticos (via decomposição NFKD).
    3. Remove pontuações e caracteres especiais, preservando letras e números.
    4. Remove espaços em branco redundantes e formata as bordas.

    Args:
        texto: Texto original em formato string.

    Returns:
        String limpa, sem acentos, sem pontuações e em caixa baixa.
    """
    if not texto:
        return ""

    # Converte para caixa baixa
    texto_limpo = texto.lower()

    # Decomposição unicode para separar caracteres de seus acentos
    texto_nfkd = unicodedata.normalize("NFKD", texto_limpo)
    texto_sem_acento = "".join(
        caractere for caractere in texto_nfkd if not unicodedata.combining(caractere)
    )

    # Substitui qualquer caractere que não seja letra (a-z) ou dígito (0-9) por espaço
    texto_sem_pontuacao = re.sub(r"[^\w\s]", " ", texto_sem_acento)
    texto_sem_pontuacao = texto_sem_pontuacao.replace("_", " ")

    # Reduz sequências de múltiplos espaços a um único espaço e remove espaços nas pontas
    return re.sub(r"\s+", " ", texto_sem_pontuacao).strip()
