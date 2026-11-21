"""
Módulo responsável pelo carregamento e validação da Base de Conhecimento JSON.

Garante que o arquivo exista, possua estrutura JSON válida e siga o contrato
obrigatório de metadados, intenções fixas e intenções dinâmicas.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


CAMPOS_OBRIGATORIOS_BASE = [
    "nome_bot",
    "limiar_confianca",
    "saudacao_inicial",
    "fallback",
    "intencoes",
]


def load_base(caminho_arquivo: str | Path) -> Dict[str, Any]:
    """
    Lê e valida o arquivo JSON que define a base de conhecimento do chatbot.

    Args:
        caminho_arquivo: Caminho relativo ou absoluto para o arquivo JSON.

    Returns:
        Dicionário com todos os dados validados da base de conhecimento.

    Raises:
        FileNotFoundError: Se o arquivo especificado não existir.
        ValueError: Se o arquivo não contiver JSON válido ou violar o contrato esperado.
    """
    caminho = Path(caminho_arquivo)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo de base de conhecimento não encontrado: '{caminho.resolve()}'"
        )

    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Erro de sintaxe JSON no arquivo '{caminho}': {err}"
        ) from err
    except Exception as err:
        raise ValueError(f"Falha ao ler o arquivo '{caminho}': {err}") from err

    _validar_estrutura_base(dados, caminho)
    return dados


def _validar_estrutura_base(dados: Any, caminho: Path) -> None:
    """
    Valida internamente as chaves e os tipos de dados do dicionário carregado.
    """
    if not isinstance(dados, dict):
        raise ValueError(
            f"O arquivo '{caminho}' deve conter um objeto JSON (dicionário) na raiz."
        )

    # Validação dos campos obrigatórios de nível superior
    for campo in CAMPOS_OBRIGATORIOS_BASE:
        if campo not in dados:
            raise ValueError(
                f"Campo obrigatório ausente na base '{caminho}': '{campo}'"
            )

    # Validação do limiar de confiança
    limiar = dados["limiar_confianca"]
    if not isinstance(limiar, (int, float)) or not (0.0 <= limiar <= 1.0):
        raise ValueError(
            f"O campo 'limiar_confianca' deve ser um número decimal entre 0.0 e 1.0. Valor recebido: {limiar}"
        )

    # Validação do nome do bot, saudação e fallback
    for campo_texto in ["nome_bot", "saudacao_inicial", "fallback"]:
        if not isinstance(dados[campo_texto], str) or not dados[campo_texto].strip():
            raise ValueError(
                f"O campo '{campo_texto}' deve ser uma string não vazia."
            )

    # Validação da lista de intenções
    intencoes: List[Any] = dados["intencoes"]
    if not isinstance(intencoes, list) or len(intencoes) == 0:
        raise ValueError(
            f"A lista 'intencoes' deve conter ao menos uma intenção definida em '{caminho}'."
        )

    tags_vistas = set()
    for idx, intencao in enumerate(intencoes):
        if not isinstance(intencao, dict):
            raise ValueError(
                f"A intenção no índice {idx} deve ser um objeto JSON (dicionário)."
            )

        tag = intencao.get("tag")
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(
                f"Intenção no índice {idx} possui 'tag' inválida ou ausente."
            )

        if tag in tags_vistas:
            raise ValueError(
                f"Tag duplicada encontrada na base: '{tag}'."
            )
        tags_vistas.add(tag)

        exemplos = intencao.get("exemplos")
        if (
            not isinstance(exemplos, list)
            or len(exemplos) == 0
            or not all(isinstance(ex, str) and ex.strip() for ex in exemplos)
        ):
            raise ValueError(
                f"A intenção '{tag}' deve conter uma lista não vazia de exemplos (strings)."
            )

        # Distinção entre intenção fixa e dinâmica
        possui_respostas = "respostas" in intencao
        possui_fatos = "fatos" in intencao

        if possui_respostas and possui_fatos:
            raise ValueError(
                f"A intenção '{tag}' não pode conter simultaneamente 'respostas' e 'fatos'. "
                f"Defina como intenção fixa OU dinâmica."
            )

        if not possui_respostas and not possui_fatos:
            raise ValueError(
                f"A intenção '{tag}' deve ser fixa (conter 'respostas') "
                f"ou dinâmica (conter 'fatos', 'template_resposta' e 'erro_entidade')."
            )

        if possui_respostas:
            respostas = intencao["respostas"]
            if (
                not isinstance(respostas, list)
                or len(respostas) == 0
                or not all(isinstance(r, str) and r.strip() for r in respostas)
            ):
                raise ValueError(
                    f"A intenção fixa '{tag}' deve conter uma lista não vazia de 'respostas'."
                )

        if possui_fatos:
            fatos = intencao["fatos"]
            if not isinstance(fatos, dict) or len(fatos) == 0:
                raise ValueError(
                    f"A intenção dinâmica '{tag}' deve conter um dicionário não vazio em 'fatos'."
                )

            template = intencao.get("template_resposta")
            if not isinstance(template, str) or not template.strip():
                raise ValueError(
                    f"A intenção dinâmica '{tag}' deve conter uma string 'template_resposta' não vazia."
                )

            erro_entidade = intencao.get("erro_entidade")
            if not isinstance(erro_entidade, str) or not erro_entidade.strip():
                raise ValueError(
                    f"A intenção dinâmica '{tag}' deve conter uma string 'erro_entidade' não vazia."
                )
