"""
Motor de inferência e Processamento de Linguagem Natural (NLP) do Chatbot Especialista.

Utiliza Machine Learning clássico (TF-IDF + Regressão Logística) do scikit-learn
treinado dinamicamente em tempo de execução ao inicializar a classe.
Suporta intenções fixas (respostas randômicas) e dinâmicas (extração de fatos/entidades).
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import normalizar


class ChatbotEngine:
    """
    Motor central de PLN para classificação de intenções e extração de entidades.
    """

    def __init__(self, dados_base: Dict[str, Any]) -> None:
        """
        Inicializa e treina o pipeline de Machine Learning em tempo de execução.

        Args:
            dados_base: Dicionário contendo a base de conhecimento validada pelo loader.
        """
        self.nome_bot: str = dados_base["nome_bot"]
        self.limiar_confianca: float = float(dados_base["limiar_confianca"])
        self.saudacao_inicial: str = dados_base["saudacao_inicial"]
        self.fallback: str = dados_base["fallback"]
        self.intencoes: List[Dict[str, Any]] = dados_base["intencoes"]

        # Mapa rápido de tags para os metadados da intenção
        self.intencoes_map: Dict[str, Dict[str, Any]] = {
            inc["tag"]: inc for inc in self.intencoes
        }

        # Índice pré-calculado de fatos normalizados para cada intenção dinâmica
        # Estrutura: tag -> list of (chave_normalizada, chave_original, valor)
        # Ordenado por comprimento decrescente da chave normalizada para casamento mais específico
        self.fatos_index: Dict[str, List[Tuple[str, str, str]]] = {}
        self._construir_indice_fatos()

        # Pipeline Scikit-Learn: TF-IDF (1 a 2 ngrams) + Regressão Logística com C=10.0
        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=10.0,
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Treinamento imediato na inicialização
        self._treinar()

    def _construir_indice_fatos(self) -> None:
        """
        Indexa e normaliza as chaves do dicionário de fatos das intenções dinâmicas,
        ordenando-as pelo tamanho do texto para priorizar correspondências mais longas.
        """
        for inc in self.intencoes:
            tag = inc["tag"]
            if "fatos" in inc and isinstance(inc["fatos"], dict):
                itens_fatos = []
                for chave_orig, valor in inc["fatos"].items():
                    chave_norm = normalizar(str(chave_orig))
                    itens_fatos.append((chave_norm, str(chave_orig), str(valor)))

                # Ordena decrescente pelo comprimento da chave normalizada
                itens_fatos.sort(key=lambda item: len(item[0]), reverse=True)
                self.fatos_index[tag] = itens_fatos

    def _treinar(self) -> None:
        """
        Prepara as frases de treino normalizadas e ajusta o modelo scikit-learn.
        """
        X: List[str] = []
        y: List[str] = []

        for inc in self.intencoes:
            tag = inc["tag"]
            for exemplo in inc["exemplos"]:
                exemplo_norm = normalizar(exemplo)
                if exemplo_norm:
                    X.append(exemplo_norm)
                    y.append(tag)

        if not X:
            raise ValueError("Não há exemplos válidos para treinamento da base.")

        self.pipeline.fit(X, y)

    def extrair_entidade(
        self, tag: str, texto_norm: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Busca uma entidade associada aos fatos da intenção dentro do texto normalizado:
        1. Busca casamento exato da chave normalizada (respeitando fronteiras de palavra).
        2. Caso não encontre e a chave possua múltiplos termos (ex: 'Ayrton Senna'),
           busca termos identificadores individuais significativos (ex: 'senna').

        Args:
            tag: Tag da intenção predita.
            texto_norm: Texto da mensagem do usuário já normalizado.

        Returns:
            Tupla (chave_original_encontrada, valor_correspondente) ou (None, None).
        """
        if tag not in self.fatos_index:
            return None, None

        # Passo 1: Casamento direto da chave inteira normalizada
        for chave_norm, chave_orig, valor in self.fatos_index[tag]:
            if not chave_norm:
                continue
            padrao = rf"(?<!\w){re.escape(chave_norm)}(?!\w)"
            if re.search(padrao, texto_norm):
                return chave_orig, valor

        # Passo 2: Casamento por termos identificadores (para nomes próprios compostos)
        for chave_norm, chave_orig, valor in self.fatos_index[tag]:
            termos = [t for t in chave_norm.split() if len(t) >= 4]
            for termo in termos:
                padrao = rf"(?<!\w){re.escape(termo)}(?!\w)"
                if re.search(padrao, texto_norm):
                    return chave_orig, valor

        return None, None

    def processar(self, mensagem: str) -> Dict[str, Any]:
        """
        Processa uma mensagem do usuário:
        1. Normaliza a entrada.
        2. Classifica a intenção via TF-IDF + Logistic Regression.
        3. Avalia o limiar de confiança.
        4. Resolve intenção fixa ou dinâmica (com extração de entidade).

        Args:
            mensagem: Texto digitado pelo usuário.

        Returns:
            Dicionário com:
                - resposta (str)
                - texto_normalizado (str)
                - intencao (str)
                - confianca (float)
                - entidade_encontrada (Optional[str])
        """
        texto_norm = normalizar(mensagem)

        # Se a mensagem for vazia após normalização
        if not texto_norm:
            return {
                "resposta": self.fallback,
                "texto_normalizado": "",
                "intencao": "desconhecida",
                "confianca": 0.0,
                "entidade_encontrada": None,
            }

        # Obtenção das probabilidades de todas as classes
        probabilidades = self.pipeline.predict_proba([texto_norm])[0]
        classes = self.pipeline.classes_
        indice_maior = int(np.argmax(probabilidades))

        intencao_predita = str(classes[indice_maior])
        confianca = float(probabilidades[indice_maior])

        # Se a confiança for menor que o limiar estipulado na base
        if confianca < self.limiar_confianca:
            return {
                "resposta": self.fallback,
                "texto_normalizado": texto_norm,
                "intencao": intencao_predita,
                "confianca": confianca,
                "entidade_encontrada": None,
            }

        intencao_config = self.intencoes_map[intencao_predita]

        # Caso 1: Intenção Fixa (sorteia uma das respostas da lista)
        if "respostas" in intencao_config:
            resposta_escolhida = random.choice(intencao_config["respostas"])
            return {
                "resposta": resposta_escolhida,
                "texto_normalizado": texto_norm,
                "intencao": intencao_predita,
                "confianca": confianca,
                "entidade_encontrada": None,
            }

        # Caso 2: Intenção Dinâmica (busca fato/entidade)
        entidade, valor = self.extrair_entidade(intencao_predita, texto_norm)

        if entidade is not None and valor is not None:
            template = intencao_config.get("template_resposta", "{entidade}: {valor}")
            resposta_formatada = template.format(entidade=entidade, valor=valor)
            return {
                "resposta": resposta_formatada,
                "texto_normalizado": texto_norm,
                "intencao": intencao_predita,
                "confianca": confianca,
                "entidade_encontrada": entidade,
            }

        # Entidade não foi encontrada no texto do usuário
        erro_entidade = intencao_config.get(
            "erro_entidade",
            self.fallback,
        )
        return {
            "resposta": erro_entidade,
            "texto_normalizado": texto_norm,
            "intencao": intencao_predita,
            "confianca": confianca,
            "entidade_encontrada": None,
        }
