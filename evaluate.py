"""
Script de avaliação de métricas de Inteligência Artificial para o Chatbot Especialista.

Executa validação cruzada estratificada sobre os exemplos de treino da base de conhecimento
e calcula:
- Acurácia geral
- Relatório de classificação detalhado (Precisão, Recall, F1-Score e Suporte)
- Matriz de confusão formatada em formato tabular
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.loader import load_base
from src.preprocessor import normalizar


def parse_arguments() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Avaliação de métricas (Acurácia, F1-Score, Matriz de Confusão) do Chatbot Especialista."
    )
    parser.add_argument(
        "--base",
        type=str,
        default="bases/f1.json",
        help="Caminho do arquivo JSON da base de conhecimento (padrão: bases/f1.json)",
    )
    return parser.parse_args()


def imprimir_matriz_confusao(cm: List[List[int]], labels: List[str]) -> None:
    """
    Imprime uma matriz de confusão perfeitamente formatada e legível no terminal.
    """
    largura_prefixo = max(len(f"[{i:02d}] {lbl}") for i, lbl in enumerate(labels)) + 2
    largura_coluna = 8

    # Cabeçalho das colunas numeradas
    colunas_header = "".join(f"{i:>{largura_coluna}}" for i in range(len(labels)))
    print(f"\n{'Matriz de Confusão:':<{largura_prefixo}} {colunas_header}")
    print("-" * (largura_prefixo + 1 + largura_coluna * len(labels)))

    for i, (label, linha) in enumerate(zip(labels, cm)):
        prefixo = f"[{i:02d}] {label}"
        linha_formatada = "".join(f"{val:>{largura_coluna}}" for val in linha)
        print(f"{prefixo:<{largura_prefixo}} {linha_formatada}")

    print("\nLegenda das classes preditas (colunas 00 a {:02d}):".format(len(labels) - 1))
    for i, label in enumerate(labels):
        print(f"  [{i:02d}] = {label}")


def main() -> None:
    """Fluxo de avaliação do modelo."""
    args = parse_arguments()
    caminho_base = Path(args.base)

    try:
        dados_base = load_base(caminho_base)
    except Exception as erro:
        print(f"[ERRO] Falha ao carregar a base de dados: {erro}", file=sys.stderr)
        sys.exit(1)

    intencoes = dados_base["intencoes"]
    X: List[str] = []
    y: List[str] = []

    for inc in intencoes:
        tag = inc["tag"]
        for ex in inc["exemplos"]:
            ex_norm = normalizar(ex)
            if ex_norm:
                X.append(ex_norm)
                y.append(tag)

    if not X:
        print("[ERRO] A base não contém exemplos válidos para avaliação.", file=sys.stderr)
        sys.exit(1)

    contagem_classes = Counter(y)
    menor_classe = min(contagem_classes.values())

    if menor_classe < 2:
        print(
            f"[ERRO] Todas as classes devem ter pelo menos 2 exemplos para validação cruzada. "
            f"Menor contagem encontrada: {menor_classe}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Define o número de folds da validação cruzada estratificada (máximo 5)
    n_splits = min(5, menor_classe)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=10.0, max_iter=500, random_state=42)),
        ]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    except Exception as erro:
        print(f"[ERRO] Falha ao executar a validação cruzada: {erro}", file=sys.stderr)
        sys.exit(1)

    acuracia = accuracy_score(y, y_pred)
    classes_ordenadas = sorted(list(set(y)))
    report = classification_report(
        y,
        y_pred,
        labels=classes_ordenadas,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=classes_ordenadas)

    print("=" * 72)
    print("      RELATÓRIO DE AVALIAÇÃO DE INTELIGÊNCIA ARTIFICIAL (NLP)")
    print(f"      Base avaliada: {caminho_base}")
    print(f"      Bot: {dados_base.get('nome_bot')}")
    print("=" * 72)
    print(f"Total de exemplos de treino: {len(X)}")
    print(f"Total de intenções/classes:  {len(classes_ordenadas)}")
    print(f"Estratégia de validação:     Validação Cruzada Estratificada ({n_splits}-Folds)")
    print("-" * 72)
    print(f">>> ACURÁCIA GERAL: {acuracia * 100:.2f}% ({accuracy_score(y, y_pred, normalize=False)}/{len(X)} acertos)")
    print("-" * 72)
    print("\nRelatório de Classificação (Precision, Recall, F1-Score):")
    print(report)
    imprimir_matriz_confusao(cm.tolist(), classes_ordenadas)
    print("=" * 72)


if __name__ == "__main__":
    main()
