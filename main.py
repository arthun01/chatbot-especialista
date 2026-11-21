"""
Interface de Linha de Comando (CLI) para interação com o Chatbot Especialista.

Permite selecionar a base de conhecimento (--base) e ativar o modo de depuração (--debug)
para inspecionar a normalização, intenção predita, confiança percentual e entidade capturada.
"""

import argparse
import sys
from pathlib import Path

from src.loader import load_base
from src.engine import ChatbotEngine


def parse_arguments() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Chatbot Especialista Genérico com Machine Learning Clássico (scikit-learn)."
    )
    parser.add_argument(
        "--base",
        type=str,
        default="bases/f1.json",
        help="Caminho do arquivo JSON da base de conhecimento (padrão: bases/f1.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa o modo de depuração com exibição da normalização, intenção, confiança e entidade.",
    )
    return parser.parse_args()


def main() -> None:
    """Fluxo principal de execução do chatbot via terminal."""
    args = parse_arguments()
    caminho_base = Path(args.base)

    try:
        print(f"[*] Carregando base de conhecimento: {caminho_base} ...")
        dados_base = load_base(caminho_base)
        print("[*] Treinando motor de inferência (TF-IDF + Regressão Logística)...")
        engine = ChatbotEngine(dados_base)
        print("[*] Treinamento concluído com sucesso!\n")
    except Exception as erro:
        print(f"\n[ERRO CRÍTICO] Falha ao inicializar o chatbot: {erro}", file=sys.stderr)
        sys.exit(1)

    nome_bot = engine.nome_bot

    print("=" * 64)
    print(f"  {nome_bot} - CHATBOT ESPECIALISTA (MODO CLI)")
    print(f"  Base ativa: {caminho_base}")
    if args.debug:
        print("  Modo Debug: ATIVADO")
    print("  Digite 'sair' ou 'exit' a qualquer momento para encerrar.")
    print("=" * 64)

    # Saudação inicial configurada na base
    print(f"\n{nome_bot}: {engine.saudacao_inicial}")

    comandos_saida = {"sair", "exit", "quit"}

    while True:
        try:
            mensagem_usuario = input("\nVocê: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{nome_bot}: Conversa interrompida pelo usuário. Até mais!")
            break

        if not mensagem_usuario:
            continue

        if mensagem_usuario.lower() in comandos_saida:
            print(f"\n{nome_bot}: Encerrando sessão. Até a próxima!")
            break

        resultado = engine.processar(mensagem_usuario)

        if args.debug:
            norm_str = resultado["texto_normalizado"]
            intent_str = resultado["intencao"]
            conf_str = f"{resultado['confianca'] * 100:.1f}%"
            entidade_str = (
                f"'{resultado['entidade_encontrada']}'"
                if resultado["entidade_encontrada"] is not None
                else "None"
            )
            print(
                f"[DEBUG] Norm: '{norm_str}' | Intent: {intent_str} ({conf_str}) | Entidade: {entidade_str}"
            )

        print(f"{nome_bot}: {resultado['resposta']}")

        # Encerra o loop se a intenção for 'despedida' detectada com confiança acima do limiar
        if (
            resultado["intencao"] == "despedida"
            and resultado["confianca"] >= engine.limiar_confianca
        ):
            break


if __name__ == "__main__":
    main()
