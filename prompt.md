Crie um projeto completo em Python de um **Chatbot Especialista Genérico (agnóstico ao tema)** para um trabalho acadêmico de Inteligência Artificial.

### Objetivo e Restrições
- O projeto deve separar completamente o **Motor de Inferência (código Python)** da **Base de Conhecimento (arquivos JSON)**.
- **NÃO** use APIs de LLMs externas (OpenAI, Gemini, etc.) nem modelos locais pesados.
- Use exclusivamente Machine Learning clássico com `scikit-learn` (apenas `scikit-learn` no `requirements.txt`) para garantir que o projeto seja leve, offline e rode facilmente na máquina do professor.
- O modelo deve ser treinado em tempo de execução ao iniciar o script (sem salvar arquivos `.pkl`), pois o treino é instantâneo e evita erros de versão.

### Estrutura de Pastas Obrigatória
```text
chatbot-especialista/
├── bases/
│   ├── f1.json                # Base de exemplo completa sobre Fórmula 1
│   └── template.json          # Molde vazio documentado para novos temas
├── src/
│   ├── __init__.py
│   ├── preprocessor.py        # Normalização de texto
│   ├── loader.py              # Leitura e validação do JSON da base
│   └── engine.py              # Motor de NLP (TF-IDF + Regressão Logística + extração de fatos)
├── main.py                    # Interface CLI com argparse (--base e --debug)
├── evaluate.py                # Script de métricas (Acurácia, F1-Score, Matriz de Confusão)
├── requirements.txt
└── README.md
Contrato do JSON da Base de Conhecimento
O motor deve suportar dois tipos de intenções no mesmo arquivo JSON:

Intenções Fixas: retornam uma resposta aleatória da lista respostas.

Intenções Dinâmicas (com Fatos): procuram uma chave do dicionário fatos dentro do texto normalizado do usuário e preenchem o template_resposta com {entidade} e {valor}. Se não encontrar a entidade na frase, retorna erro_entidade.

Estrutura esperada do JSON:

JSON
{
  "nome_bot": "F1-Bot",
  "limiar_confianca": 0.40,
  "saudacao_inicial": "Olá! Sou especialista em Fórmula 1. Pergunte algo!",
  "fallback": "Não entendi bem. Pode reformular a pergunta?",
  "intencoes": [
    {
      "tag": "saudacao",
      "exemplos": ["oi", "ola", "bom dia", "e ai"],
      "respostas": ["Fala! O que quer saber sobre F1?", "Olá! Mande sua dúvida."]
    },
    {
      "tag": "campeao_ano",
      "exemplos": ["quem ganhou em", "qual foi o campeao de", "quem levou o titulo em"],
      "template_resposta": "Em {entidade}, o campeão foi {valor}.",
      "erro_entidade": "De qual ano você está falando? Não encontrei esse ano na minha base.",
      "fatos": {
        "1988": "Ayrton Senna (McLaren)",
        "2021": "Max Verstappen (Red Bull)"
      }
    }
  ]
}
Especificação dos Módulos
src/preprocessor.py:

Função normalizar(texto: str) -> str usando unicodedata e re para remover acentos, pontuação, espaços extras e converter para minúsculas.

src/loader.py:

Carrega o JSON, valida se possui os campos obrigatórios (nome_bot, intencoes, etc.) e levanta erros claros caso o arquivo não exista ou esteja malformado.

src/engine.py:

Classe ChatbotEngine que recebe os dados carregados pelo loader.

Treina um Pipeline do scikit-learn com TfidfVectorizer(ngram_range=(1, 2)) e LogisticRegression(C=10.0, max_iter=500).

Método processar(mensagem: str) -> dict que retorna um dicionário contendo: resposta, texto_normalizado, intencao, confianca (float) e entidade_encontrada (str ou None).

Se confianca < limiar_confianca, retorna a mensagem de fallback.

Para intenções dinâmicas, normalize também as chaves de fatos na hora da busca para garantir o match mesmo com acentos/maiúsculas.

main.py:

Interface de terminal (CLI) usando argparse com os argumentos:

--base (caminho do JSON, padrão bases/f1.json)

--debug (flag booleana; quando ativa, imprime [DEBUG] Norm: '...' | Intent: ... (XX.X%) | Entidade: ... antes da resposta do bot).

Encerra o loop se o usuário digitar sair/exit ou se a intenção detectada for despedida acima do limiar.

evaluate.py:

Aceita --base via argparse.

Executa validação cruzada (StratifiedKFold ou cross_val_predict) sobre os exemplos da base e imprime no terminal o classification_report (precisão, recall, F1-score), a acurácia geral e a matriz de confusão formatada.

bases/f1.json:

Crie uma base bem completa de Fórmula 1 com pelo menos 6 intenções variadas (saudação, despedida, campeão por ano, títulos por piloto, regras como DRS/Safety Car, circuitos famosos) e pelo menos 6 a 8 exemplos de treino por intenção para o modelo generalizar bem.

README.md:

Documentação curta e direta em português explicando a arquitetura (TF-IDF + Regressão Logística), como criar o ambiente virtual, instalar o requirements.txt, rodar o chat com --debug, rodar o evaluate.py e como criar uma nova base de conhecimento.

Gere todos os arquivos com código limpo, tipado (type hints) e bem comentado.