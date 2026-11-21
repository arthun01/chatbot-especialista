# Chatbot Especialista Genérico (Offline & Baseado em Machine Learning Clássico)

Projeto acadêmico de Inteligência Artificial implementando um **Chatbot Especialista Genérico** em Python. O sistema é 100% agnóstico ao tema, operando com total desacoplamento entre o **Motor de Inferência (PLN/ML)** e a **Base de Conhecimento (JSON)**.

O projeto **não utiliza** APIs externas de LLMs (OpenAI, Gemini, etc.) nem pesos neurais pesados. Utiliza exclusivamente Machine Learning clássico com **scikit-learn** (`TF-IDF + Regressão Logística`), sendo leve, offline e treinado instantaneamente em tempo de execução ao inicializar.

---

## 🏛 Arquitetura do Sistema

```
chatbot-especialista/
├── bases/
│   ├── f1.json                # Base de conhecimento de exemplo (Fórmula 1)
│   └── template.json          # Molde documentado para novos temas
├── src/
│   ├── __init__.py            # Exportações do pacote
│   ├── preprocessor.py        # Normalização textual (Unicode NFKD, Regex)
│   ├── loader.py              # Leitura e validação rigorosa do JSON
│   └── engine.py              # Motor NLP (TF-IDF, Regressão Logística, Entidades)
├── main.py                    # Interface CLI interativa (--base e --debug)
├── evaluate.py                # Avaliação de métricas com Validação Cruzada
├── requirements.txt           # Apenas scikit-learn
└── README.md                  # Documentação do projeto
```

### Funcionamento do Motor de Inferência (`src/engine.py`):
1. **Normalização Textual (`src/preprocessor.py`):** Converte o texto para minúsculas, decompõe diacríticos com `unicodedata` (NFKD) para remover acentos, elimina pontuações e colapsa espaços em branco repetidos.
2. **Vetorização com TF-IDF:** Extrai características numéricas baseadas em unigramas e bigramas (`ngram_range=(1, 2)`) com escala sublinear (`sublinear_tf=True`), capturando contexto e pares de palavras.
3. **Classificação com Regressão Logística:** Ajusta os pesos das intenções (`C=10.0, max_iter=500`). Fornece estimativas probabilísticas de pertinência via `predict_proba`.
4. **Limiar de Confiança (`limiar_confianca`):** Se a probabilidade máxima da intenção predita for inferior ao limiar definido no JSON, o chatbot dispara a mensagem de `fallback`.
5. **Separação de Intenções:**
   - **Intenções Fixas:** O chatbot seleciona aleatoriamente uma das respostas cadastradas na lista `respostas`.
   - **Intenções Dinâmicas (com Fatos):** O motor busca uma das chaves do dicionário `fatos` dentro do texto normalizado (respeitando fronteiras de palavras `\b` e priorizando as chaves mais específicas). Se encontrada, preenche o `template_resposta` com `{entidade}` e `{valor}`. Caso contrário, responde com a mensagem `erro_entidade`.

---

## 🚀 Como Configurar e Executar

### 1. Criar e Ativar o Ambiente Virtual

No Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar as Dependências

```bash
pip install -r requirements.txt
```

*(O único pacote externo exigido é o `scikit-learn`)*

---

## 💬 Executando o Chatbot (`main.py`)

### Modo Padrão (Base de Fórmula 1):
```bash
python main.py
```

### Modo de Depuração (`--debug`):
Exibe o texto normalizado, a intenção predita com sua porcentagem de confiança e a entidade capturada antes de cada resposta:
```bash
python main.py --debug
```

Exemplo de saída no console com `--debug`:
```text
Você: quem foi o campeao de 1988?
[DEBUG] Norm: 'quem foi o campeao de 1988' | Intent: campeao_ano (93.4%) | Entidade: '1988'
F1-Bot: Em 1988, o campeão mundial de Fórmula 1 foi Ayrton Senna (McLaren).
```

### Usando outra Base de Conhecimento:
```bash
python main.py --base bases/f1.json --debug
```

Para encerrar a conversa, digite `sair` ou `exit` a qualquer momento, ou envie uma mensagem de despedida (ex: *"tchau"*).

---

## 📊 Avaliação do Modelo (`evaluate.py`)

Para medir o desempenho acadêmico do classificador, execute o script de validação:

```bash
python evaluate.py
```

Ou especificando outra base:
```bash
python evaluate.py --base bases/f1.json
```

O script realiza **Validação Cruzada Estratificada (`StratifiedKFold`)** sobre todos os exemplos da base e gera:
- **Acurácia Geral (%)**
- **Relatório de Classificação Completo:** Precisão, Recall e F1-Score por intenção
- **Matriz de Confusão Formatada** com numeração de linhas e colunas

---

## 📝 Como Criar uma Nova Base de Conhecimento

Para criar um chatbot especialista em outro tema (por exemplo, Medicina, Culinária, História, Cinema):

1. Copie o arquivo molde [`bases/template.json`](bases/template.json):
   ```bash
   cp bases/template.json bases/cinema.json
   ```
2. Abra o novo arquivo JSON e configure:
   - `nome_bot`: Nome do assistente (ex: `"CineBot"`).
   - `limiar_confianca`: Ponto de corte para fallback (ex: `0.40`).
   - `saudacao_inicial`: Mensagem exibida na abertura.
   - `fallback`: Resposta caso o bot não compreenda a pergunta.
   - `intencoes`: Lista de intenções:
     - **Intenção Fixa:** Contém `"tag"`, `"exemplos"` (ao menos 6 a 8 frases variadas) e `"respostas"`.
     - **Intenção Dinâmica:** Contém `"tag"`, `"exemplos"`, `"template_resposta"`, `"erro_entidade"` e `"fatos"` (dicionário de `"Chave": "Valor"`).
3. Execute o chatbot apontando para a sua nova base:
   ```bash
   python main.py --base bases/cinema.json --debug
   ```
