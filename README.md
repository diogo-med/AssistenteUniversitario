# Assistente Universitário

Este projeto é um assistente de Inteligência Artificial que utiliza a arquitetura RAG (Retrieval-Augmented Generation) para responder perguntas sobre documentos universitários, como regulamentos e editais.

## Como Funciona

O fluxo de operação é o seguinte:
1.  **Processamento de Documentos**: Um documento PDF é carregado, dividido em pedaços menores (chunks) e processado para gerar *embeddings* (representações vetoriais).
2.  **Armazenamento**: Os embeddings e os textos correspondentes são armazenados em um banco de dados vetorial local (`ChromaDB`).
3.  **Busca e Resposta**: Quando uma pergunta é feita, o sistema a converte em um embedding e busca os chunks mais similares no banco de dados.
4.  **Geração de Resposta**: O contexto recuperado é enviado para um Large Language Model (LLM), como o Gemini, que gera uma resposta em linguagem natural.

## Arquitetura e Implementação

Esta seção detalha a estrutura do projeto e os componentes técnicos para desenvolvedores que desejam entender a implementação.

### Estrutura do Projeto

-   `src/coordenador.py`: Ponto de entrada do assistente. Contém a lógica principal do agente, a definição das ferramentas e o loop de conversação interativo.
-   `documents/`: Diretório para armazenar os arquivos PDF que serão processados e indexados.
-   `chroma_db/`: Diretório onde o ChromaDB armazena os dados vetoriais localmente. Este diretório não deve ser versionado.
-   `requirements.txt`: Lista de dependências Python do projeto.

### Ferramentas do Agente

O assistente utiliza um agente que interage com um conjunto de ferramentas para executar suas tarefas:

-   `pdf_embedding(filename: str)`: Processa um arquivo PDF da pasta `documents/`. A implementação divide o texto em chunks de 800 caracteres com uma sobreposição de 100 caracteres, gera os embeddings e os armazena no ChromaDB.
-   `search_in_document(question: str, document_name: str)`: Busca uma resposta para a `question` dentro de um `document_name` já processado no banco vetorial.
-   `list_available_documents()`: Lista todos os documentos que já foram processados e estão disponíveis para consulta.

## Como Usar

Siga os passos abaixo para configurar e executar o projeto.

### Requisitos

-   Python 3.11+
-   Uma chave de API do Google Gemini, configurada como variável de ambiente.

### Passos para Execução

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd AssistenteUniversitario
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # No Windows: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure a chave de API:**
    Crie um arquivo `.env` na raiz do projeto e adicione sua chave:
    ```
    GOOGLE_API_KEY="SUA_API_KEY_AQUI"
    ```

5.  **Adicione um documento:**
    Coloque um arquivo PDF (ex: `regulamento.pdf`) na pasta `documents/`.

6.  **Execute o assistente:**
    ```bash
    python3 src/coordenador.py
    ```
    O assistente iniciará em modo interativo no seu terminal, pronto para receber comandos como "Processe o documento regulamento.pdf" ou "Quais são as regras para trancamento de matrícula no regulamento?".

## Considerações Técnicas

-   **Banco de Dados Vetorial**: O projeto utiliza o `ChromaDB` para armazenamento local dos embeddings. Para evitar que o banco de dados seja incluído no controle de versão, o diretório `chroma_db/` deve ser adicionado ao `.gitignore`.
    ```bash
    echo "chroma_db/" >> .gitignore
    git rm -r --cached chroma_db
    git commit -m "Stop tracking chroma_db"
    ```

# AssistenteUniversitario

AssistenteUniversitario é um projeto de Perguntas & Respostas (RAG) para documentos universitários — principalmente regulamentos.

Visão rápida:
- Converter PDFs -> fragmentar (chunking) -> gerar embeddings -> armazenar em ChromaDB local.
- Perguntas são respondidas buscando por similaridade no ChromaDB e resumindo o contexto com a LLM.

Arquivos principais:
- `src/coordenador.py` — entrada do assistente e definição das ferramentas (`pdf_embedding`, `search_in_document`, `list_available_documents`).
- `documents/` — coloque aqui seus PDFs para indexação.
- `chroma_db/` — pasta com banco local do ChromaDB (não versionar).

Requisitos:
- Python 3.11+
- Instale dependências com `pip install -r requirements.txt`.
- Defina variáveis de ambiente para credenciais da API do provedor LLM (ex.: Google Gemini) conforme sua configuração.

Quickstart:
1. Crie/ative um ambiente Python (ex.: `conda create -n agentsenv python=3.11 && conda activate agentsenv`).
2. Instale dependências: `pip install -r requirements.txt`.
3. Coloque um PDF em `documents/` (ex.: `documents/regulamento.pdf`).
4. Rode o assistente interativo:
   - `cd src`
   - `python3 coordenador.py`

Operações úteis:
- Indexar manualmente (REPL):
  - `python3 -c "from src.coordenador import pdf_embedding; print(pdf_embedding('regulamento.pdf'))"`
- Depurar busca:
  - `python3 -c "from src.coordenador import search_in_document; print(search_in_document('Como funciona a matrícula?', 'regulamento'))"`

Boas práticas:
- Não commite a pasta `chroma_db/`.
  - Para ignorar e remover do índice:
    ```bash
    echo "chroma_db/" >> .gitignore
    git rm -r --cached chroma_db || true
    git add .gitignore
    git commit -m "Ignore local ChromaDB files"
    ```
- Ajuste `chunk_size` e `chunk_overlap` em `pdf_embedding` (recomendado 600–1000 chars, overlap 100–200).
- Sempre valide `metadata` antes de usar `.get()` para evitar `NoneType`.

Depuração do agente:
- Se o agente não estiver chamando `search_in_document` consistentemente, ative `verbose=True` no `AgentExecutor` ou converta para uma cadeia RAG explícita que sempre executa a busca antes do LLM.

Próximos passos sugeridos:
- Extrair indexação para um script dedicado (`scripts/index_pdf.py`).
- Adicionar testes unitários para `pdf_embedding` e `search_in_document`.
- Remover importações não usadas (ex.: `docling`) se não forem necessárias.

Contribuições: abra issues ou PRs pequenas e focadas. Inclua testes quando possível.

## Ferramentas Disponíveis

O assistente utiliza as seguintes ferramentas para interagir com os documentos:

-   `pdf_embedding(filename: str)`: Processa um arquivo PDF da pasta `documents/`, gera os embeddings e os salva no ChromaDB.
-   `search_in_document(question: str, document_name: str)`: Busca uma resposta para a `question` dentro de um `document_name` já processado.
-   `list_available_documents()`: Lista todos os documentos que já foram processados e estão disponíveis para consulta.

## Requisitos

-   Python 3.11+
-   Dependências que podem ser instaladas via `pip`.
-   Uma chave de API do Google configurada em um arquivo `.env` na raiz do projeto.

Exemplo de `.env`:
```
GOOGLE_API_KEY="SUA_API_KEY_AQUI"
```

## Como Configurar e Executar

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd AssistenteUniversitario
    ```

2.  **Crie um ambiente virtual e ative-o:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # No Windows: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    *(Nota: Crie um arquivo `requirements.txt` se ele não existir)*
    ```bash
    pip install langchain-google-genai chromadb pypdf python-dotenv
    ```

4.  **Adicione um documento**:
    Coloque um arquivo PDF (ex: `regulamento.pdf`) na pasta `documents/`.

5.  **Execute o assistente:**
    ```bash
    python3 src/coordenador.py
    ```

O assistente iniciará em modo interativo no seu terminal.

## Boas Práticas

-   **Não versionar o ChromaDB**: Para garantir que o repositório permaneça leve, adicione `chroma_db/` ao seu arquivo `.gitignore`.
    ```bash
    echo "chroma_db/" >> .gitignore
    git rm -r --cached chroma_db
    git commit -m "Stop tracking chroma_db"
    ```
-   **Ajuste de Chunking**: Em `src/coordenador.py`, os parâmetros `chunk_size` (800) e `chunk_overlap` (100) podem ser ajustados para otimizar a recuperação de informações dependendo da estrutura do seu documento.
