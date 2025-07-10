# AssistenteUniversitario
Este projeto é um Agente de Inteligência Artificial que utiliza RAG (Retrieval-Augmented Generation) para ajudar alunos a entenderem melhor a regulamentação da universidade.
Ele recebe como entrada o PDF da ementa ou regulamento, converte-o em JSON e responde perguntas exclusivamente com base neste conteúdo.

## Tecnologias Utilizadas

* Python 3
* LangChain (para orquestração do LLM)
* Google Gemini API (modelo de linguagem)
* docling (para converter PDF → JSON)
* dotenv (para variáveis de ambiente)
* Pathlib (para manipulação de arquivos)
