
from pathlib import Path
from docling.document_converter import DocumentConverter
import json

def pdf_conveter(nome_do_arquivo):
    # nome_do_arquivo = "RegulamentacaoDoEnsinoDeGrad"
    if not nome_do_arquivo.endswith(".pdf"):
        print(nome_do_arquivo)
        nome_do_arquivo+=".pdf"
        print (nome_do_arquivo)
    pdf_path = Path("//home/diogomedeiros/personalProjects/au/AssistenteUniversitarioData/" + str(nome_do_arquivo))
    print(pdf_path)
    json_path = pdf_path.with_suffix(".json")
    print(json_path)
  
    if not pdf_path.exists():
        print(1)
        print(f"PDF não encontrado em: {pdf_path}")
        exit(1)
    if json_path.exists():
        print(2)
        print(f"JSON já existe: {json_path}")
    else:
        print(3)
        # Se não existe, converte
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        # Extrai o documento
        document = result.document
        json_data = document.export_to_dict()
        # Mostra no terminal
        print(json.dumps(json_data, indent=2, ensure_ascii=False))
        # Salva o novo JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Novo JSON criado em: {json_path}")


# pdf_conveter("historico_123110779")


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI #Classe dos modelos de chat da gemini
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Classe para criar templates de prompts de chat, MessagesPlaceholder é um marcador de posição para mensagens que serão preenchidas posteriormente
from langchain_core.runnables.history import RunnableWithMessageHistory #Gerenciador do histórico de mensagens
from langchain_core.chat_history import BaseChatMessageHistory #Classe base para o histórico de mensagens de chat
from langchain_community.chat_message_histories import ChatMessageHistory #Classe para o histórico de mensagens de chat

# As primeiras linhas do template servem como instrunções "persistentes" para o modelo 
#enquanto que o Histórico de mensagens pode ser gerenciado de forma que mensagens mais antigas sejam removidas#
template = """Você é um assistente universitário que ajuda alunos a entenderem melhor o regulamento da universidade.
Você deve responder de forma clara e objetiva, sempre se referindo ao regulamento da universidade.
Você deve responder apenas com o conteúdo do regulamento, sem inventar informações.




"""

# Histórico de mensagens:{chat_history} 

# Entrada do usuário: {input}





#-Histórico de mensagens é utilizado para que haja coerência entre as mensagens, permitindo que o modelo tenha contexto sobre a conversa anterior.
#-Pergunta mais recente do usuário que dever ser respondida.

# O propmpt consiste essencialmente das instruções iniciais do modelo, das mensagens antigas e da nova mensagem, apesar
#dos chats online darem a ilusão de que a plataforma está se lembrando da conversa, na verdade ela precisa processar 
#todo o histórico a cada nova mensagem.#
prompt = ChatPromptTemplate.from_messages([ 
    ("system", template), #formata a variável template para que possa ser utilizada pela llm
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")]) #formata o imput do usuário

# 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2) #"criatividade" do modelo, quanto mais baixo, mais restrito

# 
chain = prompt | llm #pipeline que conecta o prompt ao llm

# 
historico = {}

# 
def get_session_history(session_id: str) -> BaseChatMessageHistory: # vai retornar o histórico de mensagens para um um determinado usuário
    if session_id not in historico:
        historico[session_id] = ChatMessageHistory(session_id=session_id)
    return historico[session_id]

# 
chain_with_history = RunnableWithMessageHistory( # vai juntar o histórico de mensagens com o processamento
    chain,
    get_session_history,
    imput_message_key = "input", 
    history_messages_key = "history")

# 
def iniciar_conversa_com_coordenador():
    print("Como posso ajudar você hoje? Digite 'sair' para encerrar\n")
    while True:
        duvida = input("Você: ")
        if duvida.lower() in ["sair"]: #Encerra a conversa qnd o usuário escreve 'sair'
            print("Conversa encerrada.")
            break
        
        resposta = chain_with_history.invoke( 
            {"input": duvida}, 
            config = {"configurable": {"session_id": "user123"}})
        
        print(f"Coordenador: {resposta.content}")

if __name__ == "__main__":
    iniciar_conversa_com_coordenador()