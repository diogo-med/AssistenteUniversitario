from dotenv import load_dotenv
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_google_genai import ChatGoogleGenerativeAI #Classe dos modelos de chat da gemini
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Classe para criar templates de prompts de chat, MessagesPlaceholder é um marcador de posição para mensagens que serão preenchidas posteriormente
from langchain_core.runnables.history import RunnableWithMessageHistory #Gerenciador do histórico de mensagens
from langchain_core.chat_history import BaseChatMessageHistory #Classe base para o histórico de mensagens de chat
from langchain_community.chat_message_histories import ChatMessageHistory #Classe para o histórico de mensagens de chat
from langchain_core.tools import tool #Decorator para criar ferramentas que podem ser usadas pelo agente
from langchain.agents import AgentExecutor, create_openai_functions_agent 
import json

load_dotenv()



# import os

# folder_path = '/path/to/your/folder'
# files = os.listdir(folder_path)
@tool
def pdf_converter(absolute_path: str) -> str:
    """Converte um arquivo PDF para JSON usando o DocumentConverter."""
    # absolute_path = "RegulamentacaoDoEnsinoDeGrad"

    if absolute_path.endswith(".json"):
        json_path = Path(absolute_path)
        pdf_path = json_path.with_suffix(".pdf")############
    elif absolute_path.endswith(".pdf"):
        pdf_path = Path(absolute_path)
        json_path = pdf_path.with_suffix(".json")
    else:
        # Caso sem extensão se assume que o pdf deve ser lido
        pdf_path = Path(absolute_path).with_suffix(".pdf")
        json_path = pdf_path.with_suffix(".json")

    

    # Verifica se o PDF e o JSON existem
    if not pdf_path.exists() and not json_path.exists(): 
        return f"PDF não encontrado em: {pdf_path}"
    
    # Se não existe o JSON, converte o pdf
    elif not json_path.exists():
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        # Extrai o documento
        document = result.document
        json_data = document.export_to_dict()
        # Salva o novo JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    # Se o JSON já existe, lê o conteúdo
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    
    return json.dumps(json_data, indent=2, ensure_ascii=False)



tools = [pdf_converter] #Lista de ferramentas que podem ser usadas pelo agente

# As primeiras linhas do template servem como instrunções "persistentes" para o modelo 
#enquanto que o Histórico de mensagens pode ser gerenciado de forma que mensagens mais antigas sejam removidas#
template = """Você é um assistente universitário que ajuda alunos a entenderem melhor o regulamento da universidade.
Você deve responder de forma clara e objetiva, sempre se referindo ao regulamento da universidade.
Explicar termos técnicos ou processos complexos de forma didática
Você deve responder apenas com o conteúdo do regulamento, sem inventar informações.

"""



# 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2) #"criatividade" do modelo, quanto mais baixo, mais restrito

# O propmpt consiste essencialmente das instruções iniciais do modelo, das mensagens antigas e da nova mensagem, apesar
#dos chats online darem a ilusão de que a plataforma está se lembrando da conversa, na verdade ela precisa processar 
#todo o histórico a cada nova mensagem.#
prompt = ChatPromptTemplate.from_messages([ 
    ("system", template), #formata a variável template para que possa ser utilizada pela llm
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")]) #formata o imput do usuário

agent = create_openai_functions_agent(
    llm=llm,
    prompt=prompt,
    tools=tools  
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

chain = prompt | llm #pipeline que conecta o prompt ao llm

# 
historico = {}

# 
def get_session_history(session_id: str) -> BaseChatMessageHistory: # vai retornar o histórico de mensagens para um um determinado usuário
    if session_id not in historico:
        historico[session_id] = ChatMessageHistory(session_id=session_id)
    return historico[session_id]

# 
chain_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


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