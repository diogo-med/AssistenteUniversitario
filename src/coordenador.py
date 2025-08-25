from dotenv import load_dotenv
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_google_genai import ChatGoogleGenerativeAI #Classe dos modelos de chat da gemini
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Classe para criar templates de prompts de chat, MessagesPlaceholder é um marcador de posição para mensagens que serão preenchidas posteriormente
from langchain_core.runnables.history import RunnableWithMessageHistory #Gerenciador do histórico de mensagens
from langchain_core.chat_history import BaseChatMessageHistory #Classe base para o histórico de mensagens de chat
from langchain_community.chat_message_histories import ChatMessageHistory #Classe para o histórico de mensagens de chat
from langchain_core.tools import tool #Decorator para criar ferramentas que podem ser usadas pelo agente
from langchain.agents import AgentExecutor, create_tool_calling_agent 
import json

load_dotenv()


@tool
def get_files_names() -> str:
    """
    Lista todos os arquivos disponíveis na pasta documentos em ordem alfabética.
    Use esta ferramenta SEMPRE PRIMEIRO antes de qualquer consulta sobre regulamentação.
    """
    try:
        base_dir = Path(__file__).parent.parent / "documentos"
        if not base_dir.exists():
            return "Pasta documentos não encontrada."
            
        files = sorted([f.name for f in base_dir.iterdir() if f.is_file()])
        
        if not files:
            return "Nenhum arquivo encontrado na pasta documentos."
        
        return f"Arquivos disponíveis: {', '.join(files)}"
        
    except Exception as e:
        return f"Erro ao listar arquivos: {str(e)}"


@tool
def pdf_converter(filename: str) -> str:
    """
    FERRAMENTA OBRIGATÓRIA para consultar documentos universitários.
    Use para qualquer pergunta sobre regulamentos ou vida acadêmica.
    Parâmetro: nome do arquivo (ex: 'regulamento', 'regulamento.pdf')
    A ferramenta automaticamente processa PDFs e JSONs.
    """
    try:
        base_dir = Path(__file__).parent.parent / "documentos"
        
        # Processa o nome do arquivo
        filename = filename.strip()
        if filename.endswith(".json"):
            json_path = base_dir / filename
            pdf_path = json_path.with_suffix(".pdf")
        elif filename.endswith(".pdf"):
            pdf_path = base_dir / filename
            json_path = pdf_path.with_suffix(".json")
        else:
            pdf_path = base_dir / f"{filename}.pdf"
            json_path = base_dir / f"{filename}.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        else:
            if not pdf_path.exists():
                available_files = [f.name for f in base_dir.glob("*.pdf")]
                return f"PDF não encontrado: {pdf_path.name}. Arquivos PDF disponíveis: {', '.join(available_files) if available_files else 'Nenhum'}"
            
            # Converte o PDF
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            document = result.document
            json_data = document.export_to_dict()
            
            # Salva o JSON (se conseguir)
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Continua mesmo se não conseguir salvar
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    except Exception as e:
        return f"Erro ao processar o arquivo: {str(e)}"


tools = [pdf_converter,get_files_names] #Lista de ferramentas que podem ser usadas pelo agente

# As primeiras linhas do template servem como instrunções "persistentes" para o modelo 
#enquanto que o Histórico de mensagens pode ser gerenciado de forma que mensagens mais antigas sejam removidas#
template = """
INSTRUÇÕES OBRIGATÓRIAS:

1. Para perguntas sobre regulamentos universitários: SEMPRE use pdf_converter("regulamento")
2. Responda APENAS com base no conteúdo retornado pela ferramenta
3. NUNCA diga que não tem acesso - você TEM a ferramenta pdf_converter

Você deve usar as ferramentas disponíveis para responder.
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
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")]) #necessário para o agente processar suas ações e ferramentas

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools  #deixa o agente ciente de quais ferramentas estão disponíveis
)

# agent_executor = AgentExecutor(agent=agent, tools=tools) #dá autorizção pro agente usar as ferramentas
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    max_iterations=5,
    verbose=True  # Vamos ver o que o agente está fazendo
)
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
        
        print(f"\nCoordenador: {resposta['output']}\n")

if __name__ == "__main__":
    iniciar_conversa_com_coordenador()