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



# import os

# folder_path = '/path/to/your/folder'
# files = os.listdir(folder_path)
@tool
def pdf_converter(filename: str) -> str:
    """
    FERRAMENTA OBRIGATÓRIA para consultar documentos universitários.
    Use para qualquer pergunta sobre regulamentos ou vida acadêmica.
    Parâmetro: nome do arquivo (ex: 'regulamento', 'regulamento.pdf')
    A ferramenta automaticamente processa PDFs e JSONs.
    """
    try:
        # Define o diretório base onde os PDFs ficam armazenados
        base_dir = Path(__file__).parent.parent / "documentos"  # pasta documentos na raiz do projeto
        
        # Processa o nome do arquivo
        if filename.endswith(".json"):
            json_path = base_dir / filename
            pdf_path = json_path.with_suffix(".pdf")
        elif filename.endswith(".pdf"):
            pdf_path = base_dir / filename
            json_path = pdf_path.with_suffix(".json")
        else:
            # Caso sem extensão se assume que o pdf deve ser lido
            pdf_path = base_dir / f"{filename}.pdf"
            json_path = base_dir / f"{filename}.json"

        # Verifica se o PDF existe
        if not pdf_path.exists(): 
            return f"PDF não encontrado em: {pdf_path}. Certifique-se de que o arquivo está na pasta 'documentos'."
        
        # Se não existe o JSON, converte o pdf
        if not json_path.exists():
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
    
    except Exception as e:
        return f"Erro ao processar o arquivo: {str(e)}"



tools = [pdf_converter] #Lista de ferramentas que podem ser usadas pelo agente

# As primeiras linhas do template servem como instrunções "persistentes" para o modelo 
#enquanto que o Histórico de mensagens pode ser gerenciado de forma que mensagens mais antigas sejam removidas#
template = """Você é um assistente universitário especializado em regulamentos da universidade.
Você deve responder de forma clara e objetiva, sempre consultando o regulamento da universidade.
Explicar termos técnicos ou processos complexos de forma didática.

REGRAS OBRIGATÓRIAS:
1. SEMPRE que o usuário perguntar sobre regulamentos, vida acadêmica, ou mencionar documentos, USE IMEDIATAMENTE a ferramenta pdf_converter.
2. Para usar a ferramenta: pdf_converter("regulamento.pdf") ou pdf_converter("regulamento")
3. NUNCA peça o nome do arquivo - sempre tente "regulamento" primeiro.
4. A ferramenta pdf_converter lida automaticamente com PDFs e JSONs.
5. NUNCA diga que não tem acesso aos documentos.
6. SEMPRE use a ferramenta ANTES de responder qualquer pergunta sobre regulamentos.
7. NUNCA sugira ao usuário procurar o documento em outro lugar - você TEM acesso ao documento, EXCETO quando:
   - A pergunta não é sobre regulamentação (ex: "qual o sentido da vida?")
   - O usuário pergunta sobre regulamentos de OUTRAS instituições (ex: FACISA, UEPB)
   - O usuário pergunta sobre regulamentos ESPECÍFICOS de cursos (que não estão no regulamento geral)
8. NUNCA recomende "leitura completa" ou "entre em contato com a secretaria" - responda diretamente baseado no documento.

EXCEÇÕES para sugestão de novos documentos:
- Se perguntarem sobre OUTRAS universidades: responda com base na UFCG e sugira que disponibilizem o regulamento da instituição específica.
- Se perguntarem sobre regras ESPECÍFICAS de cursos: dê a regra geral da UFCG e sugira que disponibilizem o regulamento do curso específico.

Exemplo correto para outras instituições:
Usuário: "Como funciona na FACISA?"
Você: [usa pdf_converter("regulamento")] + "Esta informação é baseada no regulamento da UFCG. Para informações específicas da FACISA, você pode disponibilizar o regulamento desta instituição."

Exemplo correto para cursos específicos:
Usuário: "Quantas horas de extensão preciso em Ciência da Computação?"
Você: [usa pdf_converter("regulamento")] + "Segundo o regulamento geral da UFCG, alunos devem integralizar pelo menos X% das horas totais como extensão. Para informações específicas do curso de Ciência da Computação, você pode disponibilizar o regulamento específico do curso."

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
    tools=tools  
)

agent_executor = AgentExecutor(agent=agent, tools=tools)

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