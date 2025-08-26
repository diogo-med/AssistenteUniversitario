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
from langchain_text_splitters import RecursiveCharacterTextSplitter #CHUNKING
from langchain_google_genai import GoogleGenerativeAIEmbeddings #EMBEDDING
from langchain_community.document_loaders import PyPDFLoader
import chromadb

load_dotenv()


# Inicializa o cliente do ChromaDB -> cria um diretório para persistir os dados.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# define o modelo que vai gerar os embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@tool
def pdf_embedding(filename: str):
    
    """Processa um PDF: carrega, faz chunking, gera embeddings e armazena no ChromaDB.
    Use esta ferramenta quando precisar processar um novo documento PDF."""

    try:
        base_dir = Path(__file__).parent.parent / "documents"
        
        # Processa o nome do arquivo
        filename = filename.strip()
        filename = Path(filename).stem
        
        pdf_path = base_dir / f"{filename}.pdf"

        #verifica se o pdf existe
        if not pdf_path.exists():
            available_files = [f.name for f in base_dir.glob("*.pdf")]
            return f"PDF não encontrado: {pdf_path.name}. Arquivos PDF disponíveis: {', '.join(available_files) if available_files else 'Nenhum'}"
        
        #verifica se já foi feito o embedding do pdf(ou pelo menos de algum arquivo com o nome do pdf)
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if filename in existing_collections:
            return f"PDF '{filename}' já foi processado e está disponível para consulta."
        
        # Converte o PDF
        loader = PyPDFLoader(pdf_path)
        loaded_text = loader.load()

        # define tamanho dos chunks
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        )

        # processa a lista de páginas do loaded_text
        chunks = text_splitter.split_documents(loaded_text)

        # extrai o texto de cada chunk
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # gera os embeddings
        embeddings = embeddings_model.embed_documents(chunk_texts)
        collection = chroma_client.get_or_create_collection(name=filename)

        # Prepara metadados com informações da página
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "chunk_id": i,
                "page": chunk.metadata.get("page", 0)
            }
            metadatas.append(metadata)
        
        # IDs únicos para cada chunk
        chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunk_texts))]
        
        # Adiciona à coleção
        collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        return f"PDF '{filename}' processado com sucesso! {len(chunk_texts)} chunks criados e armazenados."
        
    except Exception as e:
        return f"Erro ao processar o arquivo: {str(e)}"

@tool
def search_in_document(question: str, document_name: str = "regulamento"):
    """
    Busca informações em um documento já processado usando similaridade semântica.
    Use esta ferramenta para responder perguntas baseadas no conteúdo dos documentos.
    
    Args:
        question: A pergunta ou consulta do usuário
        document_name: Nome do documento (sem extensão .pdf)
    """
    try:
        # Verifica se a coleção existe
        try:
            collection = chroma_client.get_collection(name=document_name)
        except Exception:
            # Tenta processar o documento automaticamente
            processing_result = pdf_embedding(f"{document_name}.pdf")
            if "sucesso" in processing_result.lower():
                collection = chroma_client.get_collection(name=document_name)
            else:
                return f"Documento '{document_name}' não encontrado. {processing_result}"
        
        # Gera embedding da pergunta
        query_embedding = embeddings_model.embed_query(question)
        
        # Busca por similaridade
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Top 5 resultados mais similares
            include=['documents', 'metadatas']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return f"Nenhuma informação relevante encontrada no documento '{document_name}' para a pergunta: {question}"
        
        # Formata os resultados
        relevant_context = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            if metadata and isinstance(metadata, dict):
                page = metadata.get('page', 'N/A')
            else:
                page = 'N/A'
            
            relevant_context.append(
                f"[Página {page}] {doc}"
            )
        
        text_context = "\n\n".join(relevant_context)
        
        return f"""Informações encontradas no documento '{document_name}':
        {text_context}
        [Baseado na busca por: "{question}"]"""
        
    except Exception as e:
        return f"Erro ao buscar no documento '{document_name}': {str(e)}"
    
@tool
def list_available_documents():
    """
    Lista todos os documentos que foram processados e estão disponíveis para consulta.
    """
    try:
        collections = chroma_client.list_collections()
        if not collections:
            return "Nenhum documento foi processado ainda. Use 'processar_pdf' para processar documentos."
        
        available_docs = [col.name for col in collections]
        return f"Documentos disponíveis para consulta: {', '.join(available_docs)}"
        
    except Exception as e:
        return f"Erro ao listar documentos: {str(e)}"

tools = [pdf_embedding,list_available_documents,search_in_document] #Lista de ferramentas que podem ser usadas pelo agente

# As primeiras linhas do template servem como instrunções "persistentes" para o modelo 
#enquanto que o Histórico de mensagens pode ser gerenciado de forma que mensagens mais antigas sejam removidas#
template = """
Você é um assistente especializado em responder perguntas sobre documentos universitários.

FLUXO DE TRABALHO:
1. Se o usuário fizer uma pergunta sobre regulamentos ou documentos:
   - Use 'search_in_document' com a pergunta e o nome do documento
   - O documento 'regulamento' é o padrão para perguntas sobre regulamentos universitários

2. Se o usuário quiser processar um novo PDF:
   - Use 'pdf_embedding' com o nome do arquivo

3. Se o usuário quiser saber quais documentos estão disponíveis:
   - Use 'list_available_documents'

REGRAS IMPORTANTES:
- SEMPRE use as ferramentas antes de responder
- Base suas respostas APENAS no conteúdo retornado pelas ferramentas
- Se uma ferramenta retornar erro, explique o problema e sugira uma solução
- Seja preciso e cite as páginas quando disponível
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

#dá autorizção pro agente usar as ferramentas
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    max_iterations=5,
    # verbose=True  
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