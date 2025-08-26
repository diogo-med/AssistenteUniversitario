from langchain_text_splitters import RecursiveCharacterTextSplitter #CHUNKING
from langchain_google_genai import GoogleGenerativeAIEmbeddings #EMBEDDING
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
def pdf_embedding(filename):

    try:
        base_dir = Path(__file__).parent.parent / "documentos"
        
        # Processa o nome do arquivo
        filename = filename.strip()
        filename = Path(filename).stem
        
        pdf_path = base_dir / f"{filename}.pdf"

        if not pdf_path.exists():
            available_files = [f.name for f in base_dir.glob("*.pdf")]
            return f"PDF não encontrado: {pdf_path.name}. Arquivos PDF disponíveis: {', '.join(available_files) if available_files else 'Nenhum'}"
            
        # Converte o PDF
        loader = PyPDFLoader(pdf_path)
        loaded_text = loader.load()

        create_embedding(chunking(loaded_text), filename)
        return "deu certo"
        
    except Exception as e:
        return f"Erro ao processar o arquivo: {str(e)}"


def chunking(loaded_text):
    # define tamanho dos chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    # processa a lista de páginas do loaded_text
    chunks = text_splitter.split_documents(loaded_text)

    # extrai o texto de cada chunk
    chunk_texts = [chunk.page_content for chunk in chunks]

    return chunk_texts
    

def create_embedding(chunk_texts, nome):
    # define o modelo que vai gerar os embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # gera os embeddings 
    vetores = embeddings_model.embed_documents(chunk_texts)

    # Inicializa o cliente do ChromaDB -> cria um diretório para persistir os dados.
    client = chromadb.PersistentClient(path="./chroma_db")

    # 
    collection_name = nome
    collection = client.get_or_create_collection(name=collection_name)

    # Prepara os IDs para cada chunk. É necessário que sejam strings únicas.
    chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]

    # Adiciona os vetores e os documentos (chunks) à coleção
    # Note que estamos armazenando o texto original nos metadados
    collection.add(
        embeddings=vetores,
        documents=chunk_texts,
        ids=chunk_ids
    )

    return collection

if __name__ == "__main__":
    nomeDoArquivo = input()
    mensagem = pdf_embedding(nomeDoArquivo)
    print(mensagem)