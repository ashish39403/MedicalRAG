# All the Importation
from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings , HuggingFacePipeline 
import torch
from langchain_community.vectorstores import FAISS



# Defining the data path
DATA_PATH = "data/"

# Loading the files/pdf

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_file(data=DATA_PATH)


# Creating the Chunks from the above loaded  document

def create_chunks(extracted_docs):
    text_splitter =RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    text_chunks = text_splitter.split_documents(extracted_docs)
    return text_chunks
text_chunks_test = create_chunks(extracted_docs=documents)
print(len(text_chunks_test))


# Creating the embedding from the chunks
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding = get_embedding_model()



# Storing the above created emeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks_test ,embedding)
db.save_local(DB_FAISS_PATH)