from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import os

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

# To add more information to the database, run this script directly to rebuild the ChromaDB and then push the new ./chroma_db folder.
if __name__ == "__main__":
    # load lore documents
    loader = DirectoryLoader('./world_lore/', glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    docs = loader.load()

    # chunking documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"Deleted existing database at {db_path}")

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print("Lore successfully digitized!")