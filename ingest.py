import os
import pickle
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from document_processor import DocumentProcessor

# Mock UploadedFile class for ingest.py
class MockUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content

def build_index():
    # Use DocumentProcessor to handle ingestion
    doc_processor = DocumentProcessor()
    
    uploaded_files_list = []
    folder_path = "uploaded_docs"
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        with open(path, "rb") as f:
            content = f.read()
        uploaded_files_list.append(MockUploadedFile(filename, content))

    if not uploaded_files_list:
        print("No documents found in uploaded_docs to process.")
        return None

    result = doc_processor.process_uploaded_files(uploaded_files_list)
    if result["success"]:
        print(f"Index built and saved. {result["total_chunks"]} chunks processed.")
    else:
        print(f"❌ Error building index: {result["message"]}")

if __name__ == "__main__":
    build_index()