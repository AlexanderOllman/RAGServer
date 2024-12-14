import os
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from utils import download_artifact, pdf_into_documents
import shutil

class VectorDatabase:
    def __init__(self, uri, api_key, endpoint, model_name):
        """
        Initialize the vector database with NVIDIA embeddings.
        
        Args:
            uri (str): URI to download the FAISS vector database.
            api_key (str): API key for NVIDIA embeddings.
            endpoint (str): Endpoint for the NVIDIA embeddings service.
            model_name (str): Model name to use for embeddings.
        """
        self.local_dir = "local-database"
        self.uri = uri

        # Set up NVIDIA embeddings
        os.environ["NVIDIA_API_KEY"] = api_key
        self.embeddings = NVIDIAEmbeddings(model=model_name, base_url=endpoint, truncate="END")

        # Load or initialize FAISS database
        self._download_database()
        self.vector_store = FAISS.load_local(self.local_dir, self.embeddings, allow_dangerous_deserialization=True)

    def _download_database(self):
        """Download FAISS database from the URI."""
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)
        os.makedirs(self.local_dir, exist_ok=True)
        download_artifact(self.uri, self.local_dir)

    def retrieve(self, query, top_k):
        """Retrieve top_k results for a query."""
        return self.vector_store.similarity_search(query, k=top_k)

    def add_document(self, file_path):
        """Add a new document to the vector database."""
        new_documents = pdf_into_documents(file_path)
        self.vector_store.add_documents(new_documents)
        self.vector_store.save_local(self.local_dir)
