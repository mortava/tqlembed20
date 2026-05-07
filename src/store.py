import time
from google import genai
from google.genai import types


class FileSearchStoreManager:
    """Manages Gemini File Search Stores."""

    DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"

    def __init__(self, client: genai.Client):
        self.client = client

    def create(self, display_name: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """Create a new File Search Store."""
        store = self.client.file_search_stores.create(
            config={
                "display_name": display_name,
                "embedding_model": embedding_model,
            }
        )
        print(f"[store] Created: {store.name}")
        return store

    def list(self):
        """List all File Search Stores."""
        stores = list(self.client.file_search_stores.list())
        print(f"[store] Found {len(stores)} stores")
        return stores

    def get(self, name: str):
        """Get a specific File Search Store by name."""
        return self.client.file_search_stores.get(name=name)

    def delete(self, name: str, force: bool = True):
        """Delete a File Search Store and all its documents."""
        self.client.file_search_stores.delete(name=name, config={"force": force})
        print(f"[store] Deleted: {name}")

    def list_documents(self, store_name: str):
        """List all documents in a File Search Store."""
        docs = list(self.client.file_search_stores.documents.list(parent=store_name))
        print(f"[store] {len(docs)} documents in {store_name}")
        return docs

    def get_document(self, document_name: str):
        """Get a specific document by name."""
        return self.client.file_search_stores.documents.get(name=document_name)

    def delete_document(self, document_name: str):
        """Delete a specific document from a File Search Store."""
        self.client.file_search_stores.documents.delete(name=document_name)
        print(f"[store] Deleted document: {document_name}")
