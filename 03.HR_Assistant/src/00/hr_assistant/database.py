import os
import chromadb
from chromadb.utils import embedding_functions

class Database:
    def __init__(self):
       
        # Funzione di embedding basata su OpenAI
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_KEY"),
            model_name=os.getenv("MODEL_NAME")  # modello per embeddings
        )

        # Cliente ChromaDB persistente
        self.client = chromadb.PersistentClient(
            path=os.getenv("PERSISTENT_DIR", "data/chromadb")
        )

        # Collezione dove vengono salvati documenti e embeddings
        self.collection = self.client.get_or_create_collection(
            name=os.getenv("COLLECTION_NAME", "CVs"),
            embedding_function=self.openai_ef
        )

    def add_documents(self, documents, metadatas, ids):
        """
        Aggiunge documenti alla collezione.
        - documents: lista di testi
        - metadatas: lista di metadata associati
        - ids: lista di identificativi unici
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text, n_results=1):
        """
        Effettua una query di similarità sul contenuto dei documenti.
        - query_text: testo di ricerca
        - n_results: numero di risultati da restituire
        """
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
    
    def get_tracked_files(self):

        result = self.collection.get()
        tracked_files = {}

        if result and result["metadatas"]:
            for metadata in result["metadatas"]:
                if metadata["source"] not in tracked_files:
                    tracked_files[metadata["source"]] = {
                        "hash": metadata["hash"],
                        "last_modified": metadata["last_modified"],
                        "source": metadata["source"]
                    }
                    
        return tracked_files
    
    def remove_document_by_source(self, source):
        result = self.collection.get(where={"source": source})
        if result and result["ids"]:
            self.collection.delete(ids=result["ids"])