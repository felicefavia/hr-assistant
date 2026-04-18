import os 
import uuid
import hashlib
from semantic_chunking import SemanticChunking
from pathlib import Path

class DocumentProcessor: 

    @staticmethod
    def get_file_hash(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunck in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunck)

        return hash_md5.hexdigest()

    @staticmethod
    def get_document_metadata(file_path):
        return {
            "hash": DocumentProcessor.get_file_hash(file_path),
            "last_modified": os.path.getmtime(file_path),
            "source": os.path.basename(file_path)
        }

    @staticmethod
    def process_single_document(file_path):
        documents = []
        metadatas = []
        ids = []

        with open(file_path, "r") as file:
            txt = file.read()
            sc = SemanticChunking()
            chunks = sc.chunk_text(txt)
            file_metadata = DocumentProcessor.get_document_metadata(file_path)

            for chunck in chunks:
                if not chunck.isspace() and not chunck == "":
                    documents.append(chunck)
                    metadatas.append(file_metadata)
                    ids.append(str(uuid.uuid4()))

        return documents, metadatas, ids


    @staticmethod
    def process_documents(db):

        BASE_DIR = Path(__file__).resolve().parents[3]
        documents_dir = BASE_DIR / os.getenv("DOCUMENTS_DIR", "resumes")

        current_files = {
            f: DocumentProcessor.get_document_metadata(
                os.path.join(documents_dir, f)
            )
            for f in os.listdir(documents_dir)
            if f.endswith(".txt")
        }


        # PRENDE I FILE ESISTENTI
        existing_files = db.get_tracked_files()
        print("Existing files in db: ", existing_files)

        # FILE DA AGGIUNGERE
        files_to_add = set(current_files.keys()) - set(existing_files.keys())
        print("Files to add: ", files_to_add)

        files_to_remove = set(existing_files.keys()) - set(current_files.keys())
        print("Files to remove: ", files_to_remove)

        files_to_update = {
            f
            for f in set(current_files.keys()) & set(existing_files.keys())
            if current_files[f]["hash"] != existing_files[f]["hash"]
        }
        print("Files to update: ", files_to_update)

        for action, files in [("add", files_to_add), ("update", files_to_update)]:
            for filename in files:
                file_path = os.path.join(documents_dir, filename)
                documents, metadatas, ids = DocumentProcessor.process_single_document(
                    file_path
                )

                if action == "update":
                    db.remove_document_by_source(filename)

                if documents:
                    db.add_documents(documents, metadatas, ids)

        for filename in files_to_remove:
            db.remove_document_by_source(filename)

        return len(files_to_add), len(files_to_update), len(files_to_remove)


    
    @staticmethod
    def read_first_linest(file_path, n_lines):
        print("---------------------------- file_path ----------------------------: " + str(file_path))
        with open(file_path, "r") as file:
            return [line.strip() for line, _ in zip(file, range(n_lines))]