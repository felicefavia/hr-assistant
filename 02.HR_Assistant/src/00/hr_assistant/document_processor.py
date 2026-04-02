import os 
import uuid

class DocumentProcessor: 
    @staticmethod
    def process_documents():
        documents = []
        metadatas = []
        ids = []
        # Cartella dei CV
        cv_dir = "../../resumes"


        # Prende i documenti
        # splitta i documenti in chuncks
        # popola i 3 array (documents, metadatas, ids)
        for filename in os.listdir(cv_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(cv_dir, filename), "r") as file:
                    frasi = file.read().replace("\n", ".")
                    print("Sostituzione di \\n con . in frasi")
                    chuncks = frasi.split("###")
                    print("Split delle frasi in chuncks")
                    print(chuncks)

                    for chunck in chuncks:
                        if not chunck.isspace() and not chunck == "":
                            documents.append(chunck)
                            metadatas.append({"source": filename})
                            guid = str(uuid.uuid4())
                            ids.append(guid)

        return documents, metadatas, ids
    
    @staticmethod
    def read_first_linest(file_path, n_lines):
        with open(file_path, "r") as file:
            return [line.strip() for line, _ in zip(file, range(n_lines))]