import chainlit as cl
import ollama 
import chromadb
import os, uuid
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Cartella dei CV
cv_dir = "../../resumes"

documents = []
metadatas = []
ids = []

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


load_dotenv("../../.env")

# Creo la funzione di embedding, utilizzando l'api di open ai 
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = os.getenv('OPENAI_API_KEY'),
    model_name = "text-embedding-3-small"
)

# Creo il client ChromaDB
chroma_client = chromadb.Client()

# Creo la collection del db
collection = chroma_client.get_or_create_collection(
    name = "CVs",
    embedding_function= openai_ef
)

# Setto i valori nella collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# Funzione che viene chiaamta all'inzio della chat per inizializzare la sessione dell'utente
@cl.on_message
async def handle_message():
    """
    Inizializza la sessione utente con un messaggio del sistema.
    Questo messaggio definisce il ruolo del chatbot
    """
    cl.user_session.set(
        "messages",
        [
            {
                "role": "developer",
                "content": """
                    Sei un assistente specializzato nel mondo HR, rispondi in modo professionale, sintetico e pragmatico.
                    Il tuo ruolo è individurare il candidato ideale rispetto alle richieste dell'utente
                    """
            }
        ]
    )


# Funzione per gestire i messaggi dell'utente e generare risposte
@cl.on_message
async def handle_message(message: cl.Message):
    """
    Gestisce i messaggi inviati dall'utente, li passa al modello e restituisce la risposta
    """
    # Domande dell'utnete nel chat bot
    user_question = message.content

    # Interroghiamo il db e in result troveramo la risposta della query
    results = collection.query(query_texts=[user_question], n_results=1)

    # Ora recuperiamo le prime 100 righe del CV che sicuramente comprenderanno le info principali
    # come nome cognome ecc..
    def leggi_prime_100_righe(file_path):
        with open(file_path, "r") as file: 
            righe = []
            for i, riga in enumerate(file):
                if i < 100: 
                    righe.append(riga.strip())
                else:
                    break
            return righe
    
    # Recupero la prima parte del file per leggere il nome del candidato
    filename = results["metadatas"][0][0]["source"]
    context_nome_candidato = leggi_prime_100_righe(
        os.path.join(cv_dir, filename)
    )

    nome = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": f"""
                    Dato il seguente contento individua il nome e cognome del candidato e ritorna solo il nome e cognome
                    del candidato. Quello che sto per fonirti è il curriculm vite del candidato: {context_nome_candidato}
                """
            }
        ]
    )

    nome = nome["message"]["content"]

    context = f"CONTESTO: nome file {filename} ecco il paragrafo più significativo {results['documents'][0][0]}"

    prompt = f"""
    Dato il seguente contesto:
    [[[{context}]]].
    Rispondi alla domanda dell'utente: [[[{user_question}]]].
    Spiega che nel file individuato c'è il profilo più adatto.
    Assicurati di nominare il Nome del file.
    Assicurati di indicare il nome del candidato [[[{nome}]]].
    Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto.
    Se non trovi corrispondenza in nessun cv non inventare.
    """

    # Recupera i messaggi salvati nella sessione uente
    messages = cl.user_session.get("messages", [])
    messages.append({ "role": "user", "content": prompt} )

    # Inizializza un messaggio vuoto per mostrare lo streaming
    response_message = cl.Message(content="")
    await response_message.send()

    try:
        # Invio della richiesta al modello tramite Ollama
        stream = ollama.chat(model="gemma3:4b", messages=messages, stream=True)

        # Streaming della risposta del modello verso l'utente
        for token in stream:
            await response_message.stream_token(token["message"]["content"])

        # Salva la risposta del modello nella sessione
        messages.append({"role": "assistant", "context": response_message.content})
        await response_message.update()
    except Exception as e:
        # Gestione degli errori
        error_message = f'An error occured: {str(e)}'
        cl.Message(content=error_message).send()
        print(error_message)

    
    # Aggiorna i messaggi della sessione utente
    cl.user_session.set("messages", messages)