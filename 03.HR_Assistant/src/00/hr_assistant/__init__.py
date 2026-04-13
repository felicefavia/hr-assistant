import os
from pathlib import Path
import chainlit as cl
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from database import Database
from utilis import LLMHelmper

load_dotenv("../../.env")

# Inizializza il db
db = Database()

# Processo i documenti
added, updated, removed = DocumentProcessor.process_documents(db)
print(f"Document sync complete: {added} added, {updated} updated, {removed} removed")

@cl.action_callback("db_stats")
async def on_action(action: cl.Action):
    db_info = db.get_stats()
    response = await LLMHelmper.get_db_stats(db_info)
    await cl.Message(response).send()

@cl.action_callback("db_reindex")
async def on_action(action: cl.Action):
    added, updated, removed = DocumentProcessor.process_documents(db)
    message = f"DB reindicizzato con successo. Document sync complete: {added} added, {updated} updated, {removed} removed."
    await cl.Message(message).send()


@cl.action_callback("helloword")
async def on_action(action: cl.Action):
    await cl.Message("Holaaa").send()

# Funzione che viene chiamata all'inizio della chat per inizializzare la sessione dell'utente
@cl.on_chat_start
async def start():
    """
    Inizializza la sessione utente con un messaggio del sistema.
    Questo messaggio definisce il ruolo del chatbot
    """

    actions = [
        cl.Action(
            name="db_stats",
            icon="mouse-pointer-click",
            payload={"value": "db_stats"},
            label="Statistiche Database"
        ),
        cl.Action(
            name="db_reindex",
            icon="mouse-pointer-click",
            payload={"value": "db_reindex"},
            label="Reindex Database"
        ),
        cl.Action(
            name="helloword",
            icon="mouse-pointer-click",
            payload={"value": "helloword"},
            label="Hello World!"
        )
    ]

    await cl.Message(content="Informazioni del sistema", actions=actions).send()


    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",  # cambiato da 'developer' a 'system' per OpenAI
                "content": (
                    "Sei un assistente specializzato nel mondo HR, "
                    "rispondi in modo professionale, sintetico e pragmatico. "
                    "Il tuo ruolo è individuare il candidato ideale rispetto alle richieste dell'utente."
                )
            }
        ]
    )

# Funzione per gestire i messaggi dell'utente e generare risposte
@cl.on_message
async def handle_message(message: cl.Message):
    """
    Gestisce i messaggi inviati dall'utente, li passa al modello e restituisce la risposta
    """
    user_question = message.content
    results = db.query(user_question)

    BASE_DIR = Path(__file__).resolve().parent
    DOCUMENTS_DIR = BASE_DIR / os.getenv("DOCUMENTS_DIR", "resumes")

    # Nome del file dai risultati
    filename = results["metadatas"][0][0]['source']
    file_path = DOCUMENTS_DIR / filename

    # Leggi le prime 10 linee del file in UTF-8
    context_lines = DocumentProcessor.read_first_linest(file_path, 10)

    context = (
        f"CONTESTO: nome file {filename} ecco il paragrafo più significativo "
        f"{results['documents'][0][0]}"
    )

    # Crea il prompt
    prompt = LLMHelmper.create_prompt(context, user_question, context_lines)

    # Recupera i messaggi salvati nella sessione utente
    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    # Inizializza un messaggio vuoto per mostrare lo streaming
    response_message = cl.Message(content="")
    await response_message.send()

    try:
        # Streaming della risposta dal modello OpenAI
        stream = LLMHelmper.chat(messages)

        for chunk in stream:
            # Verifica se il chunk ha contenuto
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                await response_message.stream_token(content)

        # Salva la risposta del modello nella sessione
        messages.append({"role": "assistant", "content": response_message.content})
        await response_message.update()

    except Exception as e:
        # Mostra correttamente l'errore usando await
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(content=error_message).send()
        print(error_message)

    # Aggiorna i messaggi della sessione (solo una volta)
    cl.user_session.set("messages", messages)