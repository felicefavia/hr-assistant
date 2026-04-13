# utilis.py

import os
from openai import OpenAI



class LLMHelmper:

    @staticmethod
    def chat(messages):
        """
        Invia i messaggi al modello LLM e ritorna uno stream di token.
        """

        client = OpenAI(
            base_url=os.getenv("AI_API_URL"), 
            api_key=os.getenv("AI_API_KEY")
        )

        return client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=messages,
            stream=True
        )

    @staticmethod
    async def get_candidate_name(context):
        """
        Estrae il nome e cognome del candidato dal contenuto fornito.
        """

        client = OpenAI(
            base_url=os.getenv("AI_API_URL"), 
            api_key=os.getenv("AI_API_KEY")
        )

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": f"""
                        Dato il seguente contenuto, individua il nome e cognome del candidato
                        e ritorna solo il nome e cognome.
                        Ecco il curriculum vitae del candidato: {context}
                    """
                }
            ]
        )
        return response.choices[0].message.content

    @staticmethod
    def create_prompt(context, user_question, context_lines):
        """
        Crea un prompt dettagliato per la richiesta dell'utente basata sul contesto e sui dati estratti dai documenti.
        """
        return f"""Dato il seguente contesto:
        [[[{context}]]].
        Rispondi alla domanda dell'utente: [[[{user_question}]]].
        Spiega che nel file individuato c'è il profilo più adatto.
        Assicurati di nominare il Nome del file.
        Assicurati di indicare il nome del candidato, la sua email e il suo numero di telefono alla fine della risposta: [[[ {context_lines}]]].
        Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto.
        Se non trovi corrispondenza in nessun CV, non inventare nulla.
        """
    
    @staticmethod
    async def get_db_stats(context):

        client = OpenAI(
            base_url=os.getenv("AI_API_URL"), 
            api_key=os.getenv("AI_API_KEY")
        )

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {
                    "role": "user",
                    "content": f"""
                       Il tuo compito è quello di descrivere in modo testuale, ma sintentico le statistiche
                       legate al databse dei framment indicizzati da questo sistema.
                       Ecco le informazioni necessarie {context}
                    """
                }
            ]
        )
        return response.choices[0].message.content