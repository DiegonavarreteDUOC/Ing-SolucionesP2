"""
memory_config.py — Configuración de Memoria del Agente (EP2)
Implementa dos niveles de memoria:
  - Corto plazo : ConversationBufferWindowMemory  (últimos N turnos de la sesión)
  - Largo plazo : ChromaDB persistente             (base vectorial que persiste entre sesiones)
"""

import os
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory


# ─────────────────────────────────────────────────────────────
# MEMORIA DE CORTO PLAZO
# Recuerda los últimos k=5 intercambios dentro de la sesión.
# Cuando se acaba la sesión, esta memoria se pierde.
# ─────────────────────────────────────────────────────────────
def get_short_term_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """
    Crea una memoria de ventana deslizante que retiene los últimos k turnos.
    Proporciona continuidad dentro de una misma conversación.
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
    )


# ─────────────────────────────────────────────────────────────
# MEMORIA DE LARGO PLAZO
# Persiste en ChromaDB en disco.  Recupera los k=2 fragmentos
# más relevantes del historial pasado para enriquecer el contexto.
# ─────────────────────────────────────────────────────────────
def get_long_term_memory(persist_dir: str = "./chroma_db") -> VectorStoreRetrieverMemory:
    """
    Crea (o carga) un VectorStore persistente en disco para memoria semántica.
    Permite que el agente recuerde conversaciones de sesiones anteriores.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="long_term_memory"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="long_term_context",
        input_key="input",
    )
