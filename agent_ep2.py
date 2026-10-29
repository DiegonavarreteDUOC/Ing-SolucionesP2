"""
agent_ep2.py — Agente Funcional ChileAtiende (EP2 — ISY0101)
=============================================================
Agente con:
  • Herramientas : buscar_tramite | redactar_resumen | derivar_ejecutivo
  • Memoria CP   : ConversationBufferWindowMemory (últimos 5 turnos)
  • Memoria LP   : ChromaDB persistente (recuperación semántica entre sesiones)
  • Planificación: ReAct (Reasoning + Acting) vía OpenAI Functions Agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import TOOLS
from memory_config import get_short_term_memory

# ──────────────────────────────────────────────
# 1. INGESTA Y VECTORIZACIÓN (Base RAG)
# ──────────────────────────────────────────────
def build_vectorstore(data_path: str = "./data/fichas_tramites.txt",
                      persist_dir: str = "./chroma_db") -> Chroma:
    """
    Carga los documentos, los fragmenta y los almacena en ChromaDB.
    Si ya existe una base persistida, simplemente la carga.
    """
    embeddings = OpenAIEmbeddings()

    # Si ya existe la base, sólo cargar
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("📂 Cargando base vectorial existente desde disco...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("🔨 Creando base vectorial desde documentos...")
    loader = TextLoader(data_path, encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separator="\n\n"
    )
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ {len(splits)} fragmentos indexados correctamente.")
    return vectorstore


# ──────────────────────────────────────────────
# 2. PROMPT DEL SISTEMA (System Prompt)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un agente inteligente oficial de ChileAtiende, el sistema de atención ciudadana del Estado de Chile.

Tus responsabilidades son:
1. Responder consultas sobre bonos, beneficios y trámites del Estado usando la herramienta 'buscar_tramite'.
2. Formatear las respuestas de forma clara y empática usando la herramienta 'redactar_resumen'.
3. Derivar al ciudadano a un ejecutivo humano con 'derivar_ejecutivo' cuando:
   - La consulta involucra datos personales (RUT, estado de expediente, etc.)
   - La información no se encuentra en la base de conocimientos.
   - El ciudadano muestra frustración o urgencia elevada.

Reglas estrictas:
- NUNCA inventes requisitos o montos que no estén en los documentos recuperados.
- Usa siempre lenguaje formal, empático e inclusivo.
- Si el ciudadano saluda o hace preguntas generales, responde de forma cordial sin usar herramientas.
- Recuerda el contexto previo de la conversación para dar continuidad.

Historial de la conversación:
{chat_history}
"""


# ──────────────────────────────────────────────
# 3. CONSTRUCCIÓN DEL AGENTE
# ──────────────────────────────────────────────
def build_agent():
    """
    Construye el AgentExecutor con:
    - LLM: GPT-3.5-turbo (temperatura 0 para determinismo)
    - Herramientas: buscar_tramite, redactar_resumen, derivar_ejecutivo
    - Memoria CP: ConversationBufferWindowMemory (k=5)
    - Planificación: OpenAI Functions Agent (ReAct)
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm=llm, tools=TOOLS, prompt=prompt)

    memory = get_short_term_memory(k=5)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,          # Muestra razonamiento paso a paso (útil para la evaluación)
        handle_parsing_errors=True,
        max_iterations=5,
    )
    return agent_executor


# ──────────────────────────────────────────────
# 4. PLANIFICACIÓN — ESQUEMA DE DECISIÓN
# ──────────────────────────────────────────────
# El agente sigue el patrón ReAct (Reason → Act → Observe → Repeat):
#
# Turno del ciudadano
#      │
#      ▼
# ¿Es saludo / pregunta general? ─── Sí ──▶ Responder directo (sin herramienta)
#      │ No
#      ▼
# Invocar 'buscar_tramite(consulta)'
#      │
#      ▼
# ¿Se encontraron resultados? ──── No ───▶ Invocar 'derivar_ejecutivo(motivo)'
#      │ Sí
#      ▼
# Invocar 'redactar_resumen(info_encontrada)'
#      │
#      ▼
# Entregar respuesta al ciudadano + guardar en memoria CP


# ──────────────────────────────────────────────
# 5. MAIN — Interfaz conversacional por consola
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Agente Inteligente ChileAtiende — EP2 ISY0101")
    print("="*60)

    build_vectorstore()
    agent = build_agent()

    print("\n¡Agente listo! Escribe 'salir' para terminar.\n")

    while True:
        user_input = input("Ciudadano: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit"):
            print("Agente: ¡Hasta pronto! Que tenga un excelente día.")
            break

        try:
            result = agent.invoke({"input": user_input})
            print(f"\nAsistente ChileAtiende: {result['output']}\n")
        except Exception as e:
            print(f"\n[Error del agente]: {e}\n")
