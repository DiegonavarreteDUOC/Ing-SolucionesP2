"""
tools.py — Herramientas del Agente ChileAtiende (EP2)
Define las 3 herramientas que el AgentExecutor puede invocar:
  1. buscar_tramite     → consulta semántica sobre la base vectorial RAG
  2. redactar_resumen   → genera un resumen formal del trámite encontrado
  3. derivar_ejecutivo  → devuelve instrucciones de derivación a atención humana
"""

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ──────────────────────────────────────────────
# Inicialización del VectorStore (compartido)
# ──────────────────────────────────────────────
_vectorstore = None

def get_vectorstore():
    """Devuelve el vectorstore persistente; lo inicializa si no existe."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=OpenAIEmbeddings()
        )
    return _vectorstore


# ──────────────────────────────────────────────
# HERRAMIENTA 1: Búsqueda RAG en fichas de trámites
# ──────────────────────────────────────────────
@tool
def buscar_tramite(consulta: str) -> str:
    """
    Busca información oficial sobre bonos, beneficios y trámites del Estado de Chile
    en la base de conocimientos de ChileAtiende.
    Úsala cuando el ciudadano pregunte sobre requisitos, montos o cómo postular a un beneficio.
    Input: texto libre con la consulta del ciudadano.
    Output: fragmento(s) de texto con información oficial recuperada.
    """
    vectorstore = get_vectorstore()
    resultados = vectorstore.similarity_search(consulta, k=2)
    if not resultados:
        return "No se encontró información relevante en la base de datos interna."
    textos = [f"[Fuente: {doc.metadata.get('source', 'fichas_tramites.txt')}]\n{doc.page_content}" for doc in resultados]
    return "\n\n---\n\n".join(textos)


# ──────────────────────────────────────────────
# HERRAMIENTA 2: Redacción de resumen formal
# ──────────────────────────────────────────────
@tool
def redactar_resumen(informacion: str) -> str:
    """
    Toma información en bruto sobre un trámite o beneficio y la redacta
    en un resumen formal, claro y empático para el ciudadano.
    Úsala después de 'buscar_tramite' para formatear la respuesta final.
    Input: texto con información del trámite.
    Output: resumen formal en 3-5 oraciones.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    prompt = (
        "Eres un redactor oficial de ChileAtiende. "
        "Transforma la siguiente información técnica en un resumen claro, empático y formal "
        "para un ciudadano. Usa lenguaje inclusivo. Máximo 5 oraciones.\n\n"
        f"Información:\n{informacion}"
    )
    respuesta = llm.invoke(prompt)
    return respuesta.content


# ──────────────────────────────────────────────
# HERRAMIENTA 3: Derivación a ejecutivo humano
# ──────────────────────────────────────────────
@tool
def derivar_ejecutivo(motivo: str) -> str:
    """
    Genera un mensaje de derivación a un ejecutivo humano de ChileAtiende.
    Úsala cuando la consulta esté fuera del alcance del agente, cuando el ciudadano
    esté frustrado, o cuando requiera trámites personalizados (RUT, estado de solicitud, etc.).
    Input: motivo breve de la derivación.
    Output: mensaje formal de derivación con canales de contacto oficiales.
    """
    return (
        f"Entiendo que su consulta requiere atención personalizada ({motivo}). "
        "Le recomendamos contactar directamente con un ejecutivo de ChileAtiende a través de: \n"
        "📞 Call Center: 101 (lunes a viernes, 08:00 a 22:00 hrs)\n"
        "🌐 Web: https://www.chileatiende.gob.cl\n"
        "📍 Sucursal más cercana: https://www.chileatiende.gob.cl/sucursales\n"
        "Un ejecutivo podrá revisar su caso específico con total confidencialidad."
    )


TOOLS = [buscar_tramite, redactar_resumen, derivar_ejecutivo]
