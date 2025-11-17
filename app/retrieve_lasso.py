# retrieve_lasso.py
from typing import List, Dict
try:
    from typing import Literal
except Exception:
    from typing_extensions import Literal

import os
from pymilvus import connections, Collection


LASSO_COLLECTION = os.getenv("LASSO_COLLECTION", "lasso_models")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "default")

def _connect_if_needed():
    """
    Abre la conexión a Milvus si aún no existe.
    Respeta las mismas variables de entorno que usa el resto del proyecto.
    """
    try:
        connections.get_connection_addr(MILVUS_ALIAS)
    except Exception:
        connections.connect(MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)

# Campos que vamos a recuperar de lasso_models
LASSO_OUTPUT_FIELDS = [
    "nombre", "retail", "marca", "producto",
    "r_squared", "best_alpha", "n_obs",
    "coef_cambio_dolar_pct_change",
    "coef_cpi_pct_change",
    "coef_gdp_pct_change",
    "coef_gini_pct_change",
    "coef_inflation_rate_pct_change",
    "coef_interest_rate_pct_change",
    "coef_producer_prices_pct_change",
]

def _safe_term(term: str) -> str:
    """
    Normaliza el término de búsqueda:
    - recorta espacios
    - quita comillas sueltas
    """
    if not term:
        return ""
    t = term.replace('"', "").replace("'", "").strip()
    # Evita dobles espacios
    return " ".join(t.split())

def query_lasso_models(by: Literal["brand", "product"], term: str, topk: int = 5) -> List[Dict]:
    """
    Busca en la colección lasso_models por 'marca' (brand) o 'producto' (product).
    1) Intenta un LIKE directo sobre el campo (contiene).
    2) Si no encuentra resultados, hace un fallback buscando en 'canonical_text'
       (que suele estar en minúsculas) con el término en minúsculas.
    Devuelve hasta 'topk' filas y las ordena por R² descendente.
    """
    _connect_if_needed()
    col = Collection(LASSO_COLLECTION)
    try:
        col.load()
    except Exception:
        pass

    field = "marca" if by == "brand" else "producto"
    term = _safe_term(term)

    rows: List[Dict] = []

    # 1) Búsqueda principal: LIKE sobre marca/producto
    if term:
        expr = f'{field} like "%{term}%"'
        try:
            rows = col.query(expr=expr, output_fields=LASSO_OUTPUT_FIELDS, limit=topk)
        except Exception:
            rows = []

    # 2) Fallback: buscar en canonical_text (lowercase) si no encontramos nada
    if not rows and term:
        try:
            expr2 = f'canonical_text like "%{term.lower()}%"'
            rows = col.query(expr=expr2, output_fields=LASSO_OUTPUT_FIELDS, limit=topk)
        except Exception:
            rows = []

    # Ordenamos por R² descendente y truncamos al topk solicitado
    rows = rows or []
    rows.sort(key=lambda r: (r.get("r_squared") or 0.0), reverse=True)
    return rows[:topk]
