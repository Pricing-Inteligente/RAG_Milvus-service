# retrieve_macro.py — Milvus + embeddings reales (E5) con fallback semántico
from __future__ import annotations
import os, re, unicodedata
from datetime import datetime
from typing import List, Dict, Optional, Any

# =========================
# ===== Configurables =====
# =========================
# Colección en Milvus
COLL_NAME  = os.getenv("MILVUS_MACRO_COLLECTION", "macro_latam")

# Nombres de campos (esquema creado por migrar_macro_a_milv.py)
F_PK     = "macro_id"
F_EMB    = "embedding"     # FloatVector (768)
F_COUNTRY= os.getenv("PG_MACRO_COUNTRY", "pais")
F_VAR    = os.getenv("PG_MACRO_VARIABLE", "variables")
F_UNIT   = os.getenv("PG_MACRO_UNIT", "unidad")
F_DATE   = os.getenv("PG_MACRO_DATE", "fecha")     # 'YYYY-MM' o 'YYYY-MM-DD'
F_YEAR   = os.getenv("PG_MACRO_YEAR", "anio")
F_MONTH  = os.getenv("PG_MACRO_MONTH", "mes")
F_VALUE  = os.getenv("PG_MACRO_VALUE", "valor")    # double
F_CANON  = "canonical_text"

# Buscar con HNSW/IP (igual que productos)
MACRO_METRIC = os.getenv("macro_metric_type", "IP")
MACRO_INDEX  = os.getenv("macro_index_type", "HNSW")
MACRO_EF     = int(os.getenv("macro_ef_search", "96"))
MACRO_NPROBE = int(os.getenv("macro_nprobe", "48"))
MACRO_EMBED_MODEL = os.getenv("MACRO_EMBED_MODEL", "intfloat/multilingual-e5-base")

# =========================
# ===== Sinónimos/alias ===
# =========================
def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()

# Claves canónicas que la app entiende
CANON_KEYS = ["inflation_rate", "cpi", "interest_rate", "exchange_rate", "gdp", "producer_prices"]


SYN: Dict[str, List[str]] = {
    "inflation_rate": [
        "Inflation rate", "inflation rate", "inflacion", "inflación",
        "tasa de inflacion", "tasa de inflación"
    ],
    "cpi": [
        "CPI", "cpi", "IPC", "ipc",
        "indice de precios al consumidor", "índice de precios al consumidor"
    ],
    "interest_rate": [
        "Interest rate", "interest rate", "tasa de interes", "tasa de interés",
        "tipo de interes", "tipo de interés"
    ],
    "exchange_rate": [
        "Exchange rate", "exchange rate", "tipo de cambio",
        "cambio dolar", "cambio del dolar", "cambio del dólar", "dolar", "dólar"
    ],
    "gdp": [
        "GDP", "gdp", "PIB", "pib", "producto interno bruto"
    ],
    "producer_prices": [
        "producer prices", "Producer prices",
        "indice de precios al productor", "índice de precios al productor",
        "IPP", "ipp", "PPI", "ppi"
    ],
}

# País: acepta ISO2 y nombre
COUNTRY_SYNS: Dict[str, List[str]] = {
    "AR": ["AR", "Argentina"],
    "CO": ["CO", "Colombia"],
    "MX": ["MX", "Mexico", "México"],
    "BR": ["BR", "Brasil", "Brazil"],
    "CL": ["CL", "Chile"],
    "PE": ["PE", "Peru", "Perú"],
    "EC": ["EC", "Ecuador"],
    "PA": ["PA", "Panama", "Panamá"],
    "CR": ["CR", "Costa Rica"],
    "PY": ["PY", "Paraguay"],
    "UY": ["UY", "Uruguay"],
    "BO": ["BO", "Bolivia"],
}

# =========================
# ===== Milvus helpers ====
# =========================
from pymilvus import Collection, connections

def _col() -> Collection:
    host = os.getenv("MILVUS_HOST", "127.0.0.1")
    port = os.getenv("MILVUS_PORT", "19530")
    try:
        connections.get_connection_addr("default")
    except Exception:
        connections.connect(alias="default", host=host, port=port)
    c = Collection(COLL_NAME)
    try:
        c.load()
    except Exception:
        pass
    return c

def _country_expr(country_iso2: str) -> str:
    vals = COUNTRY_SYNS.get((country_iso2 or "").upper(), [country_iso2])
    quoted = ",".join([f"'{v}'" for v in vals if v])
    return f"{F_COUNTRY} in [{quoted}]" if quoted else f"{F_COUNTRY} == '{country_iso2}'"

def _var_in_expr(canon: str) -> Optional[str]:
    aliases = SYN.get(canon, [])
    if not aliases:
        return None
    def esc(v: str) -> str:
        return v.replace("'", "\\'")
    in_list = ",".join([f"'{esc(a)}'" for a in aliases])
    return f"{F_VAR} in [{in_list}]"

def _date_key(row: Dict[str, Any]) -> int:
    # Para ordenar de más reciente a más antiguo (YYYYMMDD → int)
    s = str(row.get(F_DATE) or "")
    if s and len(s) >= 7:
        try:
            dd = datetime.strptime(s[:10], "%Y-%m-%d")
            return int(dd.strftime("%Y%m%d"))
        except Exception:
            try:
                dm = datetime.strptime(s[:7], "%Y-%m")
                return int(dm.strftime("%Y%m01"))
            except Exception:
                pass
    y = row.get(F_YEAR); m = row.get(F_MONTH)
    try:
        return int(f"{int(y):04d}{int(m):02d}01")
    except Exception:
        return 0

def _fmt_date(row: Dict[str, Any]) -> str:
    s = str(row.get(F_DATE) or "")
    if s:
        return s
    y = row.get(F_YEAR); m = row.get(F_MONTH)
    if y and m:
        try:
            return f"{int(y):04d}-{int(m):02d}"
        except Exception:
            return f"{y}-{m}"
    return ""

def _norm_out(country_iso2: str, r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "country": country_iso2,
        "name": r.get(F_VAR),
        "unit": r.get(F_UNIT),
        "value": r.get(F_VALUE),
        "date": _fmt_date(r),
    }

# =========================
# == Embedding (fallback) =
# =========================
_EMBED_MODEL = None  # cache global

def _embed_query(q: str):
    global _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        import torch, numpy as np
    except Exception:
        return None

    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(
            MACRO_EMBED_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    vec = _EMBED_MODEL.encode([f"query: {q}"], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32").tolist()[0]

# =========================
# ===== API pública =======
# =========================
def macro_list(country_iso2: str) -> List[Dict[str, Any]]:
    """
    Devuelve hasta 1000 registros del país dado, normalizados a:
      {country, name, unit, value, date}
    """
    col = _col()
    expr = _country_expr(country_iso2)
    rows = col.query(
        expr=expr,
        output_fields=[F_VAR, F_UNIT, F_VALUE, F_DATE, F_YEAR, F_MONTH],
        limit=1000
    )
    # ordenar por fecha desc
    rows.sort(key=_date_key, reverse=True)
    return [_norm_out(country_iso2, r) for r in rows]

def macro_lookup(macro_canon: str, country_iso2: str) -> Optional[Dict[str, Any]]:
    """
    Busca una variable (por alias exacto primero) y si no la encuentra
    hace fallback semántico (vectorial) filtrado por país.
    Retorna {country, name, unit, value, date} o None.
    """
    if macro_canon not in CANON_KEYS:
        return None

    col = _col()
    # 1) Intento exacto por lista de alias (IN)
    vexpr = _var_in_expr(macro_canon)
    if vexpr:
        expr = f"{_country_expr(country_iso2)} and {vexpr}"
        res = col.query(
            expr=expr,
            output_fields=[F_VAR, F_UNIT, F_VALUE, F_DATE, F_YEAR, F_MONTH],
            limit=1000
        )
        if res:
            res.sort(key=_date_key, reverse=True)
            return _norm_out(country_iso2, res[0])

    # 2) Fallback semántico (si hay modelo disponible)
    q = f"{country_iso2} {macro_canon.replace('_',' ')}"
    qvec = _embed_query(q)
    if qvec is not None:
        search_param = (
            {"metric_type": MACRO_METRIC, "params": {"ef": MACRO_EF}}
            if MACRO_INDEX.upper() == "HNSW"
            else {"metric_type": MACRO_METRIC, "params": {"nprobe": MACRO_NPROBE}}
        )
        try:
            resv = col.search(
                data=[qvec],
                anns_field=F_EMB,
                param=search_param,
                limit=5,
                expr=_country_expr(country_iso2),
                output_fields=[F_VAR, F_UNIT, F_VALUE, F_DATE, F_YEAR, F_MONTH],
            )
            if resv and resv[0]:
                r = resv[0][0].entity
                return _norm_out(country_iso2, r)
        except Exception:
            pass

    # 3) Último recurso: listar y matchear por contains normalizado
    norms = [_norm(a) for a in SYN.get(macro_canon, [])]
    rows = macro_list(country_iso2)
    for r in rows:
        n = _norm(r.get("name") or "")
        if any(p in n or n in p for p in norms):
            return r
    return None

def macro_compare(macro_canon: str, countries: List[str]) -> List[Dict[str, Any]]:
    """
    Compara la variable dada en múltiples países.
    """
    out: List[Dict[str, Any]] = []
    for c in countries:
        r = macro_lookup(macro_canon, c)
        if r:
            out.append(r)
    # salida ordenada por país
    out.sort(key=lambda x: x.get("country") or "")
    return out

# Coloca esto al inicio del archivo o cerca de utilidades:
DEFAULT_COUNTRIES = ['AR','CO','MX','PA','BR','CR','PY','PE','CL','UY']

def _macro_default_countries() -> list[str]:
    """
    Devuelve la lista de países para las comparaciones macro sin requerir settings.
    - Si existe settings.S.countries la usa (sin romper si no está).
    - Si existe la variable de entorno SPI_COUNTRIES="AR,CO,MX", la usa.
    - Si no, usa un set por defecto pensado para LATAM.
    """
    try:
        from settings import S
        cs = getattr(S, "countries", None)
        if cs:
            return [str(c).upper() for c in cs if isinstance(c, str) and c.strip()]
    except Exception:
        pass


    import os
    env = os.getenv("SPI_COUNTRIES")
    if env:
        cs = [c.strip().upper() for c in env.split(",") if c.strip()]
        if cs:
            return cs

    # 3) Fallback seguro
    return list(DEFAULT_COUNTRIES)


def _macro_rank(var: str, countries: list[str]) -> list[dict]:
    out = []
    for c in countries:
        r = macro_lookup(var, c)
        if r and (r.get("value") is not None):
            out.append({
                "country": c,
                "value": float(r["value"]),
                "unit": r.get("unit"),
                "date": r.get("date"),
                "name": r.get("name"),
            })
    return out

