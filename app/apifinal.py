# app/api.py — Conversacional con memoria de sesión y comparaciones inteligentes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal
import requests, re, json, unicodedata, os
from datetime import datetime
from collections import OrderedDict
# cerca de otros helpers de Milvus
from retrieve_macro import macro_list, macro_lookup, macro_compare, _macro_default_countries, _macro_rank

import time



# Config
from settings import get_settings
S = get_settings()

# Milvus helpers
from retrieve import retrieve, list_by_filter, aggregate_prices

# Inteligencia de intención / categoría
from intent_llm import parse_intent
from category_resolver import resolve_category_semantic



# Utilidades de normalización y alias

def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")

COUNTRY_ALIASES = {
    "MX": ["mx", "mexico", "méxico"],
    "BR": ["br", "brasil", "brazil"],
    "AR": ["ar", "argentina"],
    "CO": ["co", "colombia"],
    "CL": ["cl", "chile"],
    "PE": ["pe", "peru", "perú"],
    "EC": ["ec", "ecuador"],
    "CR": ["cr", "costa rica", "costa-rica", "costa_rica", "costarica"],
    "PA": ["pa", "panama", "panamá"],
    "PY": ["py", "paraguay"],
}
NCOUNTRIES = { _norm(alias): code
               for code, aliases in COUNTRY_ALIASES.items()
               for alias in aliases }


# Macros
MACRO_ALIASES = {
    "exchange_rate": [
        "exchange rate", "tipo de cambio",
        "cambio dolar", "cambio del dolar", "cambio del dólar",
        "dolar", "dólar"
    ],
    "cpi": [
        "cpi", "ipc",
        "indice de precios al consumidor", "índice de precios al consumidor",
        "costo de vida", "coste de vida", "costo de la vida", "cost of living"
    ],
    "gdp": [
        "gdp", "pib", "producto interno bruto", "producto interno bruto (pib)"
    ],
    "gini_index": [
        "gini index", "indice de gini", "índice de gini",
        "coeficiente de gini", "gini coefficient"
    ],
    "inflation_rate": [
        "inflation rate", "tasa de inflacion", "tasa de inflación",
        "inflacion", "inflación"
    ],
    "interest_rate": [
        "interest rate", "interest trate",
        "tasa de interes", "tasa de interés",
        "tipo de interes", "tipo de interés"
    ],
    "producer_prices": [
        "producer prices",
        "indice de precios al productor", "índice de precios al productor",
        "ipp", "ppi"
    ],
}
NMACROS = { _norm(alias): canon
            for canon, aliases in MACRO_ALIASES.items()
            for alias in aliases }

def _extract_macros(text: str) -> list[str]:
    nt = _norm(text or "")
    found: list[str] = []
    for alias, canon in NMACROS.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            if canon not in found:
                found.append(canon)
    # soporta también el caso "macro"/"variables macroeconómicas"
    if ("variables macroeconomicas" in nt or "variables macroeconómicas" in nt
        or re.search(r"(?<!\w)macro(?!\w)", nt)):
        if "__ALL__" not in found:
            found.append("__ALL__")
    return found



# mapea frases comunes a tus nombres de variable en BD
_MACRO_SYNONYMS = {
    "costo de vida": "cpi",
    "ipc": "cpi",
    "índice de precios al consumidor": "cpi",
    "indice de precios al consumidor": "cpi",
    "inflación": "inflation_rate",
    "inflacion": "inflation_rate",
    "tasa de interés": "interest_rate",
    "tasa de interes": "interest_rate",
    "cambio del dolar": "usd_exchange",
    "tipo de cambio": "usd_exchange",
}

def _canonicalize_macro_var(name: str) -> str:
    n = _norm(name)
    for k, v in _MACRO_SYNONYMS.items():
        if k in n:
            return v
    return name





# MACRO: detectar consultas tipo "¿qué país tiene X más alto/mas bajo?"
_SUPER_MAX_PAT = re.compile(r"\b(más\s+alto|mas\s+alto|mayor|top|máximo|maximo|más\s+caro|mas\s+caro)\b", re.I)
_SUPER_MIN_PAT = re.compile(r"\b(más\s+bajo|mas\s+bajo|menor|mínimo|minimo|más\s+barato|mas\s+barato)\b", re.I)

def _is_macro_superlative_query(text: str) -> str | None:
    t = _norm(text)
    if _SUPER_MAX_PAT.search(t): return "max"
    if _SUPER_MIN_PAT.search(t): return "min"
    return None






import ast

def _coerce_country_list(val):
    """Acepta 'CO' | ['CO','BR'] | "['CO','BR']" y devuelve una lista ['CO', 'BR']."""
    if val is None:
        return None
    if isinstance(val, list):
        return [str(x).strip().upper() for x in val if x]
    if isinstance(val, str):
        s = val.strip()
        # string que parece lista
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                return [str(x).strip().upper() for x in arr if x]
            except Exception:
                pass
        # valor único
        return [s.strip().upper()]
    return [str(val).strip().upper()]

def _canonicalize_category(cat_or_text: str | None) -> str | None:
    """Devuelve la categoría canónica según tu resolver semántico."""
    if not cat_or_text:
        return None
    try:
        from category_resolver import resolve_category_semantic
        cat, score, _ = resolve_category_semantic(
            str(cat_or_text),
            min_cosine=float(getattr(S, "category_min_cosine", 0.60))
        )
        return cat
    except Exception:
        return None

def _normalize_plan_filters(filters: dict | None, text_for_fallback: str) -> dict:
    f = dict(filters or {})
    # country → lista
    if "country" in f:
        clist = _coerce_country_list(f.get("country"))
        if clist:
            f["country"] = clist
        else:
            f.pop("country", None)
    # category → canónica
    cat = f.get("category") or _canonicalize_category(text_for_fallback)
    if cat:
        f["category"] = cat
    else:
        f.pop("category", None)
    return f



def _guess_macro(text: str) -> str | None:
    nt = _norm(text or "")
    for alias, canon in NMACROS.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            return canon
    # “variables macroeconomicas de X”, “macro de X”
    if ("variables macroeconomicas" in nt or "variables macroeconómicas" in nt
        or re.search(r"(?<!\w)macro(?!\w)", nt)):
        return "__ALL__"  # pedir todas para un país
    return None

def _fmt_macro_row(r: dict, idx: int | None = None) -> str:
    prefix = f"{idx}. " if idx is not None else ""
    unit = f" {r.get('unit')}" if r.get('unit') else ""
    date = r.get("date") or r.get("fecha") or "-"
    prev = r.get("previous")
    prev_txt = f" · Anterior: {prev}" if prev not in (None, "") else ""
    return f"{prefix}{r['name']}: {r['value']}{unit} · Fecha: {date} · País: {r['country']}{prev_txt}"



# ALIAS: categoría canónica (según Milvus)
# las claves del dict son las CATEGORÍAS que existen en Milvus,
# y los valores son las listas de ALIAS que pueden escribir los usuarios.
CAT_ALIASES = {
    # arroz
    "arroz": ["arroz", "rice"],

    # pan de molde
    "pan_de_molde": [
        "pan de molde", "pan tajado", "pan lactal", "pan de caja",
        "pan sandwich", "pan sándwich", "pan bimbo"
    ],

    # leche líquida
    "leche_liquida": [
        "leche", "leches", "leche liquida", "leche líquida",
        "leche uht", "leche entera", "leche descremada", "leche semidescremada"
    ],

    # leche
    "leche": ["leche en polvo", "leche polvo"],

    # pasta seca
    "pasta_seca": ["pasta", "espagueti", "espaguetis", "spaghetti", "fideos", "macarrones"],

    # azúcar
    "azucar": ["azucar", "azúcar", "sugar"],

    # café (genérico) y molido
    "cafe": ["cafe", "café", "cafe instantaneo", "café instantáneo"],
    "cafe_molido": ["cafe molido", "café molido", "cafe tostado y molido", "café tostado y molido"],

    # aceite vegetal
    "aceite_vegetal": [
        "aceite", "aceite vegetal", "aceite de cocina",
        "aceite de girasol", "aceite de soya", "aceite de soja",
        "aceite de maiz", "aceite de maíz"
    ],

    # huevo
    "huevo": ["huevo", "huevos", "docena de huevos"],

    # pollo entero
    "pollo_entero": ["pollo", "pollo entero", "pollo fresco", "pollo asadero"],

    # refrescos de cola
    "refrescos_de_cola": [
        "refresco de cola", "refrescos de cola", "gaseosa", "gaseosas",
        "gaseosa de cola", "soda", "cola", "coca cola", "coca-cola"
    ],

    # papa
    "papa": ["papa", "papas", "patata", "patatas"],

    # frijol
    "frijol": ["frijol", "frijoles", "poroto", "porotos", "alubia", "alubias"],

    # harina de trigo
    "harina_de_trigo": ["harina", "harina de trigo"],

    # cerveza
    "cerveza": ["cerveza", "beer"],

    # queso blando
    "queso_blando": [
        "queso", "quesos", "queso fresco", "queso doble crema",
        "queso mozarella", "queso mozzarella"
    ],

    # atún (genérico) y en lata
    "atun": ["atun", "atún"],
    "atun_en_lata": ["atun en lata", "atún en lata", "lata de atun", "lata de atún"],

    # tomate
    "tomate": ["tomate", "jitomate", "tomates"],

    # cebolla
    "cebolla": ["cebolla", "cebollas", "onion"],

    # manzana
    "manzana": ["manzana", "manzanas", "apple"],

    # banano
    "banano": ["banano", "bananos", "banana", "bananas"],

    # pan
    "pan": ["pan", "pan frances", "pan francés", "bollos"],
}

# Construimos el mapa alias
NCATEGORIES = {
    _norm(alias): canon
    for canon, aliases in CAT_ALIASES.items()
    for alias in aliases
}

# Categorías GENÉRICAS que NO existen tal cual en Milvus.
# el refinamiento semántico en build_filters_smart
GENERIC_CATS = {"lacteos", "lácteos", "alimentos", "bebidas", "aseo", "hogar"}


STORE_ALIASES = {
    "Exito": ["exito", "éxito"],
    "Jumbo": ["jumbo", "jumboco", "jumboar", "jumbope"],
    "Olimpica": ["olimpica", "olímpica"],
    "Carulla": ["carulla"],
    "Ara": ["ara"],
    "D1": ["d1"],
    "Walmart": ["walmart"],
    "Soriana": ["soriana"],
    "Chedraui": ["chedraui"],
    "Lider": ["lider", "líder"],
    "Wong": ["wong"],
    "Metro": ["metro", "metrope"],
    "Tottus": ["tottus"],
    "Carrefour": ["carrefour", "carrefourbr", "carrefourar"],
    "Assai": ["assai", "assaí"],
    "PaoDeAcucar": ["pao de acucar", "pao de açúcar", "pão de açúcar", "paodeacucar"],
    "Atacadao": ["atacadao", "atacadão"],
    "Extra": ["extra"],
}
NSTORES = { _norm(alias): canon
            for canon, aliases in STORE_ALIASES.items()
            for alias in aliases }

def sanitize_filters(f: Dict | None) -> Dict:
    if not f:
        return {}
    out: Dict = {}

    # country
    v = f.get("country")
    if v:
        nv = _norm(str(v))
        out["country"] = NCOUNTRIES.get(nv, str(v).upper())

    # category
    v = f.get("category")
    if v:
        nv = _norm(str(v))
        canon = NCATEGORIES.get(nv)
        if canon:
            out["category"] = canon

    # store
    v = f.get("store")
    if v:
        nv = _norm(str(v))
        canon = NSTORES.get(nv)
        out["store"] = canon if canon else str(v)

    for k in ["brand", "name"]:
        if k in f and f[k] not in (None, ""):
            out[k] = f[k]
    return out

def _sanitize_resp_excerpt(text: str, max_len: int = 600) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    # Escapar comillas dobles internas
    s = s.replace('"', '\\"')
    return s

# -----------------------------------------------------------------------------
# Fusión inteligente de filtros (Heurística + LLM + Semántico)

def build_filters_smart(message: str, base: Optional[Dict] = None) -> Dict:
    """
    Prioriza:
    1) base (usuario/frontend)
    2) heurística por aliases en el texto
    3) LLM (parse_intent)
    4) resolutor semántico por embeddings (si aún no hay categoría)
    """
    filters: Dict = sanitize_filters(base or {})

    nt = _norm(message)
    for alias, code in NCOUNTRIES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            filters.setdefault("country", code); break
    for alias, canon in NCATEGORIES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            filters.setdefault("category", canon); break
    for alias, canon in NSTORES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            filters.setdefault("store", canon); break

    # LLM structured intent
    llm = parse_intent(message)
    if llm.get("country"):  filters.setdefault("country", llm["country"])
    if llm.get("store"):    filters.setdefault("store", llm["store"])
    if llm.get("brand"):    filters.setdefault("brand", llm["brand"])
    if llm.get("category"):
        c_llm = _norm(llm["category"])

        if c_llm not in { _norm(x) for x in GENERIC_CATS }:
            canon = NCATEGORIES.get(c_llm)
            if canon:
                filters.setdefault("category", canon)

    # si hay indicios macro, NO intentamos resolver categoría semántica
    macro_hint = bool(_extract_macros(message)) or bool(_guess_macro(message))
    if macro_hint:
        return sanitize_filters(filters)

    # Semantic fallback (solo category)
    if "category" not in filters or not filters.get("category"):
        cat_sem, score_sem, _ = resolve_category_semantic(
            message, min_cosine=float(getattr(S, "category_min_cosine", 0.60))
        )
        if cat_sem:
            filters["category"] = cat_sem

    return sanitize_filters(filters)

# Memoria de sesión (simple LRU en memoria)

class SessionMemory:
    def __init__(self, max_sessions: int = 500):
        self.max_sessions = max_sessions
        self.store: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, sid: Optional[str]) -> dict:
        if not sid:
            return {}
        val = self.store.get(sid) or {}
        if sid in self.store:
            self.store.move_to_end(sid, last=True)
        return val

    def set(self, sid: Optional[str], data: dict):
        if not sid:
            return
        if sid in self.store:
            self.store[sid].update(data)
            self.store.move_to_end(sid, last=True)
        else:
            self.store[sid] = dict(data)
        while len(self.store) > self.max_sessions:
            self.store.popitem(last=False)

MEM = SessionMemory()

def merge_with_memory(
    filters: Dict,
    sid: Optional[str],
    prefer_last_cat: bool = False,
    mentioned: Optional[Dict[str, bool]] = None
) -> Dict:
    """
    Rellena con la última sesión y aplica políticas:
    - Si prefer_last_cat=True y NO se mencionó categoría explícita, fuerza la última categoría.
    - country persiste si no se menciona uno nuevo.
    - store solo persiste si (category y country) permanecen iguales a los de la sesión
      y el turno NO mencionó explícitamente store.
    """
    if not sid:
        return filters

    mentioned = mentioned or {}
    m_cat = bool(mentioned.get("category"))
    m_cty = bool(mentioned.get("country"))
    m_sto = bool(mentioned.get("store"))

    last = MEM.get(sid) or {}
    lastf = last.get("last_filters", {}) or {}

    merged = dict(filters)

    # country
    if not merged.get("country") and lastf.get("country"):
        merged["country"] = lastf["country"]

    # category
    if not merged.get("category") and lastf.get("category"):
        merged["category"] = lastf["category"]
    if prefer_last_cat and not m_cat and lastf.get("category"):
        # Fuerza la categoría anterior cuando no se mencionó explícitamente una nueva
        merged["category"] = lastf["category"]

    #  store (política de persistencia condicionada)
    if m_sto:
        # Si mencionaron tienda, respetamos lo que ya venga en merged
        pass
    else:
        # Si NO mencionaron tienda:
        # Copiamos la anterior SOLO si categoría y país quedaron iguales
        # Si categoría/país cambiaron, descartamos tienda previa
        same_cat = merged.get("category") == lastf.get("category")
        same_cty = merged.get("country") == lastf.get("country")
        if not merged.get("store"):
            if same_cat and same_cty and lastf.get("store"):
                merged["store"] = lastf["store"]
        else:
            # merged ya trae store, pero si el usuario cambió
            # cat/país sin mencionar tienda, eliminamos la tienda para no arrastrarla
            if not (same_cat and same_cty):
                merged.pop("store", None)

    return merged



def _filters_head(f: Dict) -> str:
    return ("Filtros → "
            f"país: {f.get('country') or '-'} | "
            f"categoría: {f.get('category') or '-'} | "
            f"tienda: {f.get('store') or '-'}")


# Visualización — construcción de prompt para la API externa de gráficos

def _viz_title(filters: Dict, intent: str, group_by: str | None = None) -> str:
    cat = (filters or {}).get("category") or "productos"
    cty = (filters or {}).get("country")
    sto = (filters or {}).get("store")
    loc = f" en {cty}" if cty else ""
    if intent == "aggregate":
        gb = {"store": "tienda", "country": "país", "category": "categoría"}.get(group_by or "", "grupo")
        return f"Precio promedio de {cat} por {gb}{loc}"
    if intent == "compare":
        return f"Comparativa de precios de {cat}{loc}"
    if sto:
        return f"Precio de {cat}{loc} ({sto})"
    return f"Precio de {cat}{loc}"

def _viz_prompt_from_rows(
    rows: List[Dict],
    filters: Dict,
    *,
    title: str | None = None,
    max_n: int = 8,
    label_priority: List[str] = ["brand", "name"],
    user_prompt: str | None = None,
    rag_response: str | None = None,
) -> str:
    """
    Construye prompt de visualización basado en la RESPUESTA RAG (no en el prompt del usuario).
    Formato:
    creame la mejor forma de visualizar este reporte analítico, usando el tipo de grafica que consideres apropiado: "RESPUESTA RAG PETICION".
    """
    if not rows:
        return ""
    base = rag_response or user_prompt or title or _viz_title(filters, "lookup") or ""
    excerpt = _sanitize_resp_excerpt(base)
    return f'creame la mejor forma de visualizar este reporte analítico, usando el tipo de grafica que consideres apropiado: "{excerpt}".'

def _viz_prompt_from_agg(agg: Dict, filters: Dict, *, group_by: str, user_prompt: str | None = None,rag_response: str | None = None,) -> str:
    """
    Para agregados: usa el promedio como valor principal y pasa min/max para tooltips.
    Schema: [{label, value, min, max}]
    """
    # groups = (agg or {}).get("groups") or []
    # if not groups:
    #     return ""
    # labels = [str(g.get("group")) for g in groups if g.get("group") is not None]
    # if not labels:
    #     return ""
    # title = _viz_title(filters, "aggregate", group_by=group_by)
    # up = (user_prompt or title or "")
    # upq = up.replace("'", "\\'")
    # return f"creame la mejor visualizacion para responder la peticion '{upq}'."
    if not (agg or {}).get("groups"):
        return ""
    base = rag_response or user_prompt or _viz_title(filters, "aggregate", group_by=group_by) or ""
    excerpt = _sanitize_resp_excerpt(base)
    return f'creame la mejor forma de visualizar este reporte analítico, usando el tipo de grafica que consideres apropiado: "{excerpt}".'

def _viz_prompt_from_generic(
    intent: str,
    filters: Dict,
    *,
    user_prompt: str | None = None,
    rag_response: str | None = None,
) -> str:
    base = rag_response or user_prompt or _viz_title(filters, intent) or ""
    excerpt = _sanitize_resp_excerpt(base)
    return f'creame la mejor forma de visualizar este reporte analítico, usando el tipo de grafica que consideres apropiado: "{excerpt}".'

# def _maybe_viz_prompt(
#     intent: str,
#     filters: Dict,
#     *,
#     rows: List[Dict] | None = None,
#     agg: Dict | None = None,
#     group_by: str | None = None,
#     series: List[Dict] | None = None,
#     user_prompt: str | None = None,
# ) -> str | None:
#     try:
#         if intent in ("lookup", "list", "compare") and rows:
#             return _viz_prompt_from_rows(rows, filters, user_prompt=user_prompt)
#         if intent == "aggregate" and agg and (agg.get("groups") or []):
#             gb = group_by or "category"
#             return _viz_prompt_from_agg(agg, filters, group_by=gb, user_prompt=user_prompt)

#                 # TOPN: construir barras con los top N (usa "rows")
#         if intent == "topn" and rows:
#             up = (user_prompt or _viz_title(filters, "lookup") or "").strip()
#             upq = up.replace("'", "\\'")
#             return f"creame la mejor visualizacion para responder la peticion '{upq}'."

#         # TREND: línea temporal con serie (usa "series")
#         if intent == "trend" and series:
#             up = (user_prompt or "tendencia de precios")
#             upq = up.replace("'", "\\'")
#             return f"creame la mejor visualizacion para responder la peticion '{upq}'."


#     except Exception:
#         return None
#     return None

def _maybe_viz_prompt(
    intent: str,
    filters: Dict,
    *,
    rows: List[Dict] | None = None,
    agg: Dict | None = None,
    group_by: str | None = None,
    series: List[Dict] | None = None,
    user_prompt: str | None = None,
    rag_response: str | None = None,
) -> str | None:
    try:
        if intent in ("lookup", "list", "compare") and rows:
            return _viz_prompt_from_rows(rows, filters, user_prompt=user_prompt, rag_response=rag_response)
        if intent == "aggregate" and agg and (agg.get("groups") or []):
            gb = group_by or "category"
            return _viz_prompt_from_agg(agg, filters, group_by=gb, user_prompt=user_prompt, rag_response=rag_response)
        if intent == "topn" and rows:
            return _viz_prompt_from_generic("topn", filters, user_prompt=user_prompt, rag_response=rag_response)
        if intent == "trend" and series:
            return _viz_prompt_from_generic("trend", filters, user_prompt=user_prompt, rag_response=rag_response)
    except Exception:
        return None
    return None



def pick_effective_query(user_text: str, sid: Optional[str], prefer_last_cat: bool) -> str:
    """
    Si el turno no menciona categoría (prefer_last_cat=True), usamos la última
    consulta de la sesión para mantener el 'tema' (p.ej. 'leche') y evitar
    que embeddings 'adivinen' otra categoría.
    """
    if not sid:
        return user_text
    last = MEM.get(sid)
    last_q = (last or {}).get("last_query")
    if prefer_last_cat and last_q:
        return last_q
    return user_text

def remember_session(session_id, *, filters, intent, query, hits):
    last = MEM.get(session_id) or {}
    lastf = dict(last.get("last_filters") or {})

    # Solo pisa categoría si el turno la mencionó explícitamente o trae valor real
    mentioned = last.get("last_mentioned") or {}
    if "category" in filters and filters.get("category"):
        lastf["category"] = filters["category"]
    elif mentioned and mentioned.get("category"):
        lastf["category"] = filters.get("category")


    if "country" in filters and filters.get("country"):
        lastf["country"] = filters["country"]
    if "store" in filters:
        lastf["store"] = filters.get("store")

    MEM.set(session_id, {
        **last,
        "last_filters": lastf,
        "last_intent": intent,
        "last_query": query,
        "last_mentioned": mentioned
    })


# Logging / Trazabilidad

def _log_event(kind: str, payload: dict):
    try:
        os.makedirs("logs", exist_ok=True)
        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "kind": kind,
            **payload
        }
        with open("logs/trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass



def _now_ms():
    return int(time.perf_counter() * 1000)

def _log_perf(event: str, payload: dict):
    """
    Log compacto para consola y además a trace.jsonl.
    """
    try:
        print(f"[PERF] {json.dumps({'kind': event, **payload}, ensure_ascii=False)}")
    except Exception:
        pass
    _log_event(event, payload)





# CORS / App

app = FastAPI(title="RAG Pricing API", version="1.8.0")


@app.on_event("startup")
async def _warmup():
    try:
        # 1) Cargar colecciones Milvus
        from pymilvus import Collection
        from settings import get_settings
        S = get_settings()
        try:
            Collection(getattr(S, "milvus_collection", "products_latam")).load()
        except Exception:
            pass
        # Macro (si está en el mismo proceso)
        try:
            from retrieve_macro import _milvus_collection as _macro_coll
            _macro_coll().load()
        except Exception:
            pass

        # 2) Calentar LLMs y resolutores
        try:
            _ = llm_strict.generate("ok")
        except Exception:
            pass
        try:
            # Calentar embeddings del resolutor semántico de categoría
            from category_resolver import resolve_category_semantic
            _ = resolve_category_semantic("leche")
        except Exception:
            pass
    except Exception:
        # no romper inicio por warmup
        pass




@app.options("/chat/stream")
def options_chat_stream():
    # 204 vacío: el CORSMiddleware añadirá los headers CORS permitidos
    return Response(status_code=204)



origins = [
    "http://localhost:5173",  # frontend local
    "http://127.0.0.1:5173",  # a veces Vite usa 127.0.0.1
    "http://localhost:8080",  # Lovable local
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cliente LLM (Ollama)
class OllamaLLM:
    def __init__(self, model: str, base_url: str, temperature: float = 0.1,
                 num_ctx: int = 2048, num_predict: int = 256, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except Exception:
            return ""

    def stream(self, prompt: str, min_chars: int = 40):
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            },
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()
        buf = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "response" in chunk:
                    buf += chunk["response"]
                    if len(buf) >= min_chars or buf.endswith((" ", ".", ",", ":", ";", "\n")):
                        yield f"data: {buf}\n\n"; buf = ""
                if chunk.get("done"):
                    if buf: yield f"data: {buf}\n\n"
                    yield "data: [FIN]\n\n"
                    break
            except Exception:
                continue


def _stream_no_fin(prompt: str, *, model=None):
    """
    Envuelve el stream del modelo y filtra un '[FIN]' que pudiera emitir.
    Nunca se llama a sí misma.
    """
    m = model or llm_chat
    for chunk in m.stream(prompt):
        # Asegura que trabajamos con str
        if isinstance(chunk, (bytes, bytearray)):
            s = chunk.decode("utf-8", errors="ignore")
        else:
            s = str(chunk)

        # Filtrar un posible FIN que venga del modelo
        if s.strip() == "data: [FIN]":
            continue

        # Ya viene con 'data: ...\n\n'
        yield s




llm_strict = OllamaLLM(
    model=getattr(S, "gen_model", "phi3:mini"),
    base_url=getattr(S, "ollama_host", "http://127.0.0.1:11434"),
    temperature=0.0, num_ctx=1024, num_predict=128,
)
llm_chat = OllamaLLM(
    model=getattr(S, "gen_model", "phi3:mini"),
    base_url=getattr(S, "ollama_host", "http://127.0.0.1:11434"),
    temperature=getattr(S, "gen_temperature", 0.35),
    num_ctx=getattr(S, "gen_num_ctx", 2048),
    num_predict=getattr(S, "gen_num_predict", 700),
)

def _llm_json(prompt: str) -> str:
    return llm_strict.generate(prompt)


# Root/health/runtime

@app.get("/", tags=["root"])
def root():
    return {"ok": True, "name": "retail-rag-api", "mode": "read-only", "docs": "/docs"}

@app.get("/health", tags=["health"])
def health():
    return {"ok": True}

@app.get("/runtime", tags=["health"])
def runtime():
    return {
        "gen_model": S.gen_model,
        "ollama_host": S.ollama_host,
        "embed_backend": S.embed_backend,
        "embed_model": S.embed_model,
    }


# Prompts RAG


def _prompt_lookup_from_facts(question: str, facts: dict, ctx: str) -> str:
    """
    Redacta TODO (saludo, frase contextual y explicación) usando SOLO los HECHOS.
    Persona: asistente del Sistema de Pricing Inteligente , un sistema analítico que
    extrae, limpia y entrega datos del retail en América Latina. No eres una tienda.
    Estilo: profesional, cordial, conversacional; evita plantillas fijas.
    Reglas:
      - No inventes cifras ni marcas. No recomiendes visitar tiendas o webs externas.
      - Si mencionas (n), es el número de registros usados, no "tiendas".
      - Incluye SIEMPRE:
          1) el promedio nacional (moneda y n),
          2) un breve listado de promedios por marca (máx 8–10 ítems, con precio y n).
      - Si faltan datos, dilo brevemente y ofrece filtrar o comparar dentro de la base.
      - CIERRA SIEMPRE con un ÚLTIMO PÁRRAFO de “resumen + CTA”:
        usa FACTS.brand_range si existe (≈min→≈max con marcas) y relaciónalo
        con el promedio nacional; invita a filtrar por tienda, presentación o presupuesto.
    """
    import json
    facts_json = json.dumps(facts, ensure_ascii=False)
    return (
        "Eres el asistente del *Sistema Pricing Inteligente*, una plataforma de analítica\n"
        "que provee datos de retail para América Latina. Tu rol es analítico, no comercial.\n"
        "Redacta en español, tono natural y claro, variando el saludo y la redacción.\n"
        "Usa exclusivamente el bloque FACTS y, si te ayuda, algo del CONTEXTO; no inventes.\n"
        "Incluye el promedio nacional y luego un listado compacto por marca; finalmente cierra con un\n"
        "resumen breve y una invitación para seguir con filtros o comparaciones.\n"
        f"Pregunta del usuario: {question}\n"
        f"FACTS(JSON): {facts_json}\n"
        f"CONTEXTO:\n{ctx}\n"
        "Ahora redacta la respuesta completa."
    )


def _prompt_macro_humano(intent: str, facts: dict, hint_cta: str) -> str:
    import json
    return (
        "Eres el asistente del Sistema de Pricing Inteligente (SPI). "
        "Escribe en español, tono profesional y natural. "
        "Responde en 3–6 frases de TEXTO PLANO: sin markdown, sin títulos, sin viñetas, "
        "sin negritas (**), sin encabezados (#) y sin bloques de código. "
        "NO muestres el JSON ni nombres de campos; úsalo solo como fuente. "
        "Estructura: 1) saludo breve; 2) contexto (variable/país/es y fecha si aparece); "
        "3) respuesta con las cifras del JSON; 4) cierre con un único call to action.\n"
        f"FACTS(JSON): {json.dumps(facts, ensure_ascii=False)}\n"
        f"Evita términos o marcadores como 'end_of_one_example'. "
        f"CTA sugerido: {hint_cta}\n"
        "Responde solo el texto final, sin JSON ni formato markdown."
    )








# Small-talk
_GREET_PAT = re.compile(r"\b(hola|buen[oa]s?\s+d[ií]as|buenas?\s+tardes|buenas?\s+noches|hi|hello|hey)\b", re.I)
_THANKS_PAT = re.compile(r"\b(gracias|thank(s)?|mil\s+gracias)\b", re.I)
_BYE_PAT = re.compile(r"\b(chao|ad[ií]os|hasta\s+luego|bye)\b", re.I)
_HELP_PAT = re.compile(r"\b(ayud(a|e|o)|como\s+funcion|puedo\s+(preguntar|usar)|help)\b", re.I)

ASSISTANT_NAME = getattr(S, "assistant_name", "Asistente del Sistema Pricing Inteligente")

# Palabras/rasgos que indican consulta de precios → NO smalltalk
_DOMAIN_PAT = re.compile(
    r"\b(precio|precios|cu[aá]nt(o|a)|vale|costo|coste|compar(a|ar)|promedio|media|mínimo|max(imo)|historial|tendenc|hoy|ahora)\b",
    re.I
)
_CURRENCY_PAT = re.compile(r"(\$|€|£|¥|₱|₲|₡|R\$|S/\.|COP|ARS|CLP|PEN|MXN|BRL|USD|EUR)", re.I)

def _is_smalltalk(text: str) -> Optional[str]:
    t = (text or "").strip()
    tl = t.lower()

    # Si parece consulta de precios o ya detectamos filtros → no es smalltalk
    try:
        has_filters = bool(_guess_filters(t))
    except Exception:
        has_filters = False
    if _DOMAIN_PAT.search(tl) or _CURRENCY_PAT.search(t) or has_filters:
        return None

    if _GREET_PAT.search(tl):  return "greeting"
    if _THANKS_PAT.search(tl): return "thanks"
    if _BYE_PAT.search(tl):    return "goodbye"
    if _HELP_PAT.search(tl):   return "help"
    return None


# --- Conversación fluida: detectar pedidos de "ajuste de filtros" (refine) ---
# Detecta: cambia / cámbialo / cambialo / cámbiame / ponlo / ajustalo / configura / setea / quita / sin ...
# cerca de _is_smalltalk/_guess_filters
_REFINE_PAT = re.compile(
    r"\b("
    r"cambi(?:a|á)(?:me|nos|lo|la)?|cambiar|"
    r"pon(?:me|nos|lo|la)?|"
    r"ajust(?:a|e)(?:me|nos|lo|la)?|ajúst(?:a|e)(?:me|nos|lo|la)?|"
    r"usa|fija|configura|set(?:ea|ear)?|actualiza|define|"
    r"quita|quitar|saca|sacar|"
    r"háblame\s+de|hablame\s+de|ahora\s+en|y\s+en"
    r")\b",
    re.I
)

def _is_just_filters_command(text: str) -> bool:
    t = _norm(text or "")
    # Si pide datos explícitos, no es refine “silencioso”
    if re.search(r"(precio|precios|promedio|media|tendencia|historia|serie|lista|listar|compara|comparar|top|gráfic|grafica)", t):
        return False
    heur = _guess_filters(text)
    return any(heur.values())

def _is_refine_command(text: str) -> bool:
    return bool(_REFINE_PAT.search(_norm(text or "")) or _is_just_filters_command(text))




# --- TOPN: detectar pedidos tipo "top 3 más baratos" o "los más caros" ---
_TOP_PAT = re.compile(r"\btop\s*(\d+)\b", re.I)

def _extract_topn(text: str) -> tuple[int, str]:
    t = _norm(text or "")
    m = _TOP_PAT.search(t)
    n = int(m.group(1)) if m else 3
    # cheap if says barato/menor/más bajo; else expensive
    cheap = bool(re.search(r"barat|menor|minim|bajo", t))
    mode = "cheap" if cheap else "expensive"
    return max(min(n, 20), 1), mode

def _is_topn_query(text: str) -> bool:
    t = _norm(text or "")
    return bool(_TOP_PAT.search(t) or re.search(r"\b(barat|car[oa]s|más\s+car[oa]s|más\s+barat)", t))

# --- TREND: detectar “tendencia / últimos X días / evolución / serie” ---
def _is_trend_query(text: str) -> bool:
    t = _norm(text or "")
    return any(k in t for k in ["tendencia", "últimos", "ultimo", "último", "evolución", "histor", "serie"])


# --- STUB opcional: reemplázalo por tu implementación real ---
def series_prices(filters: dict | None, days: int = 30) -> list[dict]:
    """
    Devuelve [{ "date": "YYYY-MM-DD", "value": float, "currency": "COP" }].
    Sustituye este stub con tu consulta real (p.ej., tabla diaria).
    """
    try:
        # Intento: si tienes 'list_by_filter_history' úsala; si no, retorna []
        if 'list_by_filter_history' in globals():
            rows = list_by_filter_history(filters or {}, days=days) or []
            out = []
            for r in rows:
                if r.get("price") is None or not r.get("date"):
                    continue
                out.append({
                    "date": str(r.get("date"))[:10],
                    "value": float(r["price"]),
                    "currency": r.get("currency") or None
                })
            # colapsar por fecha (promedio)
            by_day = {}
            for x in out:
                d = x["date"]; by_day.setdefault(d, []).append(x["value"])
            series = []
            for d, vs in sorted(by_day.items()):
                series.append({"date": d, "value": sum(vs)/max(len(vs),1), "currency": out[-1].get("currency") if out else None})
            return series[-days:]
        return []
    except Exception:
        return []




def _prompt_smalltalk(user_msg: str, intent: str) -> str:
    base_persona = (
        f"Eres {ASSISTANT_NAME}. "
        "Habla en tono cercano y breve (1–3 frases). "
        "NO ofrezcas contactar proveedores, ni usar bases de datos externas, ni hacer cosas fuera del sistema. "
        "Nunca inventes datos de productos ni precios."
    )
    seeds = {
        "greeting": "Saluda de forma amable y ofrece ayuda para consultas de precios basadas en la base interna.",
        "thanks":   "Agradece y ofrece seguir ayudando con consultas de precios basadas en la base.",
        "goodbye":  "Despídete cordialmente.",
        "help":     ("Explica en una frase que puedes responder preguntas de precios usando la base interna "
                     "por categoría/país/tienda (p. ej., 'precio de azúcar en Colombia')."),
    }
    seed = seeds.get(intent, "Responde de forma amigable y ofrece ayuda dentro del sistema.")
    return f"{base_persona}\nUsuario: {user_msg}\nInstrucción: {seed}\nRespuesta:"


# -----------------------------------------------------------------------------
# Modelos de request
# -----------------------------------------------------------------------------


class Plan(BaseModel):
    intent: Literal["lookup","list","aggregate","compare","count"]
    filters: Optional[Dict] = Field(default_factory=dict)
    product_name: Optional[str] = None
    product_name_b: Optional[str] = None
    group_by: Optional[Literal["store","category","country"]] = None
    operation: Optional[Literal["min","max","avg"]] = None
    top_k: Optional[int] = 5
    limit: Optional[int] = 100


class ChatReqStream(BaseModel):
    message: str
    limit: Optional[int] = 100
    session_id: Optional[str] = None

# -----------------------------------------------------------------------------
# Helpers planner
# -----------------------------------------------------------------------------
def _guess_filters(text: str) -> Dict:
    nt = _norm(text)
    f: Dict = {}
    for alias, code in NCOUNTRIES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            f["country"] = code; break
    for alias, canon in NCATEGORIES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            f["category"] = canon; break
    for alias, canon in NSTORES.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            f["store"] = canon; break
    return f


# --- EXTRA: detectar N países mencionados (hasta 10) en orden de mención ---
def _extract_countries(text: str, max_n: int = 10) -> list[str]:
    """
    Devuelve códigos ISO2 según NCOUNTRIES (alias normalizados → código),
    en el orden en que aparecen en el texto (sin duplicados).
    """
    nt = _norm(text or "")
    # NCOUNTRIES ya existe más arriba: { "colombia": "CO", "costa rica": "CR", ... }
    keys = sorted(NCOUNTRIES.keys(), key=lambda s: -len(s))  # evita que "br" pise "brasil"
    pat = r"(?<!\w)(" + "|".join(map(re.escape, keys)) + r")(?!\w)"
    out: list[str] = []
    for m in re.finditer(pat, nt):
        code = NCOUNTRIES[m.group(1)]
        if code not in out:
            out.append(code)
        if len(out) >= max_n:
            break
    return out


def _fmt_row(r: dict, idx: int | None = None) -> str:
    pres = f"{(r.get('size') or '')}{(r.get('unit') or '')}".strip() or "-"
    prefix = f"{idx}. " if idx is not None else ""
    return (
        f"{prefix}{r.get('name')} · Marca: {r.get('brand')} · "
        f"Pres: {pres} · Precio: {r.get('price')} {r.get('currency')} · "
        f"Tienda: {r.get('store')} · País: {r.get('country')} "
        f"[{r.get('product_id')}]"
    )




def _classify_intent_heuristic(text: str) -> str:
    nt = _norm(text)
    if ("precio" in nt or "precios" in nt) and ("producto" in nt or "productos" in nt):return "list"
    if any(tok in nt for tok in ["comparar","compara","comparacion","vs","contra","frente a"]): return "compare"
    if any(tok in nt for tok in ["promedio","media","minimo","máximo","maximo","promedio por","por tienda","por pais","por país","por categoría","por categoria"]): return "aggregate"
    if any(tok in nt for tok in ["lista","listar","muestrame","mostrar","ver todos","todos los productos"]): return "list"
    if any(tok in nt for tok in ["cuantos","cuántos","cuantas","cuántas","numero de","número de","cantidad","total de"]): return "count"
    return "lookup"

def _plan_from_llm(message: str) -> Optional[Plan]:
    allowed_intents   = ["lookup","list","aggregate","compare","count"]
    allowed_group_by  = ["store","category","country", None]
    allowed_operation = ["min","max","avg", None]

    schema = {
        "type": "object",
        "properties": {
            "intent": {"enum": allowed_intents},
            "filters": {"type": "object"},
            "product_name": {"type": ["string","null"]},
            "product_name_b": {"type": ["string","null"]},
            "group_by": {"enum": allowed_group_by},
            "operation": {"enum": allowed_operation},
            "top_k": {"type": "integer"},
            "limit": {"type": "integer"}
        },
        "required": ["intent","filters"]
    }
    examples = [
    ("muéstrame la leche en peru", {"intent":"list","filters":{"country":"PE","category":"leche_liquida"}}),
    ("aceite vegetal 900ml en argentina", {"intent":"lookup","filters":{"country":"AR","category":"aceite_vegetal"}}),
    ("¿cuántos productos hay en chile?", {"intent":"count","filters":{"country":"CL"}}),
    ("promedio de precios por país para arroz", {"intent":"aggregate","group_by":"country","filters":{"category":"arroz"}}),
    ]

    prompt = (
        "Devuelve SOLO un JSON que cumpla exactamente este esquema, sin texto extra.\n"
        f"Esquema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        "- Normaliza country a ISO2 entre: MX, BR, AR, CO, CL, PE, EC, CR, PA, PY.\n"
        "- category usa canónicos EXACTOS de Milvus: arroz, pan_de_molde, leche_liquida, leche, pasta_seca, azucar, cafe, cafe_molido, aceite_vegetal, huevo, pollo_entero, refrescos_de_cola, papa, frijol, harina_de_trigo, cerveza, queso_blando, atun, atun_en_lata, tomate, cebolla, manzana, banano, pan.\n"
        "- Si el usuario pide \"leche\", normaliza a category=\"leche_liquida\".\n"
        "- store devuelve nombre canónico si lo reconoces; si no, null.\n"
        "- Si no estás seguro, deja null o filters vacío.\n\n"
        "Ejemplos:\n" +
        "\n".join([f"Usuario: {u}\nPlan: {json.dumps(p, ensure_ascii=False)}" for u,p in examples]) +
        f"\n\nUsuario: {message}\nPlan:"
    )
    txt = _llm_json(prompt).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if "intent" not in data or data["intent"] not in allowed_intents:
            return None
        data.setdefault("filters", {})
        data["filters"] = sanitize_filters(data["filters"])
        return Plan(**data)
    except Exception:
        return None



# -----------------------------------------------------------------------------
# /chat (planner + memoria + comparaciones inteligentes)
# -----------------------------------------------------------------------------
def _build_ctx(hits: List[Dict], k: int) -> str:
    return "\n".join(
        f"[{h['product_id']}] {h['name']} | Marca: {h['brand']} | "
        f"Pres: {h['size']}{h['unit']} | Precio: {h['price']} {h['currency']} | "
        f"Tienda: {h['store']} | País: {h['country']}"
        for h in hits[:k]
    )



# -----------------------------------------------------------------------------
# /chat/stream (memoria + encabezado de filtros)
# -----------------------------------------------------------------------------
@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatReqStream):
    text = (req.message or "").strip()
    t_req0 = _now_ms()



        # (nuevo) Diagnóstico de "sin resultados"
    def _diagnose_no_results(intent: str, *, plan=None, text:str="", macros=None, rows=None, hits=None, agg=None) -> str:
        f = (plan.filters if plan else {}) or {}
        countries = f.get("country")
        category  = f.get("category")
        store     = f.get("store")

        # --- MACRO ---
        if intent in ("macro_list","macro_compare","macro_lookup"):
            if intent == "macro_list" and not countries:
                return "faltó indicar el país para las variables macroeconómicas. ¿Indicas uno? (p. ej., CO, MX o BR)"
            if intent in ("macro_compare","macro_lookup") and not countries:
                return "faltó el país. ¿Lo agrego? (p. ej., CO, MX o BR)"
            if rows == [] or hits == [] or (agg and not agg.get("groups")):
                return "no hay datos publicados para esa variable/país en la base actual. ¿Probamos con otra variable o país?"
            return "no encontré series para esa variable/país. ¿Quieres ver el listado completo de variables disponibles?"

        # --- PRODUCTOS ---
        # 1) categoría ausente o irreconocible
        if intent in ("lookup","list","compare","aggregate") and not category:
            cat_guess = _canonicalize_category(text)
            if not cat_guess:
                return "no identifiqué una categoría válida (p. ej., 'azúcar', 'leche líquida'). ¿Te gustaría que use una de ejemplo?"

        # 2) tienda demasiado restrictiva
        if store and ((rows == []) or (hits == []) or (agg and not agg.get("groups"))):
            return "el filtro de tienda es muy restrictivo para esa categoría/país. ¿Quito la tienda o pruebo con otra?"

        # 3) “compare” sin suficientes ítems
        if intent == "compare" and hits is not None and len(hits) < 2:
            return f"necesito al menos 2 productos comparables, pero obtuve {len(hits)}. ¿Te listo más productos o cambiamos de país/tienda?"

        # 4) “aggregate” sin grupos
        if intent == "aggregate" and (not agg or not agg.get("groups")):
            return "no hubo datos suficientes para agrupar. ¿Prefieres un promedio simple o cambiar el group_by?"

        # 5) genérico
        return "no hubo coincidencias con esa combinación de filtros. ¿Relajo filtros (sin tienda) o cambiamos de categoría/país?"

    # (nuevo) SSE de “sin datos” con razón + contexto + CTA + FIN
    def _sse_no_data_ex(reason: str, filters: dict | None):
        head = _filters_head(filters or {})
        yield f"data: Hola. No pude traer resultados porque {reason}\n\n"
        if head:
            yield f"data: {head}\n\n"
        yield "data: [FIN]\n\n"




    # Small-talk corto
    small = _is_smalltalk(text)
    if small:
        prompt = _prompt_smalltalk(text, small)
        _log_event("chat_stream_smalltalk", {
            "sid": req.session_id, "message": text, "intent": small
        })
        return StreamingResponse(llm_chat.stream(prompt), media_type="text/event-stream")


    # --- REFINAR / SET FILTROS (antes de MACRO y planner) ---

    if _is_refine_command(text):
        f_now = _guess_filters(text)
        if not any(f_now.values()):
            reason = "no identifiqué qué filtro cambiar (país/categoría/tienda). ¿Cuál ajusto?"
            return StreamingResponse(_sse_no_data_ex(reason, None), media_type="text/event-stream")

        # fusiona con lo último que tengamos
        last = MEM.get(req.session_id) or {}
        lastf = (last.get("last_filters") or {})
        new_filters = sanitize_filters({**lastf, **f_now})

        # guarda la memoria
        if req.session_id:
            MEM.set(req.session_id, {"last_filters": new_filters})

        # responde breve y CIERRA con [FIN]
        def gen_refine():
            yield f"data: ¡Listo! Actualicé los filtros a → {new_filters}\n\n"
            yield "data: ¿Consulto el promedio, comparo países o te muestro los más baratos?\n\n"
            yield "data: [FIN]\n\n"
        return StreamingResponse(gen_refine(), media_type="text/event-stream")


    # --- Señaladores rápidos de intent (topN, trend) antes del planner ---
    nt = _norm(text)
    force_topn = _is_topn_query(nt)
    force_trend = _is_trend_query(nt)



        # ------ RUTA MACRO (stream) ------
    macros = _extract_macros(text)
    if macros:
        countries = _extract_countries(text) or []
        if not countries and req.session_id:
            last = MEM.get(req.session_id) or {}
            if last.get("last_country"):
                countries = [last["last_country"]]


                # ——— SUPERLATIVOS (p.ej. "más alto/más bajo") ———
        superl = _is_macro_superlative_query(text)  # devuelve "max", "min" o None
        if superl and macros and "__ALL__" not in macros:
            var = macros[0]
            cs = _macro_default_countries()  # usa tu lista de países por defecto
            ranked = _macro_rank(var, cs)    # [{country, value, unit, date, name}, ...]

            if not ranked:
                reason = _diagnose_no_results("macro_compare", plan=None, text=text, macros=macros, rows=[])
                return StreamingResponse(_sse_no_data_ex(reason, {"country": cs}), media_type="text/event-stream")

            # ordena según max/min
            ranked.sort(key=lambda x: x["value"], reverse=(superl == "max"))
            best = ranked[0]

            def gen_super():
                yield f"data: Filtros → variable: {var} | países: {', '.join(cs)}\n\n"

                # Redacción humana
                facts = {
                    "type": "macro_rank",
                    "variable": var,
                    "order": "desc" if superl == "max" else "asc",
                    "rows": ranked[:10]
                }
                hint = f"El {'más alto' if superl=='max' else 'más bajo'} es {best['country']} ({best['value']} {best.get('unit') or ''} {best.get('date') or ''}). ¿Quieres comparar los 3 primeros?"
                prompt = _prompt_macro_humano("macro_rank", facts, hint)
                for chunk in _stream_no_fin(prompt):
                    yield chunk
                yield "data: \n\n"

                # Lista compacta
                topn = 5
                titulo = "Top 5 más altos" if superl == "max" else "Top 5 más bajos"
                yield f"data: {titulo}:\n\n"
                for i, r in enumerate(ranked[:topn], start=1):
                    yield f"data: {i}. {r['country']}: {r['value']} {r.get('unit') or ''} ({r.get('date') or ''})\n\n"

                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_super(), media_type="text/event-stream")






                    # --- Superlativos macro: “¿qué país tiene X más alto/bajo?”
        superl = _is_macro_superlative_query(text)
        if superl and macros and "__ALL__" not in macros:
            var = macros[0]  # ya viene canónico por MACRO_ALIASES

            cs = getattr(S, "countries", None)
            if not cs:
                cs = sorted({v for v in COUNTRY_ALIASES.values() if isinstance(v, str) and len(v) == 2})

            rows = macro_compare(var, cs) or []
            if not rows:
                reason = _diagnose_no_results("macro_compare", plan=None, text=text, macros=macros, rows=rows)
                return StreamingResponse(_sse_no_data_ex(reason, {"country": cs}), media_type="text/event-stream")

            key = lambda r: float(r.get("value") or 0.0)
            best = (max(rows, key=key) if superl == "max" else min(rows, key=key))

            def gen_super():
                yield f"data: Filtros → variable: {var} | países: {' | '.join(cs)}\n\n"
                facts = {
                    "type": "macro_compare",
                    "variable": var,
                    "countries": cs,
                    "rows": [
                        {"country": x.get("country"), "name": x.get("name"),
                        "value": x.get("value"), "unit": x.get("unit"), "date": x.get("date")}
                        for x in rows
                    ],
                    "winner": {
                        "mode": "máximo" if superl == "max" else "mínimo",
                        "country": best.get("country"),
                        "value": best.get("value"),
                        "unit": best.get("unit"),
                        "date": best.get("date"),
                        "name": best.get("name"),
                    },
                }
                prompt = _prompt_macro_humano(
                    "macro_compare",
                    facts,
                    "¿Quieres que agregue otro país o ver la serie histórica?"
                )
                for chunk in _stream_no_fin(prompt):
                    yield chunk
                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_super(), media_type="text/event-stream")






        # ¿hay intención de productos en este mismo turno?
        heur_now = _guess_filters(text)  # país/categoría/tienda detectados por alias
        has_products = bool(heur_now.get("category")) or bool(re.search(r"\bprecio|precios\b", _norm(text)))

        # Si hay macros + productos => MIXED
        if has_products and countries:
            def gen_mix():
                # --- Sección MACRO ---
                for m in macros:
                    if m == "__ALL__":
                        rows = macro_list(countries[0]) or []
                        if rows:
                            yield f"data: [MACRO] País: {countries[0]} | variables: TODAS (mostrando 10)\n\n"

                            facts = {
                                "type": "macro_list",
                                "country": countries[0],
                                "items": [
                                    {"name": x.get("name"), "value": x.get("value"),
                                    "unit": x.get("unit"), "date": x.get("date")}
                                    for x in (rows[:10] if rows else [])
                                ]
                            }
                            prompt = _prompt_macro_humano("macro_list", facts, "¿Quieres que me enfoque en inflación, tasa o dólar?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"


                    elif len(countries) >= 2:
                        rows = macro_compare(m, countries) or []
                        if rows:
                            yield f"data: [MACRO] {m} | países: {' | '.join(countries)}\n\n"
                            # --- MACRO WRITER (COMPARE) ---
                            facts = {
                                "type": "macro_compare",
                                "variable": m,
                                "countries": countries,
                                "rows": [
                                    {"country": x.get("country"), "value": x.get("value"),
                                    "unit": x.get("unit"), "date": x.get("date")}
                                    for x in (rows or [])
                                ]
                            }
                            prompt = _prompt_macro_humano("macro_compare", facts, "¿Agrego otro país o convierto a misma base si aplica?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"

                    else:
                        r = macro_lookup(m, countries[0]) if len(countries)==1 else None
                        if r:
                            yield f"data: [MACRO] País: {countries[0]} | variable: {m}\n\n"
                            # --- MACRO WRITER (LOOKUP) ---
                            facts = {
                                "type": "macro_lookup",
                                "variable": m,
                                "country": countries[0],
                                "value": (r or {}).get("value"),
                                "unit": (r or {}).get("unit"),
                                "date": (r or {}).get("date"),
                                "name": (r or {}).get("name"),
                            }
                            prompt = _prompt_macro_humano("macro_lookup", facts, "¿La comparamos con otro país o te muestro la serie?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"


                # --- Sección PRODUCTOS ---
                pf = {k: v for k, v in (heur_now or {}).items() if k in ("country","category","store") and v}
                # si no trajo categoría exacta, igual intenta listado general por país
                rows_prod = list_by_filter(pf, limit= min(getattr(S, 'chat_list_default', 500), 40))
                if not rows_prod and pf.get("category"):
                    pf2 = dict(pf); pf2.pop("category", None)
                    rows_prod = list_by_filter(pf2, limit= min(getattr(S, 'chat_list_default', 500), 40))

                if rows_prod:
                    yield f"data: [PRODUCTOS] Filtros → {pf}\n\n"
                    for i, r in enumerate(rows_prod[:10], start=1):
                        yield f"data: {_fmt_row(r, i)}\n\n"
                yield "data: [FIN]\n\n"
            return StreamingResponse(gen_mix(), media_type="text/event-stream")

        # ---- Si NO hay productos en el mismo turno, conserva el comportamiento original ----
        try:
            if "__ALL__" in macros:
                if not countries:
                    reason = _diagnose_no_results("macro_list", plan=None, text=text, macros=macros)
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")

                rows = macro_list(countries[0]) or []
                if not rows:
                    reason = _diagnose_no_results("macro_list", plan=None, text=text, macros=macros, rows=rows)
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries}), media_type="text/event-stream")

                def gen_all():
                    yield f"data: Filtros → país: {countries[0]} | variable: TODAS\n\n"
                    # --- MACRO WRITER (LISTA) ---
                    facts = {
                        "type": "macro_list",
                        "country": countries[0],
                        "items": [
                            {"name": x.get("name"), "value": x.get("value"),
                            "unit": x.get("unit"), "date": x.get("date")}
                            for x in (rows[:10] if rows else [])
                        ]
                    }
                    prompt = _prompt_macro_humano("macro_list", facts, "¿Te muestro solo inflación, tasa o dólar?")
                    for chunk in _stream_no_fin(prompt):
                        yield chunk
                    yield "data: \n\n"

                    yield f"data: Encontré {len(rows)} variable(s). Mostrando las primeras:\n\n"
                    for i, r in enumerate(rows[:10], start=1):
                        yield f"data: {_fmt_macro_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"
                MEM.set(req.session_id, {"last_country": countries[0]})
                return StreamingResponse(gen_all(), media_type="text/event-stream")

            if len(countries) >= 2:
                rows = []
                for m in macros:
                    rows.extend(macro_compare(m, countries) or [])
                if not rows:
                    reason = _diagnose_no_results("macro_compare", plan=None, text=text, macros=macros, rows=rows)
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries}), media_type="text/event-stream")

                def gen_cmp():
                    yield f"data: Filtros → variables: {', '.join(macros)} | países: {' | '.join(countries)}\n\n"
                    # --- MACRO WRITER (COMPARE) ---
                    facts = {
                        "type": "macro_compare",
                        "variables": macros,
                        "countries": countries,
                        "rows": [
                            {"country": x.get("country"), "name": x.get("name"),
                            "value": x.get("value"), "unit": x.get("unit"), "date": x.get("date")}
                            for x in (rows or [])
                        ]
                    }
                    prompt = _prompt_macro_humano("macro_compare", facts, "¿Agrego otro país o otra variable?")
                    for chunk in _stream_no_fin(prompt):
                        yield chunk
                    yield "data: \n\n"

                    for i, r in enumerate(rows, start=1):
                        yield f"data: {_fmt_macro_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_cmp(), media_type="text/event-stream")

            if len(countries) == 1:
                # primera macro mencionada por simplicidad
                r = macro_lookup(macros[0], countries[0])
                if not r:
                    reason = _diagnose_no_results("macro_lookup", plan=None, text=text, macros=macros, rows=[])
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries[0]}), media_type="text/event-stream")
                def gen_one():
                    yield f"data: Filtros → país: {countries[0]} | variable: {macros[0]}\n\n"
                    # --- MACRO WRITER (LOOKUP) ---
                    facts = {
                        "type": "macro_lookup",
                        "variable": macros[0],
                        "country": countries[0],
                        "value": (r or {}).get("value"),
                        "unit": (r or {}).get("unit"),
                        "date": (r or {}).get("date"),
                        "name": (r or {}).get("name"),
                    }
                    prompt = _prompt_macro_humano("macro_lookup", facts, "¿La comparamos con otro país o prefieres la serie histórica?")
                    for chunk in _stream_no_fin(prompt):
                        yield chunk
                    yield "data: \n\n"

                    yield f"data: {_fmt_macro_row(r)}\n\n"
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_one(), media_type="text/event-stream")


            intent_guess = ("macro_lookup" if len(countries) == 1
                else "macro_compare" if len(countries) >= 2
                else "macro_list")
            reason = _diagnose_no_results(intent_guess, plan=None, text=text, macros=macros)
            return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")
        except Exception as e:
            reason = f"ocurrió un error interno ({type(e).__name__}). ¿Intentamos de nuevo con filtros más simples?"
            return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")


    # justo después del branch refine (y antes del planner)
    last = MEM.get(req.session_id) or {}
    if last and re.search(r"\b(háblame\s+de|hablame\s+de|ahora\s+en|y\s+en)\b", _norm(text or "")):
        # Si detecto “háblame de … / ahora en …”, reusar la última intención
        countries = _extract_countries(text) or []
        if countries:
            # parchar filtros en memoria
            lastf = dict(last.get("last_filters") or {})
            lastf["country"] = countries if len(countries) > 1 else countries[0]
            MEM.set(req.session_id, {"last_filters": lastf, "last_intent": last.get("last_intent")})

            # fuerza la intención anterior si es de negocio
            reuse_intent = last.get("last_intent") or "lookup"
            # crea un plan mínimo para caer en la rama correcta
            plan = Plan(
                intent=reuse_intent,
                filters=_normalize_plan_filters(merge_with_memory(lastf, req.session_id), text),
                top_k=getattr(S, "top_k", 5),
                limit=min(max(req.limit or 100, 1), getattr(S, "chat_list_max", 1000)),
            )
            # y deja que el flujo continúe con este plan (no devuelvas aquí)




        # Detectar intents especiales por texto ANTES del planner
    force_topn  = _is_topn_query(text)
    force_trend = _is_trend_query(text)

    # 1) Planner LLM (si aplica) + heurística directa  (TIMED)
    planner_ms = None                     # ← NUEVO
    t_pl0 = _now_ms()                     # ← NUEVO
    try:
        if 'plan' not in locals() or plan is None:
            plan = _plan_from_llm(text)
    finally:
        planner_ms = _now_ms() - t_pl0    # ← NUEVO

    heur_now = _guess_filters(text)  # SOLO alias explícitos del turno
    base_filters = plan.filters if plan else heur_now

    # 2) Filtros inteligentes (base + LLM + semántico si falta category)
    merged0 = build_filters_smart(text, base_filters)

    # 3) Preferir categoría anterior si NO se mencionó explícitamente una nueva
    prefer_last_cat = not bool(heur_now.get("category"))

    # 4) Fusión con memoria, señalando qué se mencionó explícitamente
    merged = merge_with_memory(
        merged0,
        req.session_id,
        prefer_last_cat=prefer_last_cat,
        mentioned={
            "category": "category" in heur_now,
            "country":  "country"  in heur_now,
            "store":    "store"    in heur_now,
        },
    )




    # 5) Si no hubo plan, crear uno heurístico
    if not plan:
        plan = Plan(
            intent=_classify_intent_heuristic(text),
            filters=merged,
            top_k=getattr(S, "top_k", 5),
            limit=min(max(req.limit or 100, 1), getattr(S, "chat_list_max", 1000)),
        )
    else:
        plan.filters = merged

    plan.filters = _normalize_plan_filters(plan.filters, text)

    # Si el usuario NO mencionó categoría en este turno,
# y la intención es un TOP-N o TREND, NO rellenes categoría semánticamente:
    explicit_cat = bool(heur_now.get("category"))  # heur_now ya lo calculas arriba
    if plan.intent in ("topn", "trend") and not explicit_cat:
        plan.filters.pop("category", None)  # fuerza "sin categoría" → todo el país


    if force_topn:
        plan.intent = "topn"
    elif force_trend:
        plan.intent = "trend"

    # >>> BLOQUE NUEVO: si es una consulta genérica de "precios de productos", listar por país
    def _is_generic_prices(nt: str, *, has_category: bool) -> bool:
        """
        Considera 'genérico' solo si NO hay categoría detectada explícitamente
        y el usuario habla de 'precios de productos' (en plural).
        Evita dispararse con frases como 'precio del producto ... café'.
        """
        if has_category:
            return False
        return (("precio" in nt or "precios" in nt) and ("productos" in nt))

    nt2 = _norm(text)
    has_cat_hint = bool(heur_now.get("category") or (plan.filters or {}).get("category"))
    if _is_generic_prices(nt2, has_category=has_cat_hint):
        plan.intent = "list"
        max_allowed = getattr(S, "chat_list_max", 1000)
        default_list = getattr(S, "chat_list_default", 500)
        plan.limit = min(max(req.limit or default_list, 1), max_allowed)
        # No elimines la categoría detectada; limpia solo ruido
        if plan.filters:
            for k in ("brand", "store"):
                plan.filters.pop(k, None)
    # <<< FIN BLOQUE NUEVO


    # Log pre-ejecución (plan + filtros)
    _log_event("chat_stream_plan", {
        "sid": req.session_id,
        "message": text,
        "plan": plan.model_dump(),
        "prefer_last_cat": prefer_last_cat,
        "explicit_mentions": {"category": "category" in heur_now,
                              "country": "country" in heur_now,
                              "store": "store" in heur_now},
    })

    # === INTENTS ===

    # ---- LIST → stream de tabla simple ----
    if plan.intent == "list":
        try:
            rows = list_by_filter(
                plan.filters or None,limit = min(max(plan.limit or 100, 1), getattr(S, "chat_list_max", 1000)))

            if not rows and plan.filters and plan.filters.get("category"):
                f2 = dict(plan.filters); f2.pop("category", None)
                rows = list_by_filter(f2, limit=min(max(plan.limit or 100, 1), getattr(S, "chat_list_max", 1000)))
        except Exception as e:
            _log_event("chat_stream_list_error", {"sid": req.session_id, "err": str(e)[:200]})
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")


        remember_session(req.session_id, filters=plan.filters, intent="list", query=text, hits=rows)
        _log_event("chat_stream_list", {
            "sid": req.session_id,
            "filters": plan.filters,
            "count": len(rows),
            "sample_ids": [r.get("product_id") for r in (rows[:10] or [])],
        })
        if not rows:
            reason = _diagnose_no_results("list", plan=plan, text=text, rows=rows)
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        def gen():
            # --- VIZ_PROMPT ---
            summary_txt = ""
            # try:
            #     vizp = _maybe_viz_prompt("list", plan.filters or {}, rows=rows, user_prompt=text)
            # except NameError:
            #     vizp = None
            # if vizp:
            #     yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            # yield f"data: {_filters_head(plan.filters)}\n\n"
            # # Mini-writer para saludo + contexto + 1-2 hallazgos + CTA
            # try:
            #     sample = [
            #         {
            #             "name": r.get("name"), "brand": r.get("brand"),
            #             "price": r.get("price"), "currency": r.get("currency"),
            #             "store": r.get("store")
            #         } for r in rows[:5]
            #     ]
            #     prompt_summary = (
            #         "Eres el asistente del SPI. Responde con: saludo breve → contexto "
            #         "(país/categoría si están) → breve resumen de hallazgos (menciona 1–2 ejemplos) "
            #         "→ CTA único (p.ej., \"¿Te muestro solo los más baratos por tienda?\").\n"
            #         f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}\n"
            #         f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
            #     )
            #     summary_txt = llm_chat.generate(prompt_summary).strip()
            #     if summary_txt:
            #         yield f"data: {summary_txt}\n\n"
            # except Exception:
            #     pass

            # yield f"data: Encontré {len(rows)} producto(s). Mostrando los primeros 10:\n\n"
            # for i, r in enumerate(rows[:10], start=1):
            #     line = (
            #         f"{i}. {r.get('name')} · Marca: {r.get('brand')} · "
            #         f"Pres: {r.get('size')}{r.get('unit')} · "
            #         f"Precio: {r.get('price')} {r.get('currency')} · "
            #         f"Tienda: {r.get('store')} · País: {r.get('country')} "
            #         f"[{r.get('product_id')}]"
            #     )
            #     yield f"data: {line}\n\n"

            # yield "data: Sugerencia: ¿quieres ver solo los más baratos por tienda o filtrar por marca?\n\n"
            # yield "data: [FIN]\n\n"
            try:
                sample = [
                    {
                        "name": r.get("name"), "brand": r.get("brand"),
                        "price": r.get("price"), "currency": r.get("currency"),
                        "store": r.get("store")
                    } for r in rows[:5]
                ]
                prompt_summary = (
                    "Eres el asistente del SPI. Responde con: saludo breve → contexto "
                    "(país/categoría si están) → breve resumen de hallazgos (menciona 1–2 ejemplos) "
                    "→ CTA único.\n"
                    f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}\n"
                    f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
                )
                summary_txt = llm_chat.generate(prompt_summary).strip()
            except Exception:
                pass

            yield f"data: {_filters_head(plan.filters)}\n\n"
            if summary_txt:
                yield f"data: {summary_txt}\n\n"

            # VIZ_PROMPT basado en la respuesta generada (summary_txt)
            vizp = _maybe_viz_prompt("list", plan.filters or {}, rows=rows, user_prompt=text, rag_response=summary_txt)
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: Encontré {len(rows)} producto(s). Mostrando los primeros 10:\n\n"
            for i, r in enumerate(rows[:10], start=1):
                line = (
                    f"{i}. {r.get('name')} · Marca: {r.get('brand')} · "
                    f"Pres: {r.get('size')}{r.get('unit')} · "
                    f"Precio: {r.get('price')} {r.get('currency')} · "
                    f"Tienda: {r.get('store')} · País: {r.get('country')} "
                    f"[{r.get('product_id')}]"
                )
                yield f"data: {line}\n\n"
            yield "data: Sugerencia: ¿quieres ver solo los más baratos por tienda o filtrar por marca?\n\n"
            yield "data: [FIN]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})



        # ---- TOPN → top N más baratos/caros ----
    if plan.intent == "topn":
        try:
            n, mode = _extract_topn(text)
            # trae una muestra amplia y ordena en memoria
            rows = list_by_filter(
                plan.filters or None,
                limit=min(max(plan.limit or getattr(S, "chat_list_default", 500), 1), getattr(S, "chat_list_max", 1000))
            ) or []
        except Exception as e:
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        if not rows:
            reason = _diagnose_no_results("list", plan=plan, text=text, rows=rows)
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        rows = [r for r in rows if r.get("price") is not None]
        rows.sort(key=lambda r: float(r["price"]), reverse=(mode != "cheap"))
        top = rows[:n]

        def gen_topn():
            # VIZ opcional (barras topN)
            # try:
            #     vizp = _maybe_viz_prompt("topn", plan.filters or {}, rows=top, user_prompt=text)
            # except NameError:
            #     vizp = None
            # if vizp:
            #     yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            # yield f"data: {_filters_head(plan.filters)}\n\n"

            # # Mini-writer humano (no stream ≠ evita doble FIN)
            # try:
            #     sample = [
            #         {"name": r.get("name"), "brand": r.get("brand"), "price": r.get("price"),
            #          "currency": r.get("currency"), "store": r.get("store")}
            #         for r in top[:3]
            #     ]
            #     prompt_summary = (
            #         "Eres el asistente del SPI. Formato: saludo breve → contexto (país/categoría/tienda si están) "
            #         "→ resumen del TOP con 1–2 ejemplos → CTA único (p.ej., \"¿Filtramos por tienda o marca?\").\n"
            #         f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}, n={n}, modo={mode}\n"
            #         f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
            #     )
            #     txt = llm_chat.generate(prompt_summary).strip()
            #     if txt:
            #         yield f"data: {txt}\n\n"
            # except Exception:
            #     pass

            # yield f"data: TOP {n} {'más baratos' if mode=='cheap' else 'más caros'}:\n\n"
            # for i, r in enumerate(top, 1):
            #     yield f"data: {_fmt_row(r, i)}\n\n"
            # yield "data: [FIN]\n\n"
            summary_txt = ""
            try:
                sample = [
                    {"name": r.get("name"), "brand": r.get("brand"), "price": r.get("price"),
                     "currency": r.get("currency"), "store": r.get("store")}
                    for r in top[:3]
                ]
                prompt_summary = (
                    "Eres el asistente del SPI. Formato: saludo breve → contexto "
                    "→ resumen del TOP con 1–2 ejemplos → CTA único.\n"
                    f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}, n={n}, modo={mode}\n"
                    f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
                )
                summary_txt = llm_chat.generate(prompt_summary).strip()
            except Exception:
                pass

            yield f"data: {_filters_head(plan.filters)}\n\n"
            if summary_txt:
                yield f"data: {summary_txt}\n\n"

            vizp = _maybe_viz_prompt("topn", plan.filters or {}, rows=top, user_prompt=text, rag_response=summary_txt)
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: TOP {n} {'más baratos' if mode=='cheap' else 'más caros'}:\n\n"
            for i, r in enumerate(top, 1):
                yield f"data: {_fmt_row(r, i)}\n\n"
            yield "data: [FIN]\n\n"

        return StreamingResponse(gen_topn(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})



     # ---- TREND → tendencia últimos 30 días (o rango corto) ----
    if plan.intent == "trend":
        try:
            # puedes extraer días del texto si quieres; por ahora 30
            days = 30
            ser = series_prices(plan.filters or None, days=days) or []
        except Exception as e:
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        if not ser:
            reason = _diagnose_no_results("aggregate", plan=plan, text=text, agg={"groups": []})
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        vals = [s.get("value") for s in ser if s.get("value") is not None]
        if len(vals) >= 2 and vals[0]:
            pct = (vals[-1] - vals[0]) / vals[0] * 100.0
        else:
            pct = 0.0

        facts = {
            "days": days,
            "pct": pct,
            "last": vals[-1] if vals else None,
            "currency": ser[-1].get("currency"),
            "n": len(vals),
            "date_start": ser[0].get("date"),
            "date_end": ser[-1].get("date"),
            "filters": plan.filters or {}
        }

        prompt = (
            "Eres el asistente del SPI. Responde con: saludo breve → contexto (país/categoría, rango de fechas) "
            "→ tendencia con % y último valor/fecha → CTA único (p.ej., \"¿Genero la gráfica o comparo otro país?\").\n"
            f"FACTS(JSON): {json.dumps(facts, ensure_ascii=False)}\n"
            "RESPUESTA:"
        )

        def gen_trend():
            # VIZ_PROMPT: línea temporal
            # try:
            #     vizp = _maybe_viz_prompt("trend", plan.filters or {}, series=ser, user_prompt=text)
            # except NameError:
            #     vizp = None
            # if vizp:
            #     yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            # yield f"data: {_filters_head(plan.filters)}\n\n"

            # # Writer en stream (usa _stream_no_fin para evitar doble FIN)
            # for chunk in _stream_no_fin(prompt):
            #     yield chunk
            # yield "data: [FIN]\n\n"
            rag_buf = []
            yield f"data: {_filters_head(plan.filters)}\n\n"
            for chunk in _stream_no_fin(prompt):
                try:
                    content = chunk.split("data:",1)[1].strip()
                except Exception:
                    content = chunk
                if content and content != "[FIN]":
                    rag_buf.append(content)
                yield chunk
            full_resp = " ".join(rag_buf).strip()
            vizp = _maybe_viz_prompt("trend", plan.filters or {}, series=ser, user_prompt=text, rag_response=full_resp)
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"
            yield "data: [FIN]\n\n"

        return StreamingResponse(gen_trend(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})



    # ---- COMPARE → N países (2..10) usando SOLO columnas de la BD ----
    if plan.intent == "compare":
        try:
            countries = _extract_countries(text, max_n=10)
            cat = (plan.filters or {}).get("category")

            # --- NUEVO: inferir categoría si falta (sinónimos -> semántico) ---
            if not cat:
                cat = _canonicalize_category(text)

                if cat:
                    plan.filters = dict(plan.filters or {}, category=cat)

            # Si el usuario mencionó >=2 países y (ahora sí) hay categoría → multi-compare
            if len(countries) >= 2 and cat:
                per_country_rows: list[tuple[str, list[dict]]] = []
                top_per_country = max(min(plan.top_k or 3, 5), 1)  # 1..5 por país

                # No arrastramos tienda al comparar países
                for code in countries:
                    f = dict(plan.filters or {})
                    f["country"] = code
                    f.pop("store", None)
                    rows = list_by_filter(
                        f, limit=min(max(plan.limit or 100, 1), 1000)
                    ) or []
                    per_country_rows.append((code, rows))

                with_data = [(c, rows) for c, rows in per_country_rows if rows]
                without_data = [c for c, rows in per_country_rows if not rows]

                _log_event("chat_stream_compare_multi", {
                    "sid": req.session_id,
                    "category": cat,
                    "countries": countries,
                    "with_data": {c: len(rows) for c, rows in with_data},
                    "without_data": without_data[:10],
                    "samples": {c: [r.get("product_id") for r in rows[:3]] for c, rows in with_data},
                })

                # Política: necesitamos al menos 2 países con datos
                if len(with_data) < 2:
                    try:
                        suggestions = {}
                        for code in without_data:
                            # agregados por categoría en ese país (sin forzar "cafe")
                            agg_cat = aggregate_prices({"country": code}, by="category") or {}
                            groups = agg_cat.get("groups") or []
                            # prioriza categorías que contengan "cafe" (normalizado)
                            def _n(s): return (s or "").lower()
                            candidates = [g.get("group") for g in groups if g and g.get("group")]
                            cafe_like = [c for c in candidates if "cafe" in _n(c) or "caf" in _n(c) or "coffee" in _n(c)]
                            suggestions[code] = cafe_like[:3] or candidates[:3]  # top 3
                    except Exception:
                        suggestions = {}

                    reason = f"necesito al menos 2 países con datos para '{cat}', pero tuve " \
                            f"{len(with_data)} con datos y {len(without_data)} sin datos."
                    def gen_hint():
                        yield f"data: Hola. No pude comparar porque {reason}\n\n"
                        yield f"data: Filtros → países: {countries} | categoría: {cat}\n\n"
                        for code in without_data:
                            opts = suggestions.get(code) or []
                            if opts:
                                yield f"data: Sugerencia para {code}: prueba con categoría(s) {', '.join(opts)}\n\n"
                            else:
                                yield f"data: Sugerencia para {code}: prueba sin categoría o con otra similar.\n\n"
                        yield "data: [FIN]\n\n"
                    return StreamingResponse(gen_hint(), media_type="text/event-stream")


                # --- Si quieres redacción humana, usa LLM con "hechos" agregados
                if getattr(S, "compare_llm", True):
                    # preparar hechos por país
                    facts = {"category": cat, "countries": []}
                    for c, rows in with_data:
                        prices = [r.get("price") for r in rows if r.get("price") is not None]
                        if not prices:
                            continue
                        # moneda más común
                        from collections import Counter
                        cur = None
                        if rows:
                            cur = Counter([r.get("currency") for r in rows if r.get("currency")]).most_common(1)[0][0]
                        brands = {r.get("brand") for r in rows if r.get("brand")}
                        facts["countries"].append({
                            "country": c,
                            "avg": sum(prices) / max(len(prices), 1),
                            "min": min(prices),
                            "max": max(prices),
                            "n": len(prices),
                            "brands_n": len(brands),
                            "currency": cur
                        })

                    ctx_json = json.dumps(facts, ensure_ascii=False)
                    prompt = (
                        f"Eres el asistente del Sistema Pricing Inteligente (SPI). "
                        f"Redacta de forma natural y amable una comparativa de precios para la categoría '{cat}'. "
                        f"Usa exclusivamente estos HECHOS (JSON): {ctx_json}. "
                        "Estructura: saludo breve → contexto → respuesta comparativa (promedio, mínimo, máximo y conteo de marcas por país) → "
                        "cierre con un mini resumen y un call to action para filtrar más."
                    )

                    def gen():
                        # VIZ opcional
                        try:
                            groups = [
                                {"group": d["country"], "avg": d["avg"], "min": d["min"], "max": d["max"]}
                                for d in facts["countries"]
                            ]
                            vizp = _maybe_viz_prompt(
                                "aggregate",
                                {"category": cat, "country": [d["country"] for d in facts["countries"]]},
                                agg={"groups": groups},
                                group_by="country",
                                user_prompt=text,
                            )
                        except NameError:
                            vizp = None
                        if vizp:
                            yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                        head = " | ".join(countries)
                        yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"

                        # LLM stream
                        t_llm0 = _now_ms(); first_token_ms = None
                        for chunk in _stream_no_fin(prompt):
                            if first_token_ms is None:
                                first_token_ms = _now_ms()
                            yield chunk
                        _log_perf("chat_stream_compare_llm_perf", {
                            "gen_model": llm_chat.model,
                            "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
                            "sid": req.session_id, "q": text[:120]
                        })
                        yield "data: [FIN]\n\n"


                    return StreamingResponse(
                        gen(), media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )

                # --- Respaldo determinista (sin LLM)
                def gen():
                    try:
                        groups = []
                        for c, rows in with_data:
                            prices = [r.get("price") for r in rows if r.get("price") is not None]
                            if not prices:
                                continue
                            groups.append({
                                "group": c,
                                "avg": sum(prices) / max(len(prices), 1),
                                "min": min(prices),
                                "max": max(prices),
                            })
                        vizp = _maybe_viz_prompt(
                            "aggregate",
                            {"category": cat, "country": [c for c, _ in with_data]},
                            agg={"groups": groups},
                            group_by="country",
                            user_prompt=text,
                        )
                    except NameError:
                        vizp = None
                    if vizp:
                        yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    head = " | ".join(countries)
                    yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"
                    if without_data:
                        yield f"data: Aviso: sin registros para: {', '.join(without_data)}\n\n"
                    for c, rows in with_data:
                        yield f"data: — País {c}: mostrando hasta {top_per_country} producto(s)\n\n"
                        for i, r in enumerate(rows[:top_per_country], start=1):
                            yield f"data: {_fmt_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"

                return StreamingResponse(
                    gen(), media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )

            # --- Fallback: comparación simple cuando no hay (N países + categoría) ---
            effective_q = pick_effective_query(
                text, req.session_id, prefer_last_cat=not bool((plan.filters or {}).get("category"))
            )
            _log_event("chat_stream_plan", {
                "sid": req.session_id,
                "message": text,
                "plan": plan.model_dump() if plan else None,
                "merged_filters": merged
            })
            hits = retrieve(effective_q, plan.filters or None)[: max(plan.top_k or 5, 5)]
            _log_event("chat_stream_compare_single", {
                "sid": req.session_id,
                "filters": plan.filters,
                "effective_query": effective_q,
                "ids": [h.get("product_id") for h in (hits or [])],
            })
            if len(hits) < 2:
                reason = _diagnose_no_results("compare", plan=plan, text=text, hits=hits)
                return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

            def gen_single():
                try:
                    vizp = _maybe_viz_prompt("compare", plan.filters or {}, rows=hits[:2], user_prompt=text)
                except NameError:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"
                yield f"data: Filtros → país: {(plan.filters or {}).get('country') or '-'} | categoría: {(plan.filters or {}).get('category') or '-'} | tienda: {(plan.filters or {}).get('store') or '-'}\n\n"
                # --- Mini-writer humano (no stream) para saludo + contexto + mini-comparación + CTA ---
                try:
                    sample = [
                        {
                            "name": h.get("name"),
                            "brand": h.get("brand"),
                            "price": h.get("price"),
                            "currency": h.get("currency"),
                            "store": h.get("store"),
                            "country": h.get("country")
                        } for h in hits[:2]
                    ]
                    prompt_summary = (
                        "Eres el asistente del SPI. Responde con: saludo breve → contexto "
                        "(categoría/país/tienda si están) → mini-comparación clara de los 2 resultados "
                        "→ CTA único (p.ej., \"¿Quieres que lo ordene o convertir a la misma moneda?\").\n"
                        f"Contexto: filtros={json.dumps(plan.filters or {}, ensure_ascii=False)}\n"
                        f"Ejemplos(JSON): {json.dumps(sample, ensure_ascii=False)}"
                    )
                    summary_txt = llm_chat.generate(prompt_summary).strip()
                    if summary_txt:
                        yield f"data: {summary_txt}\n\n"
                except Exception:
                    pass

                yield "data: Comparativa simple (primeros 2 resultados):\n\n"
                for i, h in enumerate(hits[:2], start=1):
                    yield f"data: {_fmt_row(h, i)}\n\n"
                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_single(), media_type="text/event-stream")

        except Exception as e:
            _log_event("chat_stream_compare_error", {"sid": req.session_id, "err": str(e)[:200]})
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")


    # ---- AGGREGATE → stream de resumen agregado ----
    if plan.intent == "aggregate":
        try:
            t_agg0 = _now_ms()
            agg = aggregate_prices(plan.filters or None, by=plan.group_by or "category")
            t_agg1 = _now_ms()
        except Exception as e:
            _log_event("chat_stream_aggregate_error", {"sid": req.session_id, "err": str(e)[:200]})
            reason = f"ocurrió un error interno ({type(e).__name__})."
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        remember_session(req.session_id, filters=plan.filters, intent="aggregate", query=text, hits=[])
        _log_event("chat_stream_aggregate", {
            "sid": req.session_id,
            "filters": plan.filters,
            "group_by": plan.group_by or "category",
            "rows": agg.get("groups", [])[:10],
        })
        if not agg.get("groups"):
            reason = _diagnose_no_results("aggregate", plan=plan, text=text, agg=agg)
            return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

        ctx = json.dumps({"group_by": plan.group_by or "category",
                        "rows": agg.get("groups", [])}, ensure_ascii=False)
        prompt = (
                    f"Eres {ASSISTANT_NAME}. "
                    "Redacta SIEMPRE en 2–4 frases con: saludo breve → contexto (group_by y filtros) "
                    "→ resumen de cifras visibles en el JSON → CTA único (p.ej., "
                    "\"¿Genero una gráfica o convierto a una moneda común?\"). "
                    f"Usa SOLO este JSON: {ctx}. No inventes valores."
                    )


        def gen():
            # VIZ_PROMPT (no cuenta para TTFB del LLM)
            try:
                vizp = _maybe_viz_prompt(
                    "aggregate",
                    plan.filters or {},
                    agg=agg,
                    group_by=plan.group_by or "category",
                    user_prompt=text,
                )
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"

            # ---- LLM STREAM con TTFB y duración total ----
            t_llm0 = _now_ms()
            first_token_ms = None
            total_chars = 0
            for chunk in _stream_no_fin(prompt):
                if first_token_ms is None:
                    first_token_ms = _now_ms()
                total_chars += len(chunk)
                yield chunk
            t_llm1 = _now_ms()

            _log_perf("chat_stream_aggregate_perf", {
                "gen_model": llm_chat.model,
                "planner_ms": planner_ms,
                "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
                "llm_stream_ms": t_llm1 - t_llm0,
                "aggregate_ms": t_agg1 - t_agg0,
                "total_ms": _now_ms() - t_req0,
                "rows": len(agg.get("groups", [])),
                "filters": plan.filters,
                "q": text[:120],
                "sid": req.session_id
            })
            yield "data: [FIN]\n\n"


        return StreamingResponse(gen(), media_type="text/event-stream")


   # ---- LOOKUP (por defecto) → stream respuesta amable + contexto ----
    try:
        t_ret0 = _now_ms()
        effective_q = pick_effective_query(text, req.session_id, prefer_last_cat)

        # Empezamos con los filtros planeados
        facts_filters = dict(plan.filters or {})

        hits = retrieve(effective_q, facts_filters)[: plan.top_k or getattr(S, "top_k", 5)]

        # Fallback: si no hay hits y había categoría, reintenta sin categoría
        if not hits and facts_filters.get("category"):
            f2 = dict(facts_filters); f2.pop("category", None)
            h2 = retrieve(effective_q, f2)[: plan.top_k or getattr(S, "top_k", 5)]
            if h2:
                hits = h2
                facts_filters = f2   # <-- ¡clave! usa estos filtros para calcular facts

        t_ret1 = _now_ms()


        # Si los hits muestran una categoría dominante, úsala para facts
        from collections import Counter
        hit_cats = [h.get("category") for h in hits if h.get("category")]
        if hit_cats:
            top_cat = Counter(hit_cats).most_common(1)[0][0]
            if top_cat:
                facts_filters["category"] = top_cat


    except Exception as e:
        _log_event("chat_stream_lookup_error", {"sid": req.session_id, "err": str(e)[:200]})
        reason = f"ocurrió un error interno ({type(e).__name__})."
        return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

    remember_session(req.session_id, filters=plan.filters, intent="lookup", query=text, hits=hits)
    _log_event("chat_stream_lookup", {
        "sid": req.session_id,
        "filters": plan.filters,
        "effective_query": effective_q,
        "ids": [h.get("product_id") for h in (hits or [])],
    })

    if not hits:
        reason = _diagnose_no_results("lookup", plan=plan, text=text, hits=hits)

        return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

    ctx = _build_ctx(hits, plan.top_k or getattr(S, "top_k", 5))

    # --- HECHOS desde la base para que el LLM redacte ---
    # --- HECHOS desde la base para que el LLM redacte ---
    base_filters = dict(facts_filters)
    base_filters.pop("store", None)  # promedio nacional sin tienda

    # 1) Promedio nacional (base de BD)
    rows_all = list_by_filter(base_filters, limit=min(getattr(S, "aggregate_limit", 5000), 5000))
    prices = [r.get("price") for r in rows_all if r.get("price") is not None]

    cur = None
    if rows_all:
        from collections import Counter
        cur = Counter([r.get("currency") for r in rows_all if r.get("currency")]).most_common(1)[0][0]

    # Fallback: si no hubo filas para avg pero sí hay hits, calcula desde hits
    if not prices and hits:
        prices = [h.get("price") for h in hits if h.get("price") is not None]
        if prices and not cur:
            from collections import Counter
            cur = Counter([h.get("currency") for h in hits if h.get("currency")]).most_common(1)[0][0]

    avg_all = (sum(prices) / max(len(prices), 1)) if prices else None

    # 2) Promedios por marca (BD)
    agg_brand = aggregate_prices(base_filters, by="brand")
    groups = agg_brand.get("groups") or []
    groups = [g for g in groups if g and g.get("group") not in (None, "", "N/A") and g.get("avg") is not None]

    # Fallback: si no hay grupos en BD, calcula desde los hits
    if not groups and hits:
        tmp_sum, tmp_n = {}, {}
        for h in hits:
            b, p = (h.get("brand") or "N/A"), h.get("price")
            if p is None: continue
            tmp_sum[b] = tmp_sum.get(b, 0.0) + float(p)
            tmp_n[b] = tmp_n.get(b, 0) + 1
        groups = [{"group": b, "avg": tmp_sum[b]/max(tmp_n[b],1), "n": tmp_n[b]} for b in tmp_sum]

    # Orden y selección top-N
    groups.sort(key=lambda g: float(g.get("avg") or 0.0), reverse=True)
    max_lines = int(getattr(S, "brand_avg_max", 8))
    brands = []
    for g in groups[:max_lines]:
        brands.append({
            "brand": str(g.get("group")),
            "avg": float(g.get("avg")),
            "n": int(g.get("n") or 0),
        })

    # Rango min–máx para el resumen final
    brand_range = None
    if brands:
        lo = min(brands, key=lambda b: b["avg"])
        hi = max(brands, key=lambda b: b["avg"])
        brand_range = {
            "min_brand": lo["brand"], "min_avg": lo["avg"],
            "max_brand": hi["brand"], "max_avg": hi["avg"],
        }

    facts = {
        "country": (plan.filters or {}).get("country"),
        "category": facts_filters.get("category") or (plan.filters or {}).get("category"),
        "currency": cur,
        "national_avg": float(avg_all) if avg_all is not None else None,
        "n": len(prices),
        "brands": brands,
        "brand_range": brand_range,
    }


    prompt = _prompt_lookup_from_facts(text, facts, ctx)

    def gen_lookup():
        # VIZ_PROMPT + encabezado (no cuentan para TTFB del LLM)
        # try:
        #     vizp = _maybe_viz_prompt("lookup", plan.filters or {}, rows=hits, user_prompt=text)
        # except NameError:
        #     vizp = None
        # if vizp:
        #     yield f"data: [VIZ_PROMPT] {vizp}\n\n"

        # yield f"data: {_filters_head(plan.filters)}\n\n"

        # # ---- LLM STREAM con TTFB y duración total ----

        # t_llm0 = _now_ms()
        # first_token_ms = None
        # total_chars = 0
        # for chunk in _stream_no_fin(prompt):
        #     if first_token_ms is None:
        #         first_token_ms = _now_ms()
        #     total_chars += len(chunk)
        #     yield chunk
        # t_llm1 = _now_ms()

        # _log_perf("chat_stream_lookup_perf", {
        #     "gen_model": llm_chat.model,
        #     "planner_ms": planner_ms,
        #     "retrieve_ms": t_ret1 - t_ret0,
        #     "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
        #     "llm_stream_ms": t_llm1 - t_llm0,
        #     "total_ms": _now_ms() - t_req0,
        #     "hits": len(hits),
        #     "ctx_len_chars": len(ctx),
        #     "top_k": plan.top_k or getattr(S, "top_k", 5),
        #     "filters": plan.filters,
        #     "q": text[:120],
        #     "sid": req.session_id,
        # })

        # yield "data: [FIN]\n\n"
        rag_buf = []
        yield f"data: {_filters_head(plan.filters)}\n\n"
        for chunk in _stream_no_fin(prompt):
            # chunk ya viene con 'data: ...'; extraer contenido
            try:
                content = chunk.split("data:",1)[1].strip()
            except Exception:
                content = chunk
            if content and content != "[FIN]":
                rag_buf.append(content)
            yield chunk
        full_resp = " ".join(rag_buf).strip()
        vizp = _maybe_viz_prompt("lookup", plan.filters or {}, rows=hits, user_prompt=text, rag_response=full_resp)
        if vizp:
            yield f"data: [VIZ_PROMPT] {vizp}\n\n"
        yield "data: [FIN]\n\n"

    return StreamingResponse(gen_lookup(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Feedback
# -----------------------------------------------------------------------------
class FeedbackReq(BaseModel):
    message: str
    reply: str
    rating: Literal["up","down"]
    comment: Optional[str] = None
    planner: Optional[Dict] = None

@app.post("/feedback", tags=["meta"])
def feedback(req: FeedbackReq):
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "message": req.message,
        "reply": req.reply,
        "rating": req.rating,
        "comment": req.comment,
        "planner": req.planner,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}
