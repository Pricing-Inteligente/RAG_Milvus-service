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
from retrieve_lasso import query_lasso_models
import time



# Config
from settings import get_settings
S = get_settings()

# Milvus helpers
from retrieve import retrieve, list_by_filter, aggregate_prices

# Inteligencia de intención / categoría
from intent_llm import parse_intent
from category_resolver import resolve_category_semantic



# --- LASSO / Influencia (solo México) ---
LASSO_COLLECTION = os.getenv("LASSO_COLLECTION", "lasso_models")

LASSO_KEYWORDS = {
    # núcleo
    "influencia", "influenciar", "influye", "influyen", "influyó",
    "impacto", "impacta", "impactan",
    "efecto", "efectos",
    "descomposicion", "descomposición",
    # otros
    "grado", "coeficiente", "coeficientes", "variacion", "variación", "sensibilidad",
    "afecta", "afectan"
}

# para bloquear países distintos a MX si aparecen explícitos
NON_MX_COUNTRIES = {
    "argentina","brasil","brasil","br","chile","cl","colombia","co","peru","pe",
    "panama","pa","uruguay","uy","paraguay","py","ecuador","ec","bolivia","bo",
    "venezuela","ve","guatemala","gt","costa rica","cr","honduras","hn","nicaragua","ni",
    "el salvador","sv","mexico df","méxico df"  # <- no bloquea, solo ejemplos de texto
}


import re

# Países distintos a MX (nombres) y sus códigos ISO2
_NON_MX_WORDS = [
    "argentina","brasil","brazil","chile","colombia","peru","panama","uruguay",
    "paraguay","ecuador","bolivia","venezuela","guatemala","costa rica",
    "honduras","nicaragua","el salvador"
]
_NON_MX_ISO2 = ["ar","br","cl","co","pe","pa","uy","py","ec","bo","ve","gt","cr","hn","ni","sv"]

# Detectores robustos con límites de palabra
_NON_MX_REGEX = re.compile(
    r"\b(" + "|".join(map(re.escape, _NON_MX_WORDS)) + r")\b|\b(" + "|".join(_NON_MX_ISO2) + r")\b",
    flags=re.IGNORECASE
)
_MX_REGEX = re.compile(r"\b(m[eé]xico|mx)\b", flags=re.IGNORECASE)


def detect_lasso_influence_intent(message: str) -> dict | None:
    text = (message or "").strip().lower()
    if not any(k in text for k in LASSO_KEYWORDS):
        return None

    # Bloquea solo si se menciona otro país y NO aparece MX
    if _NON_MX_REGEX.search(text) and not _MX_REGEX.search(text):
        return {"intent": "lasso_influence_blocked_non_mx"}

    # --- 1) MARCA explícita ---
    m = re.search(r"(?:de\s+la\s+marca|de\s+marca|marca)\s+([a-z0-9\-\.\s_]+)", text, flags=re.IGNORECASE)
    if m:
        term = re.sub(r"\s+", " ", m.group(1)).strip(" ._-")
        return {"intent": "lasso_influence", "by": "brand", "term": term}

    # --- 2) PRODUCTO: “en/sobre <producto> de/en méxico” (captura solo el producto)
    m3 = re.search(
        r"(?:en|sobre)\s+(?:el|la|los|las)?\s*([a-z0-9_áéíóúñ\-\s]+?)\s+(?:de|en)\s+m[eé]xico\b",
        text, flags=re.IGNORECASE
    )
    if m3:
        term = re.sub(r"\s+", " ", m3.group(1)).strip(" ._-")
        return {"intent": "lasso_influence", "by": "product", "term": term}

    # --- 3) FALLBACK: toma lo previo a “de/en méxico”, pero reduce al último token tipo “arroz” ---
    m2 = re.search(r"([a-z0-9\-\.\s_]+?)\s+(?:de|en)\s+m[eé]xico\b", text, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", (m2.group(1) if m2 else text)).strip(" ._-")
    # quita artículos y quédate con la última palabra útil
    tokens = [t for t in re.split(r"\s+", raw) if t not in {"de","en","la","el","las","los"}]
    term = tokens[-1] if tokens else raw

    return {"intent": "lasso_influence", "by": "product", "term": term}


# --- NUEVO: Detección de comparación LASSO entre dos productos/marcas ---
def detect_lasso_influence_compare_intent(message: str) -> dict | None:
    """Detecta consultas como 'muestrame descomposicion de precios de arroz y cafe'
    devolviendo intent 'lasso_influence_compare' con lista de términos.
    Reglas:
      - Debe contener alguna palabra clave LASSO.
      - Debe contener un separador ' y ' o ',' indicando dos términos distintos.
      - Solo México (bloquea si aparece otro país y no aparece MX).
    """
    text_raw = (message or "").strip()
    text = text_raw.lower()
    if not any(k in text for k in LASSO_KEYWORDS):
        return None
    # País distinto de MX bloquea
    if _NON_MX_REGEX.search(text) and not _MX_REGEX.search(text):
        return None  # deja que el detector normal lo bloquee si aplica

    # Normaliza separadores a DOS términos: usa el último separador encontrado
    if ' y ' in text:
        parts = [p.strip() for p in text.rsplit(' y ', 1) if p.strip()]
    else:
        # Fallback: usar la última coma
        parts = [p.strip() for p in text.rsplit(',', 1) if p.strip()]
    if len(parts) != 2:
        return None

    # Limpieza robusta de cada término (extrae lo posterior a "precios de" o al último "de")
    STOP = {
        "muestrame","muéstrame","descomposicion","descomposición","precios","precio","de","del","la","el","los","las",
        "influencia","impacto","coeficientes","coeficiente","en","mexico","méxico","variables","variable","presentacion","presentación",
        "producto","productos","marca","marcas"
    }
    def _clean_term(t: str) -> str:
        tl = t.lower().strip()
        # Quita prefijos comunes "en precios de ..."
        tl = re.sub(r"\ben\s+precios?\s+de\s+", "", tl)
        # Si hay un "precios de" explícito, toma lo que sigue
        m = re.search(r"precios?\s+de\s+(.+)$", tl)
        if m:
            tl = m.group(1)
        else:
            # En su defecto, toma lo que sigue al último " de "
            if " de " in tl:
                tl = tl.rsplit(" de ", 1)[-1]
        # Tokeniza y elimina stopwords
        tokens = [re.sub(r"[^a-z0-9áéíóúñ%/]", "", w) for w in re.split(r"\s+", tl)]
        tokens = [w for w in tokens if w and w not in STOP]
        if not tokens:
            # Fallback: última palabra alfanumérica del original
            m2 = re.findall(r"[a-z0-9áéíóúñ%/]+", t.lower())
            return m2[-1] if m2 else ""
        # Prioriza las últimas 1–2 palabras (ej.: "arroz", "leche entera")
        return " ".join(tokens[-2:])

    term_a = _clean_term(parts[0])
    term_b = _clean_term(parts[1])
    if not term_a or not term_b or term_a == term_b:
        return None

    return {"intent": "lasso_influence_compare", "terms": [term_a, term_b], "country": "MX"}


# --- NUEVO: Detección de influencia de UNA variable específica sobre un producto/marca ---
_LASSO_VAR_ALIASES = [
    ("coef_inflation_rate_pct_change", ["inflacion", "inflación", "inflation", "ipc general"]),
    ("coef_cambio_dolar_pct_change", ["tipo de cambio", "dolar", "dólar", "usd", "tc", "usd/mxn"]),
    ("coef_cpi_pct_change", ["ipc", "cpi", "indice de precios al consumidor", "índice de precios al consumidor"]),
    ("coef_interest_rate_pct_change", ["tasa de interes", "tasa de interés", "interes", "interés", "tasas"]),
    ("coef_gdp_pct_change", ["pib", "producto interno bruto", "gdp"]),
    ("coef_producer_prices_pct_change", ["precios al productor", "ipp", "ppi", "indice de precios al productor", "índice de precios al productor"]),
    ("coef_gini_pct_change", ["gini", "indice gini", "índice gini"]),
]

def _match_lasso_var_key(text: str) -> tuple[str|None, str|None]:
    t = text.lower()
    for key, aliases in _LASSO_VAR_ALIASES:
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", t):
                # Regresa key y una etiqueta humana aproximada
                label_map = {
                    "coef_inflation_rate_pct_change": "Inflación general (%)",
                    "coef_cambio_dolar_pct_change": "Tipo de cambio USD/MXN (%)",
                    "coef_cpi_pct_change": "CPI / IPC (%)",
                    "coef_interest_rate_pct_change": "Tasa de interés (%)",
                    "coef_gdp_pct_change": "PIB (%)",
                    "coef_producer_prices_pct_change": "Precios al productor (%)",
                    "coef_gini_pct_change": "Índice Gini (%)",
                }
                return key, label_map.get(key)
    return None, None

def detect_lasso_variable_influence_intent(message: str) -> dict | None:
    text_raw = (message or "").strip()
    text = text_raw.lower()
    # Debe hablar de influencia/impacto/efecto/descomposición
    if not any(k in text for k in LASSO_KEYWORDS):
        return None
    # Bloque países no MX explícitos
    if _NON_MX_REGEX.search(text) and not _MX_REGEX.search(text):
        return None

    # Detectar variable
    var_key, var_label = _match_lasso_var_key(text)
    if not var_key:
        return None

    # Extraer producto/marca (similar a compare): usa lo posterior a 'precios de' o al último ' de '
    # Primero descarta la parte donde detectaste la variable para evitar capturarla de nuevo
    t2 = re.sub(r"\b(influencia|influye|influyen|impacto|impacta|impactan|efecto|efectos|descomposici[oó]n)\b", " ", text)
    # Captura producto tras 'precios de'
    m = re.search(r"precios?\s+de\s+(.+)$", t2)
    if m:
        cand = m.group(1)
    else:
        # Busca 'en <producto>' o 'de <producto>'
        m2 = re.search(r"(?:en|sobre|para)\s+(?:los|las|el|la)?\s*(.+)$", t2)
        cand = m2.group(1) if m2 else t2
    # separa por ' y ' o coma y toma un solo término (una variable → un producto)
    parts = [p.strip() for p in re.split(r"\s+y\s+|,", cand) if p.strip()]
    # escoger el primer término como producto
    prod_raw = parts[0] if parts else cand

    STOP = {"muestrame","muéstrame","descomposicion","descomposición","precios","precio","de","del","la","el","los","las",
            "influencia","impacto","coeficientes","coeficiente","en","mexico","méxico","variables","variable","presentacion","presentación",
            "producto","productos","marca","marcas"}
    tokens = [re.sub(r"[^a-z0-9áéíóúñ%/]", "", w) for w in re.split(r"\s+", prod_raw)]
    tokens = [w for w in tokens if w and w not in STOP]
    if not tokens:
        return None
    term = " ".join(tokens[-2:])

    return {
        "intent": "lasso_var_influence",
        "var_key": var_key,
        "var_label": var_label or var_key,
        "by": "product",
        "term": term,
        "country": "MX",
    }




def _format_lasso_answer(rows: list[dict], by: str, term: str) -> str:
    """
    Estructura: saludo → contexto LASSO → hallazgos → CTA
    """
    if not rows:
        return (
            "Hola.\n\n"
            "Esta descomposición LASSO solo está disponible para **México** y no encontré ese "
            f"{'producto' if by=='product' else 'marca'} en los modelos. "
            f"¿Quieres que pruebe con otra {'marca' if by=='brand' else 'presentación'}?"
        )

    # elige el mejor por R²
    best = max(rows, key=lambda r: (r.get("r_squared") or 0.0))

    coef_map = [
        ("coef_inflation_rate_pct_change", "Inflación general (%)"),
        ("coef_cambio_dolar_pct_change",   "Tipo de cambio USD/MXN (%)"),
        ("coef_cpi_pct_change",            "CPI / IPC (%)"),
        ("coef_interest_rate_pct_change",  "Tasa de interés (%)"),
        ("coef_gdp_pct_change",            "PIB (%)"),
        ("coef_producer_prices_pct_change","Precios al productor (%)"),
        ("coef_gini_pct_change",           "Índice Gini (%)"),
    ]

    coefs = []
    for k, label in coef_map:
        v = best.get(k)
        if v is None:
            continue
        coefs.append((label, float(v)))

    # ordenar por magnitud
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    bullets = []
    for label, v in coefs:
        efecto = "↑ sube" if v > 0 else ("↓ baja" if v < 0 else "≈ neutro")
        bullets.append(f"- {label}: {v:.4f} ({efecto} el precio)")

    nombre = best.get("nombre") or (best.get("producto") or best.get("marca") or term)
    marca  = best.get("marca") or "-"
    prod   = best.get("producto") or "-"
    retail = best.get("retail") or "-"
    r2     = best.get("r_squared") or 0.0
    alpha  = best.get("best_alpha") or 0.0
    nobs   = int(best.get("n_obs") or 0)

    contexto = (
        "Usamos una regresión LASSO entrenada con historiales de **precios en México** y "
        "variables macro oficiales. LASSO selecciona variables y sus coeficientes se interpretan "
        "como influencia marginal sobre el precio (positivos empujan al alza; negativos, a la baja).\n"
    )

    hallazgos = "**Variables con mayor influencia (por magnitud):**\n" + "\n".join(bullets[:7])

    meta = f"\n\nModelo: {nombre} · Marca: {marca} · Producto: {prod} · Retail: {retail} · R²: {r2:.3f} · α: {alpha:.4f} · Obs: {nobs}"
    cta  = "\n\n¿Quieres comparar con otra marca/producto o que simule un what-if (por ejemplo, +1 pp en inflación)?"

    return "Hola.\n\n" + contexto + hallazgos + meta + cta




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
    # aliases directos (ipc, inflación, etc.)
    for alias, canon in NMACROS.items():
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
            if canon not in found:
                found.append(canon)
    # variantes "macro…"
    macro_any = (
        re.search(r"(variables?|indicadores?)\s+macro(\s|-)?economicas?", nt) or
        re.search(r"\bmacro(\s|-)?economicas?\b", nt) or
        re.search(r"(?<!\w)macro(?!\w)", nt)
    )
    if macro_any and "__ALL__" not in found:
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
        if clist: f["country"] = clist
        else:     f.pop("country", None)

    # ⚠️ clave: si huele a macro, NO infieras categoría desde el texto
    if _extract_macros(text_for_fallback) or _guess_macro(text_for_fallback):
        if not f.get("category"):
            f.pop("category", None)
        return f

    # category → canónica (con fallback semántico SOLO si no es macro)
    cat = f.get("category") or _canonicalize_category(text_for_fallback)
    if cat: f["category"] = cat
    else:   f.pop("category", None)
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

def _sanitize_resp_excerpt(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    # Escapar comillas dobles internas
    s = s.replace('"', '\\"')
    return s

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
        # LASSO: usa el mismo esquema genérico basado en la respuesta RAG
        if intent == "lasso_influence":
            return _viz_prompt_from_generic("lasso_influence", filters, user_prompt=user_prompt, rag_response=rag_response)
        if intent == "lasso_influence_compare":
            return _viz_prompt_from_generic("lasso_influence_compare", filters, user_prompt=user_prompt, rag_response=rag_response)
        if intent == "lasso_var_influence":
            return _viz_prompt_from_generic("lasso_var_influence", filters, user_prompt=user_prompt, rag_response=rag_response)
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

# 1) Cambia la firma:
def remember_session(
    session_id: str,
    *,
    filters: dict,
    intent: str,
    query: str,
    hits: int,
    mentioned: dict | None = None,
):
    last = MEM.get(session_id) or {}
    lastf = dict(last.get("last_filters") or {})
    mnow = mentioned or {}

    # pisa categoría solo si llega valor o se mencionó en este turno
    if filters.get("category"):
        lastf["category"] = filters["category"]
    elif mnow.get("category"):
        lastf["category"] = mnow["category"]

    if "country" in filters and filters.get("country") is not None:
        lastf["country"] = filters["country"]
    if "store" in filters:
        lastf["store"] = filters.get("store")

    MEM.set(session_id, {
        **last,
        "last_filters": lastf,
        "last_intent": intent,
        "last_query": query,
        "last_mentioned": mnow,
        "last_hits": int(hits or 0),
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
    Envuelve el stream del modelo y filtra un '[FIN]' que pudiera emitir,
    y también limpia encabezados/markdown/instrucciones no deseadas.
    """
    import re
    m = model or llm_chat

    # patrones de contenido que NO queremos enviar al cliente
    BAD_LINE = re.compile(
        r"^\s*(#{1,6}\s|[-*]\s|```)|"
        r"(your\s*task|sql\s+query|construct an elaborate email subject|"
        r"use actual data from|do not use hypothetical|BEGIN|END)",
        re.I
    )

    for chunk in m.stream(prompt):
        # asegurar str
        if isinstance(chunk, (bytes, bytearray)):
            s = chunk.decode("utf-8", errors="ignore")
        else:
            s = str(chunk)

        # filtrar un posible [FIN] que venga del modelo
        if s.strip() == "data: [FIN]":
            continue

        # limpiar líneas "problemáticas" (markdown/instruccionales)
        try:
            # el backend ya manda "data: ...\n\n"
            payload = s.split("data:", 1)[1]
        except Exception:
            payload = s

        # si TODA la línea luce mala, la saltamos
        if BAD_LINE.search(payload):
            continue

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

def _cta_options(intent: str, facts: dict) -> list[str]:
    """
    Devuelve SOLO opciones compatibles con el sistema para cada intent.
    No inventes opciones aquí: todo lo listado debe existir.
    """
    var = facts.get("variable") or (facts.get("variables") or [None])[0]
    country = facts.get("country")
    base_var = "esta variable" if var else "una variable"
    loc = f" en {country}" if country else ""

    if intent in ("macro_list", "macro_lookup", "macro_compare", "macro_rank"):
        return [
            "¿Te muestro solo inflación, tasa o dólar?",
            f"¿Quieres comparar {base_var} entre países?",
            f"¿Ver la serie histórica de {base_var}{loc}?",
            "¿Quieres ver el país con el valor más bajo o más alto?",
        ]

    if intent == "list":
        return [
            "¿Te muestro solo los más baratos por tienda?",
            "¿Filtramos por marca o presentación?",
        ]

    if intent == "lookup":
        return [
            "¿Filtramos por tienda o marca?",
            "¿Te muestro los más baratos por tienda?",
        ]

    if intent == "aggregate":
        return [
            "¿Genero una gráfica con estos resultados?",
            "¿Cambio el agrupamiento por país o tienda?",
        ]

    if intent == "compare":
        return [
            "¿Agrego otro país a la comparación?",
            "¿Quieres comparar otra categoría?",
        ]

    # === NUEVO: CTA para influencia LASSO ===
    if intent == "lasso_influence":
        return [
            "¿Quieres comparar la influencia con otra marca o producto?",
            "¿Te muestro la descomposición para otra presentación del producto?",
            "¿Simulamos un what-if sencillo (por ejemplo, +1 pp en inflación)?",
        ]

    return ["¿Quieres que profundice?"]



def _prompt_cta_from_facts(intent: str, facts: dict) -> str:
    """
    Pide al LLM que devuelva UNA sola pregunta elegida EXCLUSIVAMENTE
    de la lista de opciones permitidas para el intent dado.
    """
    import json
    options = _cta_options(intent, facts)
    opts_txt = "\n- ".join(options)
    return (
        "Eres el asistente del SPI. Devuelve UNA sola pregunta (CTA) en español, "
        "sin emojis ni markdown, terminada en '?'. "
        "Debes elegirla EXACTAMENTE de la lista de opciones permitidas; "
        "no reformules, no combines ni inventes nuevas opciones. "
        "Devuelve solo la pregunta.\n"
        f"INTENT: {intent}\n"
        f"FACTS(JSON): {json.dumps(facts, ensure_ascii=False)}\n"
        f"Opciones permitidas:\n- {opts_txt}\n"
        "Pregunta:"
    )


def _gen_cta(intent: str, facts: dict) -> str:
    try:
        cta = (llm_chat.generate(_prompt_cta_from_facts(intent, facts)) or "").strip()
        # fallback muy breve si el modelo se queda en blanco
        return cta if cta else "¿Quieres que profundice?"
    except Exception:
        return "¿Quieres que profundice?"








def _prompt_macro_humano(intent: str, facts: dict, hint_cta: str, include_cta: bool = True) -> str:
    import json
    base = (
        "Eres el asistente del SPI. Español, tono profesional y natural. "
        "Escribe en 3–6 frases de TEXTO PLANO: sin markdown, sin títulos, sin viñetas ni bloques de código. "
        "Prohibido subtítulos ('###'), 'Your task:', SQL o instrucciones meta. "
        "NO muestres el JSON ni nombres de campos. "
        "Estructura: 1) saludo breve; 2) contexto; 3) respuesta con cifras.\n"
        f"FACTS(JSON): {json.dumps(facts, ensure_ascii=False)}\n"
    )
    if include_cta:
        base += f"Termina con un único CTA: {hint_cta}\n"
    else:
        base += "No incluyas CTA todavía; termina después de la explicación.\n"
    return base + "Responde solo el texto final, sin JSON ni markdown."




def _prompt_lasso_humano(facts: dict, hint_cta: str | None = None, include_cta: bool = True) -> str:
    import json
    base = (
        "Eres el asistente del SPI. Escribe en español, tono cercano y profesional. "
        "Devuelve 3–6 frases de TEXTO PLANO (sin markdown, sin viñetas). "
        "Estructura: 1) saludo amable; 2) contexto breve: explica que LASSO estima y cuáles fueron los valores de los coeficientes; "
        "3) respuesta: menciona las variables con mayor magnitud, su valor y su signo (sube/baja el precio); "
        "4) cierra con R², alfa y el número de observaciones, citando la descripción del producto y el retail. "
        "REGLA CRÍTICA: Usa ÚNICAMENTE los datos provistos en HECHOS(JSON). No consultes ni asumas información adicional. "
        "Si algún dato no aparece en HECHOS, indica 'no disponible'. No inventes cifras ni contexto externo."
    )
    if include_cta and hint_cta:
        base += f" Termina con un único CTA: {hint_cta}"
    else:
        base += " No incluyas CTA."
    return base + f"\nHECHOS(JSON): {json.dumps(facts, ensure_ascii=False)}\nRespuesta:"


# --- NUEVO: Prompt para comparación LASSO entre dos productos/marcas ---
def _prompt_lasso_compare(facts: dict, hint_cta: str | None = None, include_cta: bool = True) -> str:
    import json
    base = (
        "Eres el asistente del SPI. Escribe en español, tono cercano y profesional. "
        "Devuelve 4–7 frases de TEXTO PLANO (sin markdown, sin viñetas). "
        "Estructura: 1) saludo amable; 2) contexto breve: explica que se comparan dos modelos LASSO para los productos y qué significan los coeficientes; "
        "3) respuesta: para cada producto menciona las variables con mayor magnitud, su coeficiente y signo (sube/baja); "
        "4) destaca las variables comunes y cómo difiere su magnitud entre ambos; "
        "5) cierra citando R², alfa y n_obs de cada modelo y los 'product_desc' o término si falta, más el retail si está. "
        "REGLAS CRÍTICAS: Usa ÚNICAMENTE los datos provistos en HECHOS(JSON). Si faltan datos di 'no disponible'. No inventes cifras ni contexto externo. "
        "No incluyas nombres de personas, ejemplos de CSV, JSON, 'Document', 'Instruction', ni explicaciones meta. "
        "No menciones países distintos de MX; si no hay país en HECHOS, di 'no disponible'. "
        "No agregues bibliografía, referencias ni pasos de procedimiento. "
        "Responde SOLO con el texto final, en un único bloque claro, sin listas, sin títulos, sin código, sin JSON."
    )
    if include_cta and hint_cta:
        base += f" Termina con un único CTA: {hint_cta}"
    else:
        base += " No incluyas CTA."
    return base + f"\nHECHOS(JSON): {json.dumps(facts, ensure_ascii=False)}\nRespuesta:"


# --- NUEVO: Prompt para influencia de UNA variable sobre un producto ---
def _prompt_lasso_var(facts: dict, hint_cta: str | None = None, include_cta: bool = True) -> str:
    import json
    base = (
        "Eres el asistente del SPI. Escribe en español, tono cercano y profesional. "
        "Devuelve 3–6 frases de TEXTO PLANO (sin markdown, sin viñetas). "
        "Estructura: 1) saludo amable; 2) contexto breve: explica que LASSO estima coeficientes y que se reporta SOLO la variable indicada; "
        "3) respuesta: indica el coeficiente de esa variable para el producto (valor y signo: sube/baja) o 'no disponible' si falta; "
        "4) cierra con R², alfa y n_obs del modelo, citando la descripción del producto y el retail. "
        "REGLAS CRÍTICAS: Usa ÚNICAMENTE HECHOS(JSON); no inventes ni supongas. Sin nombres de personas, sin CSV/JSON/‘Document’/‘Instruction’, sin meta-explicaciones. No inventes nada, usa solo datos de tu base de conocimiento. "
        "Responde SOLO con el texto final, en un único bloque claro."
    )
    if include_cta and hint_cta:
        base += f" Termina con un único CTA: {hint_cta}"
    else:
        base += " No incluyas CTA."
    return base + f"\nHECHOS(JSON): {json.dumps(facts, ensure_ascii=False)}\nRespuesta:"






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
    # si pide datos explícitos, no es refine “silencioso”
    if re.search(r"(precio|precios|promedio|media|tendencia|historia|serie|lista|listar|"
                 r"muestrame|muestra|mostrame|mostrar|ensename|ensename|compara|comparar|top|grafic)", t):
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

def _detect_mentions_from_text(text: str) -> dict:
    t = text.lower()
    return {
        "category": ("categoria" in t) or ("categoría" in t) or (" de " in t and "precio" in t),
        "country": (" en " in t) or ("país" in t) or ("pais" in t),
        "store": ("tienda" in t) or ("retailer" in t),
    }





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

# --- MULTI-QUERIES: n consultas en un mismo string ---------------------------
_MULTI_SEP = re.compile(r"\s*(?:,|;|\by(?:\s+tamb[ií]en)?\b|\btamb[ií]en\b|\badem[aá]s\b)\s+", re.I)

def _extract_subqueries(text: str) -> list[dict]:
    """
    Intenta extraer múltiples consultas homogéneas del mismo string.
    Soporta patrones sencillos del tipo:
      - 'costo de vida en Brasil' / 'IPC Colombia'
      - 'precio del arroz en Colombia' / 'café en Brasil'
    Devuelve una lista ordenada de dicts como:
      {'type':'macro','var':'cpi','countries':['BR']}
      {'type':'product','filters':{'category':'arroz','country':'CO'}}
    """
    out: list[dict] = []
    if not text:
        return out
    parts = _MULTI_SEP.split(text)

    for raw in parts:
        s = raw.strip()
        if not s:
            continue
        nt = _norm(s)

        # --- macro? (usa alias ya definidos en NMACROS)
        macs = [NMACROS[a] for a in NMACROS.keys() if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", nt)]
        countries = _extract_countries(s, max_n=3)
        if macs:
            # prioriza una variable concreta si aparece, si no deja la primera
            var = next((m for m in macs if m != "__ALL__"), macs[0])
            out.append({"type": "macro", "var": var, "countries": countries[:2] or []})
            continue

        # --- productos? (alias → categoría canónica con NCATEGORIES)
        cat = None
        for alias, canon in NCATEGORIES.items():
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nt):
                cat = canon
                break
        if cat:
            ctys = _extract_countries(s, max_n=2)
            out.append({"type": "product", "filters": {"category": cat, "country": ctys[0] if ctys else None}})
            continue

    return [q for q in out if q]





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
    def _diagnose_no_results(intent: str, *, plan=None, text:str="", macros=None, rows=None, hits=None, agg=None, filters: dict | None = None) -> str:
        f = (filters or (plan.filters if plan else {}) or {})
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

    if _is_refine_command(text) and not (_extract_macros(text) or _guess_macro(text)):
        f_now = _guess_filters(text)
        if not any(f_now.values()):
            reason = "no identifiqué qué filtro cambiar (país/categoría/tienda). ¿Cuál ajusto?"
            return StreamingResponse(_sse_no_data_ex(reason, None), media_type="text/event-stream")

        # fusiona con lo último que tengamos
        last = MEM.get(req.session_id) or {}
        lastf = (last.get("last_filters") or {})
        new_filters = _normalize_plan_filters({**lastf, **f_now}, text)


        # guarda la memoria
        if req.session_id:
            remember_session(
                req.session_id,
                filters=new_filters,
                intent="refine",
                query=text,
                hits=0,
                mentioned=_detect_mentions_from_text(text),
            )

        # responde breve y CIERRA con [FIN]
        def gen_refine():
            yield f"data: ¡Listo! Actualicé los filtros a → {new_filters}\n\n"
            yield "data: ¿Consulto el promedio, comparo países o te muestro los más baratos?\n\n"
            yield "data: [FIN]\n\n"
        return StreamingResponse(gen_refine(), media_type="text/event-stream")

    # --- DETECCIÓN TEMPRANA: comparación LASSO de dos productos/marcas ---
    try:
        early_lasso_cmp = detect_lasso_influence_compare_intent(text)
    except Exception:
        early_lasso_cmp = None
    if early_lasso_cmp and early_lasso_cmp.get("intent") == "lasso_influence_compare":
        terms = early_lasso_cmp.get("terms") or []
        if len(terms) != 2:
            pass  # cae al flujo normal si falla
        else:
            term_a, term_b = terms
            rows_a = query_lasso_models("product", term_a, topk=5) or []
            if not rows_a:
                rows_a = query_lasso_models("brand", term_a, topk=5) or []
            rows_b = query_lasso_models("product", term_b, topk=5) or []
            if not rows_b:
                rows_b = query_lasso_models("brand", term_b, topk=5) or []

            if (not rows_a) or (not rows_b):
                def gen_missing_cmp():
                    yield "data: Filtros → país: MX | comparación LASSO\n\n"
                    if not rows_a and not rows_b:
                        yield "data: No encontré modelos LASSO para ninguno de los dos términos. ¿Intento con otros productos?\n\n"
                    elif not rows_a:
                        yield f"data: No encontré modelos LASSO para '{term_a}'. ¿Intento con otro producto para comparar con {term_b}?\n\n"
                    else:
                        yield f"data: No encontré modelos LASSO para '{term_b}'. ¿Intento con otro producto para comparar con {term_a}?\n\n"
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_missing_cmp(), media_type="text/event-stream")

            def _extract_coefs_cmp(row: dict) -> list[dict]:
                mapping = [
                    ("coef_inflation_rate_pct_change", "Inflación general (%)"),
                    ("coef_cambio_dolar_pct_change", "Tipo de cambio USD/MXN (%)"),
                    ("coef_cpi_pct_change", "CPI / IPC (%)"),
                    ("coef_interest_rate_pct_change", "Tasa de interés (%)"),
                    ("coef_gdp_pct_change", "PIB (%)"),
                    ("coef_producer_prices_pct_change", "Precios al productor (%)"),
                    ("coef_gini_pct_change", "Índice Gini (%)"),
                ]
                out = []
                for k, label in mapping:
                    v = row.get(k)
                    if v is None:
                        continue
                    try:
                        out.append({"name": label, "value": float(v)})
                    except Exception:
                        continue
                out.sort(key=lambda c: abs(c["value"]), reverse=True)
                return out

            best_a = max(rows_a, key=lambda r: (r.get("r_squared") or 0.0))
            best_b = max(rows_b, key=lambda r: (r.get("r_squared") or 0.0))
            coefs_a = _extract_coefs_cmp(best_a)
            coefs_b = _extract_coefs_cmp(best_b)
            map_a = {c["name"]: c for c in coefs_a}
            map_b = {c["name"]: c for c in coefs_b}
            shared = []
            for k in sorted(set(map_a) & set(map_b)):
                shared.append({
                    "name": k,
                    "value_a": map_a[k]["value"],
                    "value_b": map_b[k]["value"],
                    "sign_a": "sube" if map_a[k]["value"] > 0 else "baja",
                    "sign_b": "sube" if map_b[k]["value"] > 0 else "baja",
                })

            facts_cmp = {
                "type": "lasso_influence_compare",
                "country": "MX",
                "terms": terms,
                "models": [
                    {
                        "term": term_a,
                        "product_desc": best_a.get("nombre") or best_a.get("producto") or term_a,
                        "brand": best_a.get("marca"),
                        "retail": best_a.get("retail"),
                        "r2": float(best_a.get("r_squared") or 0.0),
                        "alpha": float(best_a.get("best_alpha") or 0.0),
                        "n_obs": int(best_a.get("n_obs") or 0),
                        "coefs": coefs_a,
                    },
                    {
                        "term": term_b,
                        "product_desc": best_b.get("nombre") or best_b.get("producto") or term_b,
                        "brand": best_b.get("marca"),
                        "retail": best_b.get("retail"),
                        "r2": float(best_b.get("r_squared") or 0.0),
                        "alpha": float(best_b.get("best_alpha") or 0.0),
                        "n_obs": int(best_b.get("n_obs") or 0),
                        "coefs": coefs_b,
                    },
                ],
                "shared_coefs": shared,
            }
            cta_cmp = "¿Te muestro otro par de productos o profundizamos en uno solo?"
            def gen_cmp_early():
                yield "data: Filtros → país: MX | comparación LASSO\n\n"
                prompt = _prompt_lasso_compare(facts_cmp, hint_cta=cta_cmp, include_cta=True)
                rag_buf: list[str] = []
                for chunk in _stream_no_fin(prompt):
                    try:
                        content = chunk.split("data:", 1)[1].strip()
                    except Exception:
                        content = chunk
                    if content and content != "[FIN]":
                        rag_buf.append(content)
                    yield chunk
                yield "data: \n\n"
                try:
                    full_resp = " ".join(rag_buf).strip()
                    vizp = _maybe_viz_prompt(
                        "lasso_influence_compare",
                        {"country": "MX"},
                        user_prompt=text,
                        rag_response=full_resp,
                    )
                except Exception:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"
                yield "data: [FIN]\n\n"

            try:
                remember_session(
                    req.session_id,
                    filters={"country": "MX"},
                    intent="lasso_influence_compare",
                    query=text,
                    hits=len(shared),
                    mentioned={"country": True},
                )
            except Exception:
                pass
            return StreamingResponse(gen_cmp_early(), media_type="text/event-stream")

    # --- DETECCIÓN TEMPRANA: influencia de UNA variable en un producto/marca ---
    try:
        early_lasso_var = detect_lasso_variable_influence_intent(text)
    except Exception:
        early_lasso_var = None
    if early_lasso_var and early_lasso_var.get("intent") == "lasso_var_influence":
        var_key = early_lasso_var.get("var_key")
        var_label = early_lasso_var.get("var_label")
        term = early_lasso_var.get("term")
        # Busca modelo por producto; fallback a marca
        rows = query_lasso_models("product", term, topk=5) or []
        if not rows:
            rows = query_lasso_models("brand", term, topk=5) or []
        if not rows:
            def gen_empty_var():
                yield "data: Filtros → país: MX | variable: {var_label} | producto: {term}\n\n".format(var_label=var_label, term=term)
                yield "data: No encontré modelos LASSO para ese producto o marca en México. ¿Intento con otro?\n\n"
                yield "data: [FIN]\n\n"
            return StreamingResponse(gen_empty_var(), media_type="text/event-stream")

        best = max(rows, key=lambda r: (r.get("r_squared") or 0.0))
        coef_val = best.get(var_key)
        coef_val_f = float(coef_val) if coef_val is not None else None
        facts_var = {
            "type": "lasso_var_influence",
            "country": "MX",
            "term": term,
            "var_key": var_key,
            "var_label": var_label,
            "product_desc": best.get("nombre") or best.get("producto") or term,
            "brand": best.get("marca"),
            "retail": best.get("retail"),
            "r2": float(best.get("r_squared") or 0.0),
            "alpha": float(best.get("best_alpha") or 0.0),
            "n_obs": int(best.get("n_obs") or 0),
            "coef": None if coef_val_f is None else {
                "name": var_label,
                "value": coef_val_f,
                "sign": ("sube" if coef_val_f > 0 else "baja")
            }
        }
        cta_var = "¿Quieres ver otra variable o comparar con otro producto?"

        def gen_var():
            yield f"data: Filtros → país: MX | variable: {var_label} | producto: {term}\n\n"
            prompt = _prompt_lasso_var(facts_var, hint_cta=cta_var, include_cta=True)
            rag_buf: list[str] = []
            for chunk in _stream_no_fin(prompt):
                try:
                    content = chunk.split("data:", 1)[1].strip()
                except Exception:
                    content = chunk
                if content and content != "[FIN]":
                    rag_buf.append(content)
                yield chunk
            yield "data: \n\n"
            try:
                full_resp = " ".join(rag_buf).strip()
                vizp = _maybe_viz_prompt(
                    "lasso_var_influence",
                    {"country": "MX"},
                    user_prompt=text,
                    rag_response=full_resp,
                )
            except Exception:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"
            yield "data: [FIN]\n\n"

        try:
            remember_session(
                req.session_id,
                filters={"country": "MX"},
                intent="lasso_var_influence",
                query=text,
                hits=1 if coef_val_f is not None else 0,
                mentioned={"country": True},
            )
        except Exception:
            pass
        return StreamingResponse(gen_var(), media_type="text/event-stream")



        # --- N consultas en un mismo string (separadas por ',' 'y' 'también') ---
    subs = _extract_subqueries(text)
    if len(subs) >= 2:

        def gen_multi():
            for i, q in enumerate(subs, 1):
                # ======================
                #   1) MACRO
                # ======================
                if q["type"] == "macro":
                    var = q.get("var")
                    cs  = q.get("countries") or []

                    # -- Comparación multi-país
                    if var and len(cs) >= 2:
                        rows = macro_compare(var, cs) or []

                        yield f"data: [MACRO] variable: {var} | países: {' | '.join(cs)}\n\n"
                        for j, r in enumerate((rows or [])[:5], 1):
                            yield f"data: {j}. {r.get('country')}: {r.get('value')} {r.get('unit') or ''} ({r.get('date') or ''})\n\n"
                        # Redacción humana (macro_compare)
                        facts = {
                            "type": "macro_compare",
                            "variable": var,
                            "countries": cs,
                            "rows": [
                                {"country": x.get("country"), "name": x.get("name"),
                                 "value": x.get("value"), "unit": x.get("unit"),
                                 "date": x.get("date")}
                                for x in (rows or [])
                            ]
                        }
                        sub_text = f"compara {var} entre {', '.join(cs)} — resume en 2–3 líneas."
                        prompt = _prompt_macro_humano("macro_compare", facts, "¿Agregar otro país o variable?")
                        for chunk in _stream_no_fin(prompt):
                            yield chunk
                        yield "data: \n\n"

                        # Mini tabla factual (máx 5)
                        for j, r in enumerate(rows[:5], 1):
                            yield f"data: {j}. {r.get('country')}: {r.get('value')} {r.get('unit') or ''} ({r.get('date') or ''})\n\n"
                        yield "data: \n\n"
                        continue

                    # -- Lookup 1 país
                    if var and len(cs) == 1:
                        r = macro_lookup(var, cs[0])
                        if r:
                            yield f"data: [MACRO] País: {cs[0]} | variable: {var}\n\n"
                            yield f"data: {_fmt_macro_row(r)}\n\n"
                            facts = {
                                "type": "macro_lookup",
                                "variable": var,
                                "country": cs[0],
                                "row": {"country": r.get("country"), "name": r.get("name"),
                                        "value": r.get("value"), "unit": r.get("unit"),
                                        "date": r.get("date")}
                            }
                            prompt = _prompt_macro_humano("macro_lookup", facts, "¿Ver serie temporal o comparar con otro país?")
                            for chunk in _stream_no_fin(prompt):
                                yield chunk
                            yield "data: \n\n"

                        else:
                            yield f"data: No hallé datos macro para {var} en {cs[0]}.\n\n"
                        yield "data: \n\n"
                        continue

                    yield "data: Para variables macro necesito el país (ej.: 'IPC en CO').\n\n"
                    yield "data: \n\n"
                    continue

                # ======================
                #   2) PRODUCTOS (lookup)
                # ======================
                if q["type"] == "product":
                    from collections import Counter
                    f = {k: v for k, v in (q.get('filters') or {}).items() if v}
                    rows = list_by_filter(f, limit=min(getattr(S, 'chat_list_default', 500), 40)) or []

                    yield f"data: [PRODUCTOS] Filtros → {f}\n\n"
                    # --- Construcción de FACTS (como en lookup) ---
                    facts_filters = dict(f)
                    hits = rows[:min(20, len(rows))]
                    ctx = _build_ctx(hits, 10)

                    base_filters = dict(facts_filters)
                    base_filters.pop("store", None)  # promedio nacional sin tienda

                    rows_all = list_by_filter(base_filters, limit=min(getattr(S, "aggregate_limit", 5000), 5000))
                    prices = [r.get("price") for r in rows_all if r.get("price") is not None]

                    cur = None
                    if rows_all:
                        cur = Counter([r.get("currency") for r in rows_all if r.get("currency")]).most_common(1)[0][0]

                    avg_all = (sum(prices) / max(len(prices), 1)) if prices else None

                    # Promedios por marca
                    agg_brand = aggregate_prices(base_filters, by="brand")
                    groups = agg_brand.get("groups") or []
                    groups = [g for g in groups if g and g.get("group") not in (None, "", "N/A") and g.get("avg") is not None]

                    # Fallback a partir de hits si no hubo grupos en BD
                    if not groups and hits:
                        tmp_sum, tmp_n = {}, {}
                        for h in hits:
                            b, p = (h.get("brand") or "N/A"), h.get("price")
                            if p is None: continue
                            tmp_sum[b] = tmp_sum.get(b, 0) + p
                            tmp_n[b]   = tmp_n.get(b, 0) + 1
                        groups = [{"group": b, "avg": tmp_sum[b] / tmp_n[b], "count": tmp_n[b]} for b in tmp_sum.keys()]

                    brands = sorted(
                        [{"brand": g["group"], "avg": g["avg"], "count": g.get("count") or g.get("n") or 0} for g in groups],
                        key=lambda x: -x["count"]
                    )[:10]

                    brand_range = None
                    if brands:
                        lo = min(brands, key=lambda b: b["avg"])
                        hi = max(brands, key=lambda b: b["avg"])
                        brand_range = {"min_brand": lo["brand"], "min_avg": lo["avg"],
                                       "max_brand": hi["brand"], "max_avg": hi["avg"]}

                    facts = {
                        "country": f.get("country"),
                        "category": f.get("category"),
                        "currency": cur,
                        "national_avg": float(avg_all) if avg_all is not None else None,
                        "n": len(prices),
                        "brands": brands,
                        "brand_range": brand_range,
                    }

                    # Writer humano + VIZ_PROMPT después de tener facts
                    try:
                        prompt = _prompt_lookup_from_facts(text, facts, ctx)
                        for chunk in _stream_no_fin(prompt):
                            yield chunk
                        yield "data: \n\n"
                    except Exception:
                        yield "data: (error interno en writer de productos multi-subconsulta)\n\n"
                    try:
                        vizp = _maybe_viz_prompt("lookup", f, rows=rows)
                        if vizp:
                            yield f"data: [VIZ_PROMPT] {vizp}\n\n"
                    except Exception:
                        pass
                    for j, r in enumerate(rows[:10], 1):
                        yield f"data: {_fmt_row(r, j)}\n\n"

                    # Memoria mínima por subconsulta (sin CTA adicional para evitar duplicados)
                    try:
                        remember_session(req.session_id, filters=f, intent="list",
                                         query=text, hits=len(rows),
                                         mentioned=_detect_mentions_from_text(text))
                    except Exception:
                        pass

                    yield "data: \n\n"

            # Cierre único para todo el combo
            yield "data: [FIN]\n\n"

        return StreamingResponse(gen_multi(), media_type="text/event-stream")


    # --- LASSO: influencia/descomposición solo para México ---
    # --- LASSO: influencia/descomposición solo para México ---
    lasso_plan = detect_lasso_influence_intent(text)
    # --- NUEVO: Comparación LASSO entre dos productos ---
    lasso_cmp_plan = detect_lasso_influence_compare_intent(text)
    if lasso_cmp_plan and lasso_cmp_plan.get("intent") == "lasso_influence_compare":
        terms = lasso_cmp_plan.get("terms") or []
        if len(terms) != 2:
            # fallback si la detección fue incorrecta
            pass
        else:
            term_a, term_b = terms
            # Intentamos por producto primero; si no, por marca
            rows_a = query_lasso_models("product", term_a, topk=5)
            if not rows_a:
                rows_a = query_lasso_models("brand", term_a, topk=5)
            rows_b = query_lasso_models("product", term_b, topk=5)
            if not rows_b:
                rows_b = query_lasso_models("brand", term_b, topk=5)

            if (not rows_a) or (not rows_b):
                def gen_missing():
                    yield "data: Filtros → país: MX | comparación LASSO\n\n"
                    if not rows_a and not rows_b:
                        yield ("data: No encontré modelos LASSO para ninguno de los dos términos. ¿Intento con otros productos?\n\n")
                    elif not rows_a:
                        yield (f"data: No encontré modelos LASSO para '{term_a}'. ¿Intento con otro producto para comparar con {term_b}?\n\n")
                    else:
                        yield (f"data: No encontré modelos LASSO para '{term_b}'. ¿Intento con otro producto para comparar con {term_a}?\n\n")
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_missing(), media_type="text/event-stream")

            # Elegimos mejor modelo de cada lado
            best_a = max(rows_a, key=lambda r: (r.get("r_squared") or 0.0))
            best_b = max(rows_b, key=lambda r: (r.get("r_squared") or 0.0))

            def _extract_coefs(row: dict) -> list[dict]:
                mapping = [
                    ("coef_inflation_rate_pct_change", "Inflación general (%)"),
                    ("coef_cambio_dolar_pct_change", "Tipo de cambio USD/MXN (%)"),
                    ("coef_cpi_pct_change", "CPI / IPC (%)"),
                    ("coef_interest_rate_pct_change", "Tasa de interés (%)"),
                    ("coef_gdp_pct_change", "PIB (%)"),
                    ("coef_producer_prices_pct_change", "Precios al productor (%)"),
                    ("coef_gini_pct_change", "Índice Gini (%)"),
                ]
                out = []
                for key, label in mapping:
                    v = row.get(key)
                    if v is not None:
                        try:
                            v_f = float(v)
                        except Exception:
                            continue
                        out.append({"name": label, "value": v_f})
                out.sort(key=lambda c: abs(c["value"]), reverse=True)
                return out

            coefs_a = _extract_coefs(best_a)
            coefs_b = _extract_coefs(best_b)

            # Variables comunes (por nombre) presentes en ambos
            map_a = {c["name"]: c for c in coefs_a}
            map_b = {c["name"]: c for c in coefs_b}
            shared = []
            for k in sorted(set(map_a.keys()) & set(map_b.keys())):
                shared.append({
                    "name": k,
                    "value_a": map_a[k]["value"],
                    "value_b": map_b[k]["value"],
                    "sign_a": "sube" if map_a[k]["value"] > 0 else "baja",
                    "sign_b": "sube" if map_b[k]["value"] > 0 else "baja",
                })

            facts_cmp = {
                "type": "lasso_influence_compare",
                "country": "MX",
                "terms": terms,
                "models": [
                    {
                        "term": term_a,
                        "product_desc": best_a.get("nombre") or best_a.get("producto") or term_a,
                        "brand": best_a.get("marca"),
                        "retail": best_a.get("retail"),
                        "r2": float(best_a.get("r_squared") or 0.0),
                        "alpha": float(best_a.get("best_alpha") or 0.0),
                        "n_obs": int(best_a.get("n_obs") or 0),
                        "coefs": coefs_a,
                    },
                    {
                        "term": term_b,
                        "product_desc": best_b.get("nombre") or best_b.get("producto") or term_b,
                        "brand": best_b.get("marca"),
                        "retail": best_b.get("retail"),
                        "r2": float(best_b.get("r_squared") or 0.0),
                        "alpha": float(best_b.get("best_alpha") or 0.0),
                        "n_obs": int(best_b.get("n_obs") or 0),
                        "coefs": coefs_b,
                    },
                ],
                "shared_coefs": shared,
            }

            # CTA simple (sin usar _gen_cta para evitar agregar lógica nueva):
            cta_cmp = "¿Te muestro otro par de productos o profundizamos en uno solo?"

            def gen_cmp_lasso():
                yield "data: Filtros → país: MX | comparación LASSO\n\n"
                prompt = _prompt_lasso_compare(facts_cmp, hint_cta=cta_cmp, include_cta=True)
                rag_buf = []
                for chunk in _stream_no_fin(prompt):
                    try:
                        content = chunk.split("data:", 1)[1].strip()
                    except Exception:
                        content = chunk
                    if content and content != "[FIN]":
                        rag_buf.append(content)
                    yield chunk
                yield "data: \n\n"
                # VIZ_PROMPT basado en la respuesta RAG comparativa
                try:
                    full_resp = " ".join(rag_buf).strip()
                    vizp = _maybe_viz_prompt(
                        "lasso_influence_compare",
                        {"country": "MX"},
                        user_prompt=text,
                        rag_response=full_resp,
                    )
                except Exception:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"
                yield "data: [FIN]\n\n"

            # Memoria de sesión mínima
            try:
                remember_session(
                    req.session_id,
                    filters={"country": "MX"},
                    intent="lasso_influence_compare",
                    query=text,
                    hits=len(shared),
                    mentioned={"country": True},
                )
            except Exception:
                pass

            return StreamingResponse(gen_cmp_lasso(), media_type="text/event-stream")
    if lasso_plan:
        if lasso_plan.get("intent") == "lasso_influence_blocked_non_mx":
            def gen_block():
                yield "data: Hola. La descomposición por coeficientes LASSO solo está disponible para México por ahora.\n\n"
                yield "data: ¿Busco la influencia para ese producto o marca en México?\n\n"
                yield "data: [FIN]\n\n"
            return StreamingResponse(gen_block(), media_type="text/event-stream")

        # 1) Consulta a Milvus
        rows = query_lasso_models(lasso_plan["by"], lasso_plan["term"], topk=5)

        # 2) Sin filas → mensaje claro
        if not rows:
            def gen_empty():
                yield "data: Filtros → país: MX | categoría: - | tienda: -\n\n"
                yield ("data: Hola. La descomposición LASSO está disponible para México, "
                    f"pero no encontré esa {('marca' if lasso_plan['by']=='brand' else 'presentación')} "
                    "en los modelos. ¿Quieres que pruebe con otra marca o producto?\n\n")
                yield "data: [FIN]\n\n"
            return StreamingResponse(gen_empty(), media_type="text/event-stream")

        # 3) Tomamos el mejor por R² y armamos HECHOS para el writer
        best = max(rows, key=lambda r: (r.get('r_squared') or 0.0))
        coefs = [
            {"name": "Inflación general (%)",          "value": best.get("coef_inflation_rate_pct_change")},
            {"name": "Tipo de cambio USD/MXN (%)",     "value": best.get("coef_cambio_dolar_pct_change")},
            {"name": "CPI / IPC (%)",                  "value": best.get("coef_cpi_pct_change")},
            {"name": "Tasa de interés (%)",            "value": best.get("coef_interest_rate_pct_change")},
            {"name": "PIB (%)",                        "value": best.get("coef_gdp_pct_change")},
            {"name": "Precios al productor (%)",       "value": best.get("coef_producer_prices_pct_change")},
            {"name": "Índice Gini (%)",                "value": best.get("coef_gini_pct_change")},
        ]
        coefs = [c for c in coefs if c["value"] is not None]
        # ordena por magnitud
        coefs.sort(key=lambda c: abs(float(c["value"] or 0.0)), reverse=True)

        facts_lasso = {
            "type": "lasso_influence",
            "by": lasso_plan["by"],
            "term": lasso_plan["term"],
            "country": "MX",
            "product_desc": best.get("nombre"),
            "brand": best.get("marca"),
            "product": best.get("producto"),
            "retail": best.get("retail"),
            "r2": float(best.get("r_squared") or 0.0),
            "alpha": float(best.get("best_alpha") or 0.0),
            "n_obs": int(best.get("n_obs") or 0),
            "coefs": [{"name": c["name"], "value": float(c["value"])} for c in coefs],
        }

        # 4) CTA generado por LLM (opciones válidas para 'lasso_influence')
        cta_lasso = _gen_cta("lasso_influence", facts_lasso)

        def gen_lasso():
            # Encabezado de filtros consistente con el front
            yield "data: Filtros → país: MX | categoría: - | tienda: -\n\n"

            # Writer humano (saludo + contexto LASSO + respuesta). SIN CTA aquí:
            prompt = _prompt_lasso_humano(facts_lasso, hint_cta=None, include_cta=False)
            rag_buf = []
            for chunk in _stream_no_fin(prompt):
                try:
                    content = chunk.split("data:", 1)[1].strip()
                except Exception:
                    content = chunk
                if content and content != "[FIN]":
                    rag_buf.append(content)
                yield chunk
            yield "data: \n\n"

            # VIZ_PROMPT (basado en la respuesta RAG de LASSO)
            try:
                full_resp = " ".join(rag_buf).strip()
                vizp = _maybe_viz_prompt(
                    "lasso_influence",
                    {"country": "MX"},
                    user_prompt=text,
                    rag_response=full_resp,
                )
            except Exception:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            # Etiqueta de “Descripción del producto” (antes decía 'Modelo')
            base_desc = best.get("nombre") or best.get("producto") or best.get("marca") or "-"
            yield f"data: Descripción del producto base: {base_desc}\n\n"

            # CTA final (1 sola pregunta)
            yield f"data: {cta_lasso}\n\n"
            yield "data: [FIN]\n\n"

        # Memoria mínima de sesión
        try:
            remember_session(req.session_id,
                            filters={"country": "MX"},
                            intent="lasso_influence",
                            query=text,
                            hits=len(rows),
                            mentioned={"country": True})
        except Exception:
            pass

        return StreamingResponse(gen_lasso(), media_type="text/event-stream")






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
                rag_buf = []
                for chunk in _stream_no_fin(prompt):
                    try:
                        content = chunk.split("data:", 1)[1].strip()
                    except Exception:
                        content = chunk
                    if content and content != "[FIN]":
                        rag_buf.append(content)
                    yield chunk
                yield "data: \n\n"

                # Lista compacta
                topn = 5
                titulo = "Top 5 más altos" if superl == "max" else "Top 5 más bajos"
                yield f"data: {titulo}:\n\n"
                for i, r in enumerate(ranked[:topn], start=1):
                    yield f"data: {i}. {r['country']}: {r['value']} {r.get('unit') or ''} ({r.get('date') or ''})\n\n"

                # VIZ_PROMPT (macro rank/topN)
                try:
                    full_resp = " ".join(rag_buf).strip()
                    vizp = _maybe_viz_prompt(
                        "topn",
                        {"country": cs},
                        rows=ranked[:topn],
                        user_prompt=text,
                        rag_response=full_resp,
                    )
                except Exception:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                # CTA final (coherente con otros flujos macro)
                yield f"data: {_gen_cta('macro_rank', facts)}\n\n"
                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_super(), media_type="text/event-stream")


        # ¿hay intención de productos en este mismo turno?
        heur_now = _guess_filters(text)  # país/categoría/tienda detectados por alias
        has_products = bool(heur_now.get("category")) or bool(re.search(r"\bprecio|precios\b", _norm(text)))

        # Si hay macros + productos => MIXED
        if has_products and countries:
            # <-- construir 'mentioned' temprano (aquí sí existe)
            mentioned_early = {
                "category": "category" in heur_now,
                "country":  "country"  in heur_now,
                "store":    "store"    in heur_now,
            }
            # filtros de productos para este turno
            pf = {k: v for k, v in (heur_now or {}).items()
                if k in ("country", "category", "store") and v}

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
                            prompt = _prompt_macro_humano("macro_list", facts, " ", include_cta=False)
                            rag_buf = []
                            for chunk in _stream_no_fin(prompt):
                                try:
                                    content = chunk.split("data:", 1)[1].strip()
                                except Exception:
                                    content = chunk
                                if content and content != "[FIN]":
                                    rag_buf.append(content)
                                yield chunk
                            yield "data: \n\n"

                            # VIZ_PROMPT (lista de variables del país)
                            try:
                                full_resp = " ".join(rag_buf).strip()
                                vizp = _maybe_viz_prompt(
                                    "list",
                                    {"country": countries[0]},
                                    rows=rows[:10],
                                    user_prompt=text,
                                    rag_response=full_resp,
                                )
                            except Exception:
                                vizp = None
                            if vizp:
                                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    elif len(countries) >= 2:
                        rows = macro_compare(m, countries) or []
                        if rows:
                            yield f"data: [MACRO] {m} | países: {' | '.join(countries)}\n\n"
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
                            prompt = _prompt_macro_humano("macro_compare", facts, " ", include_cta=False)
                            rag_buf = []
                            for chunk in _stream_no_fin(prompt):
                                try:
                                    content = chunk.split("data:", 1)[1].strip()
                                except Exception:
                                    content = chunk
                                if content and content != "[FIN]":
                                    rag_buf.append(content)
                                yield chunk
                            yield "data: \n\n"

                            # VIZ_PROMPT (comparación de variable entre países)
                            try:
                                full_resp = " ".join(rag_buf).strip()
                                vizp = _maybe_viz_prompt(
                                    "compare",
                                    {"country": countries},
                                    rows=rows,
                                    user_prompt=text,
                                    rag_response=full_resp,
                                )
                            except Exception:
                                vizp = None
                            if vizp:
                                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    else:
                        r = macro_lookup(m, countries[0]) if len(countries) == 1 else None
                        if r:
                            yield f"data: [MACRO] País: {countries[0]} | variable: {m}\n\n"
                            facts = {
                                "type": "macro_lookup",
                                "variable": m,
                                "country": countries[0],
                                "value": (r or {}).get("value"),
                                "unit": (r or {}).get("unit"),
                                "date": (r or {}).get("date"),
                                "name": (r or {}).get("name"),
                            }
                            prompt = _prompt_macro_humano("macro_lookup", facts, " ", include_cta=False)
                            rag_buf = []
                            for chunk in _stream_no_fin(prompt):
                                try:
                                    content = chunk.split("data:", 1)[1].strip()
                                except Exception:
                                    content = chunk
                                if content and content != "[FIN]":
                                    rag_buf.append(content)
                                yield chunk
                            yield "data: \n\n"

                            # VIZ_PROMPT (lookup de variable en un país)
                            try:
                                full_resp = " ".join(rag_buf).strip()
                                vizp = _maybe_viz_prompt(
                                    "lookup",
                                    {"country": countries[0]},
                                    rows=[r] if r else None,
                                    user_prompt=text,
                                    rag_response=full_resp,
                                )
                            except Exception:
                                vizp = None
                            if vizp:
                                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                # --- Sección PRODUCTOS ---
                pf2 = dict(pf)
                rows_prod = list_by_filter(pf2, limit=min(getattr(S, "chat_list_default", 500), 40))
                if not rows_prod and pf2.get("category"):
                    pf2.pop("category", None)
                    rows_prod = list_by_filter(pf2, limit=min(getattr(S, "chat_list_default", 500), 40))

                if rows_prod:
                    yield f"data: [PRODUCTOS] Filtros → {pf2}\n\n"
                    for i, r in enumerate(rows_prod[:10], start=1):
                        yield f"data: {_fmt_row(r, i)}\n\n"

                # CTA final acotado a opciones válidas (intent 'list' de productos)
                facts_for_cta = {"filters": pf2, "n_listados": len(rows_prod or [])}
                yield f"data: {_gen_cta('list', facts_for_cta)}\n\n"
                yield "data: [FIN]\n\n"

            # Persistimos memoria SIN usar 'plan' ni 'with_data'
            try:
                hits_mix = len(list_by_filter(pf, limit=min(getattr(S, "chat_list_default", 500), 40)) or [])
            except Exception:
                hits_mix = 0

            remember_session(
                req.session_id,
                filters=pf if pf else {"country": countries},
                intent="list",  # la parte de productos del MIX es un listado
                query=text,
                hits=hits_mix,
                mentioned=mentioned_early,
            )

            return StreamingResponse(
                gen_mix(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

        # ---- Si NO hay productos en el mismo turno, conserva el comportamiento original ----
        try:
            if "__ALL__" in macros:
                if not countries:
                    reason = _diagnose_no_results("macro_list", text=text, macros=macros, filters={"country": countries})
                    return StreamingResponse(_sse_no_data_ex(reason, {"country": countries or None}), media_type="text/event-stream")

                rows = macro_list(countries[0]) or []
                if not rows:
                    reason = _diagnose_no_results("macro_list", text=text, macros=macros, rows=rows, filters={"country": countries})
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
                    rag_buf = []
                    for chunk in _stream_no_fin(prompt):
                        try:
                            content = chunk.split("data:", 1)[1].strip()
                        except Exception:
                            content = chunk
                        if content and content != "[FIN]":
                            rag_buf.append(content)
                        yield chunk
                    yield "data: \n\n"

                    # VIZ_PROMPT (listado macro por país)
                    try:
                        full_resp = " ".join(rag_buf).strip()
                        vizp = _maybe_viz_prompt(
                            "list",
                            {"country": countries[0]},
                            rows=rows[:10],
                            user_prompt=text,
                            rag_response=full_resp,
                        )
                    except Exception:
                        vizp = None
                    if vizp:
                        yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    yield f"data: Encontré {len(rows)} variable(s). Mostrando las primeras:\n\n"
                    for i, r in enumerate(rows[:10], start=1):
                        yield f"data: {_fmt_macro_row(r, i)}\n\n"

                    yield f"data: {_gen_cta('macro_list', facts)}\n\n"
                    yield "data: [FIN]\n\n"
                MEM.set(req.session_id, {"last_country": countries[0]})
                return StreamingResponse(gen_all(), media_type="text/event-stream")

            if len(countries) >= 2:
                rows = []
                for m in macros:
                    rows.extend(macro_compare(m, countries) or [])
                if not rows:
                    reason = _diagnose_no_results("macro_compare", text=text, macros=macros, rows=rows, filters={"country": countries})
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
                    rag_buf = []
                    for chunk in _stream_no_fin(prompt):
                        try:
                            content = chunk.split("data:", 1)[1].strip()
                        except Exception:
                            content = chunk
                        if content and content != "[FIN]":
                            rag_buf.append(content)
                        yield chunk
                    yield "data: \n\n"

                    # VIZ_PROMPT (comparación multi-país)
                    try:
                        full_resp = " ".join(rag_buf).strip()
                        vizp = _maybe_viz_prompt(
                            "compare",
                            {"country": countries},
                            rows=rows,
                            user_prompt=text,
                            rag_response=full_resp,
                        )
                    except Exception:
                        vizp = None
                    if vizp:
                        yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    for i, r in enumerate(rows, start=1):
                        yield f"data: {_fmt_macro_row(r, i)}\n\n"

                    yield f"data: {_gen_cta('macro_compare', facts)}\n\n"
                    yield "data: [FIN]\n\n"
                return StreamingResponse(gen_cmp(), media_type="text/event-stream")

            if len(countries) == 1:
                # primera macro mencionada por simplicidad
                r = macro_lookup(macros[0], countries[0])
                if not r:
                    reason = _diagnose_no_results("macro_lookup", text=text, macros=macros, rows=[], filters={"country": countries[0] if countries else None})
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
                    rag_buf = []
                    for chunk in _stream_no_fin(prompt):
                        try:
                            content = chunk.split("data:", 1)[1].strip()
                        except Exception:
                            content = chunk
                        if content and content != "[FIN]":
                            rag_buf.append(content)
                        yield chunk
                    yield "data: \n\n"

                    # VIZ_PROMPT (lookup de variable)
                    try:
                        full_resp = " ".join(rag_buf).strip()
                        vizp = _maybe_viz_prompt(
                            "lookup",
                            {"country": countries[0]},
                            rows=[r] if r else None,
                            user_prompt=text,
                            rag_response=full_resp,
                        )
                    except Exception:
                        vizp = None
                    if vizp:
                        yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    yield f"data: {_fmt_macro_row(r)}\n\n"

                    yield f"data: {_gen_cta('macro_lookup', facts)}\n\n"
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
        # usa el limit que vino en el request si existe; si no, un default
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

    # <-- pega aquí
    mentioned = dict(getattr(plan, "explicit_mentions", {}) or {}) or _detect_mentions_from_text(text)


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


        remember_session(req.session_id, filters=plan.filters, intent="list", query=text, hits=len(rows), mentioned=mentioned)
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
                if summary_txt:
                    yield f"data: {summary_txt}\n\n"
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

            # Mini-writer humano (no stream ≠ evita doble FIN)
            try:
                sample = [
                    {"name": r.get("name"), "brand": r.get("brand"), "price": r.get("price"),
                     "currency": r.get("currency"), "store": r.get("store")}
                    for r in top[:3]
                ]
                prompt_summary = (
                    "Eres el asistente del SPI. Formato: saludo breve → contexto (país/categoría/tienda si están) "
                    "→ resumen del TOP con 1–2 ejemplos → CTA único (p.ej., \"¿Filtramos por tienda o marca?\").\n"
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

            # Writer en stream (usa _stream_no_fin para evitar doble FIN)
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


        remember_session(
            req.session_id,
            filters=plan.filters,
            intent="trend",
            query=text,
            hits=len(ser),
            mentioned=mentioned,
                    )

        return StreamingResponse(gen_trend(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})



    # ---- COMPARE → N países (2..10) usando SOLO columnas de la BD ----
    if plan.intent == "compare":
        try:
            countries = _extract_countries(text, max_n=10)
            cat = (plan.filters or {}).get("category")

            # --- Inferir categoría si falta (sinónimos -> semántico) ---
            if not cat:
                cat = _canonicalize_category(text)
                if cat:
                    plan.filters = dict(plan.filters or {}, category=cat)

            # === MULTI-PAÍS + CATEGORÍA → comparación por país ===
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
                            # agregados por categoría en ese país (sin forzar una categoría específica)
                            agg_cat = aggregate_prices({"country": code}, by="category") or {}
                            groups = agg_cat.get("groups") or []
                            candidates = [g.get("group") for g in groups if g and g.get("group")]
                            suggestions[code] = candidates[:3]
                    except Exception:
                        suggestions = {}

                    reason = (
                        f"necesito al menos 2 países con datos para '{cat}', "
                        f"pero tuve {len(with_data)} con datos y {len(without_data)} sin datos."
                    )

                    def gen_hint():
                        yield f"data: Hola. No pude comparar porque {reason}\n\n"  # saludo + respuesta breve
                        yield f"data: Filtros → países: {countries} | categoría: {cat}\n\n"  # contexto
                        for code in without_data:
                            opts = suggestions.get(code) or []
                            if opts:
                                yield f"data: Sugerencia para {code}: prueba con categoría(s) {', '.join(opts)}\n\n"
                            else:
                                yield f"data: Sugerencia para {code}: prueba sin categoría o con otra similar.\n\n"
                        # CTA al final (acotado a opciones válidas de compare)
                        facts_for_cta = {"category": cat, "countries": countries, "with_data": [c for c, _ in with_data]}
                        yield f"data: {_gen_cta('compare', facts_for_cta)}\n\n"
                        yield "data: [FIN]\n\n"

                    return StreamingResponse(gen_hint(), media_type="text/event-stream")

                # Persistimos sesión aquí (ya existe with_data)
                total_hits = sum(len(rows) for _, rows in with_data)
                remember_session(
                    req.session_id,
                    filters=dict(plan.filters or {}, country=countries),
                    intent="compare",
                    query=text,
                    hits=total_hits,
                    mentioned=mentioned,
                )

                # --- Redacción humana con LLM (sin CTA en el prompt; CTA al final) ---
                # preparar hechos por país
                facts = {"category": cat, "countries": []}
                from collections import Counter
                for c, rows in with_data:
                    prices = [r.get("price") for r in rows if r.get("price") is not None]
                    if not prices:
                        continue
                    cur = Counter([r.get("currency") for r in rows if r.get("currency")]).most_common(1)[0][0] if rows else None
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
                    "Formato de TEXTO PLANO, sin markdown ni listas. "
                    "Estructura: 1) saludo breve; 2) contexto (países, categoría, moneda si aplica); "
                    "3) respuesta comparativa (promedio, mínimo, máximo y conteo de marcas por país). "
                    "No incluyas CTA ni instrucciones; solo el texto."
                )

                if getattr(S, "compare_llm", True):
                    def gen_llm():
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
                        yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"  # contexto

                        # Writer humano (sin CTA)
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

                        # CTA AUTOGENERADO (opciones válidas de compare) — al final
                        facts_for_cta = {"category": cat, "countries": countries, "with_data": [c for c, _ in with_data]}
                        yield f"data: {_gen_cta('compare', facts_for_cta)}\n\n"
                        yield "data: [FIN]\n\n"

                    return StreamingResponse(
                        gen_llm(), media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                    )

                # --- Respaldo determinista (sin LLM) con mini-writer humano breve ---
                def gen_dtrm():
                    try:
                        # datos agregados para posible VIZ
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
                    yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"  # contexto

                    # Mini-writer (2–3 frases, sin CTA)
                    try:
                        brief_facts = {"category": cat, "groups": groups}
                        p = (
                            "Escribe 2–3 frases en español, tono profesional y natural, sin markdown, "
                            "que expliquen la comparación entre países para la categoría indicada. "
                            "Incluye una frase de saludo corto y otra con lectura de promedios/mín/máx. "
                            "No añadas CTA ni instrucciones.\n"
                            f"Hechos(JSON): {json.dumps(brief_facts, ensure_ascii=False)}"
                        )
                        txt = (llm_chat.generate(p) or "").strip()
                        if txt:
                            yield f"data: {txt}\n\n"
                    except Exception:
                        # fallback muy simple si el LLM no responde
                        yield "data: Hola, te comparto la comparación por país basada en promedios, mínimos y máximos observados.\n\n"

                    # Lista compacta de ejemplos por país
                    for c, rows in with_data:
                        yield f"data: — País {c}: mostrando hasta {top_per_country} producto(s)\n\n"
                        for i, r in enumerate(rows[:top_per_country], start=1):
                            yield f"data: {_fmt_row(r, i)}\n\n"

                    # CTA AUTOGENERADO — al final
                    facts_for_cta = {"category": cat, "countries": countries, "with_data": [c for c, _ in with_data]}
                    yield f"data: {_gen_cta('compare', facts_for_cta)}\n\n"
                    yield "data: [FIN]\n\n"

                return StreamingResponse(
                    gen_dtrm(), media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )

            # === Fallback: comparación simple cuando no hay (N países + categoría) ===
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
                reason = _diagnose_no_results("compare", plan=plan, text=text, hits=hits, filters=plan.filters)
                return StreamingResponse(_sse_no_data_ex(reason, plan.filters), media_type="text/event-stream")

            # Persistimos aquí (NO hay with_data en este camino)
            remember_session(
                req.session_id,
                filters=plan.filters,
                intent="compare",
                query=text,
                hits=len(hits),
                mentioned=mentioned,
            )

            def gen_single():
                try:
                    vizp = _maybe_viz_prompt("compare", plan.filters or {}, rows=hits[:2], user_prompt=text)
                except NameError:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                yield f"data: Filtros → país: {(plan.filters or {}).get('country') or '-'} | categoría: {(plan.filters or {}).get('category') or '-'} | tienda: {(plan.filters or {}).get('store') or '-'}\n\n"  # contexto

                # Mini-writer humano (sin CTA)
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
                        "Eres el asistente del SPI. Escribe 2–3 frases, sin markdown: "
                        "1) saludo breve; 2) contexto (categoría/país/tienda si están); "
                        "3) mini-comparación clara de los 2 resultados con precios/moneda. "
                        "No incluyas CTA.\n"
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

                # CTA AUTOGENERADO — al final
                facts_for_cta = {"filters": plan.filters or {}, "hits": len(hits)}
                yield f"data: {_gen_cta('compare', facts_for_cta)}\n\n"
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

        remember_session(req.session_id, filters=plan.filters, intent="aggregate", query=text, hits=1 if agg else 0, mentioned=mentioned)
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
            facts_for_cta = {"filters": plan.filters or {}, "group_by": plan.group_by or "category", "groups_n": len(agg.get('groups', []))}
            yield f"data: {_gen_cta('aggregate', facts_for_cta)}\n\n"
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

    remember_session(req.session_id, filters=plan.filters, intent="lookup", query=text,  hits=len(hits), mentioned=mentioned)
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

        # ---- LLM STREAM con TTFB y duración total ----
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


        facts_for_cta = {"filters": plan.filters or {}, "hits": len(hits), "brands_n": len(brands)}
        yield f"data: {_gen_cta('lookup', facts_for_cta)}\n\n"
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
