# app/intent_llm.py — Parseo de intención con LLM (Ollama phi-3-mini) a JSON
from __future__ import annotations
import json, re, unicodedata
import requests
from typing import Optional, Dict
from settings import get_settings
S = get_settings()

# ---------------------------
# utilidades locales
# ---------------------------
def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower())
                   if unicodedata.category(c) != "Mn").strip()

def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: 
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        t = raw.replace("'", '"')
        t = re.sub(r",(\s*[\}\]])", r"\1", t)
        try:
            return json.loads(t)
        except Exception:
            return None

# Canónicos REALES en Milvus (deben coincidir 1:1)
CANON_CATS = {
    "arroz","pan_de_molde","leche_liquida","leche","pasta_seca","azucar","cafe","cafe_molido",
    "aceite_vegetal","huevo","pollo_entero","refrescos_de_cola","papa","frijol","harina_de_trigo",
    "cerveza","queso_blando","atun","atun_en_lata","tomate","cebolla","manzana","banano","pan"
}
# Términos genéricos: no fijar categoría (dejarlos en null para refinamiento semántico)
GENERIC_TERMS = {"lacteos","lácteos","bebidas","alimentos","hogar","aseo"}

# Mapeos simples alias → canónico (por si el LLM se sale del guion)
ALIAS_TO_CANON = {
    "aceite": "aceite_vegetal",
    "refresco": "refrescos_de_cola", "gaseosa": "refrescos_de_cola",
    "soda": "refrescos_de_cola", "cola": "refrescos_de_cola",
    "pan de molde": "pan_de_molde",
    "pasta": "pasta_seca",
    "harina": "harina_de_trigo",
    "atun en lata": "atun_en_lata", "atún en lata": "atun_en_lata",
    "cafe molido": "cafe_molido", "café molido": "cafe_molido",
    "queso": "queso_blando",
    "leche en polvo": "leche",
    "leche polvo": "leche",
    "leche": "leche_liquida",  # regla clave
}

COUNTRY_MAP = {
    "argentina":"AR","colombia":"CO","mexico":"MX","méxico":"MX","brasil":"BR","brazil":"BR",
    "chile":"CL","peru":"PE","perú":"PE","ecuador":"EC","costa rica":"CR","panama":"PA","panamá":"PA",
    "paraguay":"PY","uruguay":"UY","bolivia":"BO"
}

def _canon_from_text(txt: str) -> Optional[str]:
    """Normaliza y convierte a canónico si podemos; si es genérico -> None."""
    n = _norm(txt)
    if n in GENERIC_TERMS:
        return None
    # prueba directo (ya canónico)
    if n in CANON_CATS:
        return n
    # cambia espacios por guión bajo y reintenta
    n_us = n.replace(" ", "_")
    if n_us in CANON_CATS:
        return n_us
    # alias conocidos
    if n in ALIAS_TO_CANON:
        return ALIAS_TO_CANON[n]
    return None

# ---------------------------
# LLM
# ---------------------------
def parse_intent(message: str) -> Dict:
    """
    Devuelve:
      {"category": <canónico o None>, "country": <ISO2 o None>, "store": <str|None>, "brand": <str|None>}
    """
    base = getattr(S, "ollama_host", "http://127.0.0.1:11434").rstrip("/")
    model = getattr(S, "gen_model", "phi3:mini")

    # Prompt ESTRICTO: solo canónicos reales + reglas
    canons_list = ", ".join(sorted(CANON_CATS))
    system = (
        "Eres un parser de intención para consultas de precios de supermercado en LATAM.\n"
        "Devuelve SOLO un JSON con posibles filtros, usando claves EXACTAS: category, country, store, brand.\n"
        "Reglas:\n"
        f"- category DEBE ser uno de estos canónicos: {canons_list}.\n"
        "- Si el usuario dice 'leche en polvo' o 'leche polvo' → category='leche'.\n"
        "- Si el usuario dice 'leche' (sin 'polvo') → category='leche_liquida'.\n"
        "- Si el usuario usa términos genéricos como 'lácteos', 'bebidas', 'alimentos', 'hogar', 'aseo', deja category=null.\n"
        "- country: usa códigos ISO-2 si están explícitos (AR, BR, CL, CO, CR, MX, PA, PY, etc.). Si no está claro, deja null.\n"
        "- brand/store: solo si están explícitos en el texto; si no, deja null.\n"
        "Responde SOLO el JSON, sin explicaciones.\n"
        "Ejemplo válido: {\"category\":\"leche_liquida\",\"country\":\"CO\",\"store\":null,\"brand\":null}\n"
    )
    prompt = f"Usuario: {message}\nJSON:"

    # Llamada a Ollama
    try:
        r = requests.post(f"{base}/api/generate", json={
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 1024, "num_predict": 128}
        }, timeout=60)
        r.raise_for_status()
        txt = (r.json().get("response") or "").strip()
        data = _extract_json(txt) or {}
    except Exception:
        data = {}

    # Sanitización mínima de claves
    out = {k: None for k in ("category","country","store","brand")}
    for k in out.keys():
        if k in data:
            v = data.get(k)
            out[k] = v if (v not in ("", None)) else None

    # ---- Normaliza categoría a canónico o None ----
    cat = out.get("category")
    if isinstance(cat, str):
        cat_norm = _canon_from_text(cat)
        out["category"] = cat_norm  # puede ser None si fue genérico o desconocido

    # ---- Fallback simple para país (si el LLM no lo dio) ----
    if not out.get("country"):
        nt = _norm(message)
        for k, iso2 in COUNTRY_MAP.items():
            if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", nt):
                out["country"] = iso2
                break

    return out
