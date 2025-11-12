# app/synonyms.py — Normalización de consulta a categorías canónicas

from __future__ import annotations
import re
import unicodedata
from typing import Optional

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    # quitar acentos
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    # colapsar espacios
    s = re.sub(r"\s+", " ", s)
    return s

# Mapa de categorías CANÓNICAS de tu colección → patrones/sinónimos frecuentes
# OJO: la columna category en tu DB usa valores como 'arroz', 'aceite', 'lacteos', 'azucar', 'atun', etc.
CATEGORY_SYNONYMS = {
    "arroz": [
        r"\barroz(es)?\b",
    ],
    "aceite": [
        r"\baceite(s)?\b",
        r"\baceite vegetal(es)?\b",
        r"\baceite de (soya|soja|girasol|maiz|maíz|canola|cocina)\b",
    ],
    # En tus datos la leche aparece bajo 'lacteos'
    "lacteos": [
        r"\bleche(s)?\b",
        r"\bleche liquida\b",
        r"\bproducto(s)? lacte(o|os|os)?\b",
        r"\bderivado(s)? lacteo(s)?\b",
        r"\bqueso(s)?\b",
        r"\byogur(t|s)?\b",
        r"\bmantequilla\b",
        r"\bcrema de leche\b",
    ],
    "pan": [
        r"\bpan(es)?\b",
        r"\bpan de molde\b",
        r"\bpan tajado\b",
        r"\bpan lactal\b",
        r"\bsandwich bread\b",
    ],
    "azucar": [
        r"\bazucar(es)?\b",
        r"\bazucar blanca\b",
        r"\bazucar morena\b",
        r"\bazucar refinada\b",
    ],
    "cafe": [
        r"\bcafe(s)?\b",
        r"\bcafe molido\b",
        r"\bcafe tostado\b",
    ],
    "pasta": [
        r"\bpasta(s)?\b",
        r"\bpasta seca\b",
        r"\bfideo(s)?\b",
        r"\bspaghetti\b",
        r"\bmacarr(on|on(es)?)\b",
    ],
    "pollo": [
        r"\bpollo(s)?\b",
        r"\bpollo entero\b",
        r"\bpollo fresco\b",
    ],
    "huevos": [
        r"\bhuevo(s)?\b",
    ],
    "queso": [
        r"\bqueso(s)?\b",
        r"\bqueso blando\b",
        r"\bqueso fresco\b",
        r"\bqueso campesino\b",
        r"\bqueso cuajada\b",
        # Nota: si prefieres que TODO queso caiga en 'lacteos', deja esta cat vacía y confía en 'lacteos'
    ],
    "harina": [
        r"\bharina(s)?\b",
        r"\bharina de trigo\b",
        r"\bharina de trigo\b",
    ],
    "cerveza": [
        r"\bcerveza(s)?\b",
        r"\bcerveza en lata(s)?\b",
        r"\bbeer\b",
    ],
    "gaseosa": [
        r"\bgaseosa(s)?\b",
        r"\brefresco(s)?\b",
        r"\brefrescos de cola\b",
        r"\bcola\b",
        r"\bsoda(s)?\b",
    ],
    "papa": [
        r"\bpapa(s)?\b",
        r"\bpatata(s)?\b",
    ],
    "banano": [
        r"\bbanano(s)?\b",
        r"\bbanana(s)?\b",
        r"\bplatano(s)?\b",
    ],
    "tomate": [
        r"\btomate(s)?\b",
        r"\bjitomate(s)?\b",
    ],
    "cebolla": [
        r"\bcebolla(s)?\b",
        r"\bcebolla cabezona\b",
    ],
    "frijoles": [
        r"\bfrijol(es)?\b",
        r"\bporoto(s)?\b",
        r"\bhabichuela(s)?\b",
        r"\bfrijoles\b",
    ],
    "manzana": [
        r"\bmanzana(s)?\b",
    ],
    "atun": [
        r"\batun(es)?\b",
        r"\bat[uú]n(es)?\b",
        r"\batun enlatado\b",
        r"\blata(s)? de atun\b",
    ],
}

# Compilar patrones (solo una vez)
_COMPILED = {cat: [re.compile(pat, re.IGNORECASE) for pat in pats] for cat, pats in CATEGORY_SYNONYMS.items()}

def detect_category(query: str) -> Optional[str]:
    """
    Devuelve la categoría canónica (la misma que guarda Milvus en 'category')
    si la consulta contiene algún sinónimo/patrón conocido.
    """
    if not query:
        return None
    q = _norm(query)
    for cat, regs in _COMPILED.items():
        if any(r.search(q) for r in regs):
            # Regla especial: si match fue 'queso' y prefieres agrupar en 'lacteos'
            if cat == "queso":
                return "lacteos"
            return cat
    return None
