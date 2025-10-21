# app/intent_llm.py — Parseo de intención con LLM (Ollama phi-3-mini) a JSON
from __future__ import annotations
import json, re
import requests
from typing import Optional, Dict
from settings import get_settings
S = get_settings()



def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # segundo intento: arreglar comas finales / comillas simples simples
        t = m.group(0).replace("'", '"')
        t = re.sub(r",(\s*[\}\]])", r"\1", t)
        try:
            return json.loads(t)
        except Exception:
            return None

def parse_intent(message: str) -> Dict:
    """
    Devuelve un JSON con campos propuestos por el LLM:
    {
      "category": "lacteos" | null,
      "country": "CO" | "AR" | ... | null,
      "store": "Exito" | null,
      "brand": "LaLechera" | null
    }
    """
    base = getattr(S, "ollama_host", "http://127.0.0.1:11434").rstrip("/")
    model = getattr(S, "gen_model", "phi3:mini")
    system = (
        "Eres un parser de intención para consultas de precios de supermercado en LATAM.\n"
        "Devuelve SOLO un JSON con posibles filtros, usando claves: category, country, store, brand.\n"
        "Reglas:\n"
        "- category debe coincidir con categorías genéricas (arroz, aceite, lacteos, pan, azucar, cafe, pasta, pollo, huevos, queso, harina, cerveza, gaseosa, papa, banano, tomate, cebolla, frijoles, manzana, atun).\n"
        "- country usa códigos de país ISO-2 si están explícitos (AR, BR, CL, CO, CR, MX, PA, etc.). Si no está claro, deja null.\n"
        "- Si piden 'leche', category = 'lacteos'. Si piden 'refresco/cola/soda', category = 'gaseosa'.\n"
        "- Si no estás seguro, deja null.\n"
        "Ejemplo de salida: {\"category\":\"lacteos\",\"country\":\"CO\",\"store\":null,\"brand\":null}\n"
    )
    prompt = f"Usuario: {message}\nJSON:"
    try:
        r = requests.post(f"{base}/api/generate", json={
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 1024, "num_predict": 256}
        }, timeout=60)
        r.raise_for_status()
        txt = (r.json().get("response") or "").strip()
        data = _extract_json(txt) or {}
        # saneo mínimo
        for k in ["category","country","store","brand"]:
            if k not in data: data[k] = None
        if isinstance(data.get("category"), str):
            data["category"] = data["category"].lower().strip()
        return data
    except Exception:
        return {"category": None, "country": None, "store": None, "brand": None}
