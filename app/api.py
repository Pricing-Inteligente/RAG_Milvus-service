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

import time



# === Config ===
from settings import get_settings
S = get_settings()

# === Milvus helpers ===
from retrieve import retrieve, list_by_filter, aggregate_prices

# === Inteligencia de intención / categoría ===
from intent_llm import parse_intent
from category_resolver import resolve_category_semantic



# cerca de imports
def _log_event(kind: str, payload: dict):
    try:
        os.makedirs("logs", exist_ok=True)
        rec = {"ts": datetime.utcnow().isoformat() + "Z", "kind": kind, **payload}
        with open("logs/trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Utilidades de normalización y alias
# -----------------------------------------------------------------------------
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

CAT_ALIASES = {
    "azucar":   ["azucar", "azúcar", "acucar", "açucar"],
    "arroz":    ["arroz"],
    "tomate":   ["tomate", "jitomate"],
    "aceite":   ["aceite", "oleo", "óleo", "aceite vegetal"],
    "huevo":    ["huevo", "huevos", "ovo", "ovos"],
    "pan":      ["pan", "pao", "pão", "pan de molde", "pan tajado", "pan lactal"],
    "atun":     ["atun", "atún", "atum"],
    "bebidas":  ["bebidas", "bebida"],
    "lacteos":  ["lacteos", "lácteos", "leche", "leches", "leite", "leche liquida", "leche líquida", "yogur", "yogurt", "mantequilla", "queso"],
    "pasta":    ["pasta", "macarrao", "macarrão", "fideos", "spaghetti"],
    "legumbres":["legumbres", "feijao", "feijão", "frijoles", "porotos", "habichuelas"],
    "gaseosa":  ["gaseosa", "refresco", "refrescos de cola", "cola", "soda"],
    "papa":     ["papa", "patata"],
    "banano":   ["banano", "banana", "plátano", "platano"],
    "cebolla":  ["cebolla"],
    "manzana":  ["manzana"],
    "harina":   ["harina", "harina de trigo"],
    "cerveza":  ["cerveza", "cerveza en lata", "beer"],
    "queso":    ["queso", "queso blando", "queso fresco"],
}

NCATEGORIES = { _norm(alias): canon
                for canon, aliases in CAT_ALIASES.items()
                for alias in aliases }

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

# -----------------------------------------------------------------------------
# Fusión inteligente de filtros (Heurística + LLM + Semántico)
# -----------------------------------------------------------------------------
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
    if llm.get("category"): filters.setdefault("category", llm["category"])

    # Semantic fallback (only category)
    if "category" not in filters or not filters.get("category"):
        cat_sem, score_sem, _ = resolve_category_semantic(
            message, min_cosine=float(getattr(S, "category_min_cosine", 0.45))
        )
        if cat_sem:
            filters["category"] = cat_sem

    return sanitize_filters(filters)

# -----------------------------------------------------------------------------
# Memoria de sesión (simple LRU en memoria)
# -----------------------------------------------------------------------------
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

    merged = dict(filters)  # ya viene "sanitize_filters" aguas arriba normalmente

    # --- country ---
    if not merged.get("country") and lastf.get("country"):
        merged["country"] = lastf["country"]

    # --- category ---
    if not merged.get("category") and lastf.get("category"):
        merged["category"] = lastf["category"]
    if prefer_last_cat and not m_cat and lastf.get("category"):
        # Fuerza la categoría anterior cuando no se mencionó explícitamente una nueva
        merged["category"] = lastf["category"]

    # --- store (política de persistencia condicionada) ---
    if m_sto:
        # Si mencionaron tienda, respetamos lo que ya venga en merged
        pass
    else:
        # Si NO mencionaron tienda:
        #   - Copiamos la anterior SOLO si categoría y país quedaron iguales
        #   - Si categoría/país cambiaron, descartamos tienda previa
        same_cat = merged.get("category") == lastf.get("category")
        same_cty = merged.get("country") == lastf.get("country")
        if not merged.get("store"):
            if same_cat and same_cty and lastf.get("store"):
                merged["store"] = lastf["store"]
        else:
            # merged ya trae store (p.ej. por heurística), pero si el usuario cambió
            # cat/país sin mencionar tienda, eliminamos la tienda para no arrastrarla
            if not (same_cat and same_cty):
                merged.pop("store", None)

    return merged



def _filters_head(f: Dict) -> str:
    return ("Filtros → "
            f"país: {f.get('country') or '-'} | "
            f"categoría: {f.get('category') or '-'} | "
            f"tienda: {f.get('store') or '-'}")

# -----------------------------------------------------------------------------
# Visualización — construcción de prompt para la API externa de gráficos
# -----------------------------------------------------------------------------
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

def _viz_prompt_from_rows(rows: List[Dict], filters: Dict, *, title: str | None = None,
                          max_n: int = 8, label_priority: List[str] = ["brand","name"]) -> str:
    """
    Construye un prompt NL + dataset JSON para una barra simple (label vs price).
    El front/servicio de charts puede entender 'label' como eje X y 'value' como eje Y.
    """
    if not rows:
        return ""
    data = []
    for r in rows[:max_n]:
        label = None
        for k in label_priority:
            v = (r.get(k) or "").strip()
            if v: label = v; break
        if not label:
            label = (r.get("name") or r.get("brand") or r.get("store") or r.get("product_id"))
        if r.get("price") is None:
            continue
        data.append({
            "label": label,
            "value": float(r["price"]),
            "currency": r.get("currency"),
            "store": r.get("store"),
            "brand": r.get("brand"),
            "country": r.get("country"),
            "product_id": r.get("product_id"),
        })
    if not data:
        return ""
    title = title or _viz_title(filters, "lookup")
    nl = (
        f"Genera una gráfica de barras titulada '{title}'. "
        "Eje X: 'label'. Eje Y: 'value' (precio). "
        "Usa el campo 'currency' solo para rotular si aplica. "
        "Muestra etiquetas con 'brand' y 'store' cuando existan. "
        f"Datos (JSON): {json.dumps(data, ensure_ascii=False)}"
    )
    return nl

def _viz_prompt_from_agg(agg: Dict, filters: Dict, *, group_by: str) -> str:
    """
    Para agregados: usa el promedio como valor principal y pasa min/max para tooltips.
    Schema: [{label, value, min, max}]
    """
    groups = (agg or {}).get("groups") or []
    if not groups:
        return ""
    data = [{
        "label": str(g.get("group")),
        "value": float(g.get("avg")) if g.get("avg") is not None else None,
        "min": float(g.get("min")) if g.get("min") is not None else None,
        "max": float(g.get("max")) if g.get("max") is not None else None,
    } for g in groups if g.get("group") is not None]
    data = [d for d in data if d["value"] is not None]
    if not data:
        return ""
    title = _viz_title(filters, "aggregate", group_by=group_by)
    nl = (
        f"Genera una gráfica de barras titulada '{title}'. "
        f"Eje X: '{group_by}'. Eje Y: 'value' (precio promedio). "
        "Incluye bandas o tooltips con 'min' y 'max' si el sistema lo soporta. "
        f"Datos (JSON): {json.dumps(data, ensure_ascii=False)}"
    )
    return nl

def _maybe_viz_prompt(intent: str, filters: Dict, *, rows: List[Dict] | None = None,
                      agg: Dict | None = None, group_by: str | None = None) -> str | None:
    try:
        if intent in ("lookup", "list", "compare") and rows:
            return _viz_prompt_from_rows(rows, filters)
        if intent == "aggregate" and agg and (agg.get("groups") or []):
            gb = group_by or "category"
            return _viz_prompt_from_agg(agg, filters, group_by=gb)
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

def remember_session(sid: Optional[str], *, filters: Dict, intent: str, query: str, hits: List[Dict]):
    if not sid:
        return
    MEM.set(sid, {
        "last_filters": filters,
        "last_intent": intent,
        "last_query": query,
        "last_product_ids": [h.get("product_id") for h in (hits[:10] if hits else [])],
        "last_category": filters.get("category"),
        "last_country": filters.get("country"),
    })

# -----------------------------------------------------------------------------
# Logging / Trazabilidad
# -----------------------------------------------------------------------------
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




# -----------------------------------------------------------------------------
# CORS / App
# -----------------------------------------------------------------------------
app = FastAPI(title="RAG Pricing API", version="1.8.0")



@app.options("/chat/stream")
def options_chat_stream():
    # 204 vacío: el CORSMiddleware añadirá los headers CORS permitidos
    return Response(status_code=204)



origins = [
    "http://localhost:5173",  # tu frontend local
    "http://127.0.0.1:5173",  # a veces Vite usa 127.0.0.1
    "http://localhost:8080",  # Lovable local
    "http://localhost:8081",      # <— añade esto
    "http://127.0.0.1:8081",
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Cliente LLM (Ollama)
# -----------------------------------------------------------------------------
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

llm_strict = OllamaLLM(
    model=getattr(S, "gen_model", "phi3:mini"),
    base_url=getattr(S, "ollama_host", "http://127.0.0.1:11434"),
    temperature=0.0, num_ctx=1024, num_predict=128,
)
llm_chat = OllamaLLM(
    model=getattr(S, "gen_model", "phi3:mini"),
    base_url=getattr(S, "ollama_host", "http://127.0.0.1:11434"),
    temperature=0.35, num_ctx=1024, num_predict=256,
)

def _llm_json(prompt: str) -> str:
    return llm_strict.generate(prompt)

# -----------------------------------------------------------------------------
# Root/health/runtime
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Prompts RAG
# -----------------------------------------------------------------------------
def _prompt_answer_friendly(question: str, ctx: str) -> str:
    return (
        "Eres un asistente de retail. SOLO puedes usar los datos del CONTEXTO.\n"
        "Si el CONTEXTO no contiene la respuesta exacta, responde exactamente:\n"
        "\"No tengo esa información en la base\".\n"
        "NO ofrezcas contactar proveedores ni consultar otras bases o páginas externas.\n"
        "No inventes valores, ni menciones campos que no estén en la base.\n"
        "Redacta de forma cordial y natural, en 2 a 4 frases máximo. "
        "Comienza con un saludo breve similar a: \"Hola, con gusto te ayudo.\" "
        "Incluye la presentación (tamaño) y tienda si están en el CONTEXTO. "
        "Debes citar al menos un [product_id] de los productos que uses, preferiblemente al final.\n\n"
        f"CONTEXTO:\n{ctx}\n\n"
        f"PREGUNTA: {question}\n"
        "RESPUESTA:"
    )


def _prompt_compare_friendly(user_text: str, ctx_lines: List[str]) -> str:
    return (
        "Compara SOLO los productos del CONTEXTO (precio y presentación). "
        "Si no es concluyente, responde exactamente: \"No tengo esa información en la base\".\n"
        "Redacta de forma cordial y natural, en 2 a 4 frases máximo. "
        "Comienza con un saludo breve similar a: \"Hola, claro que sí.\" "
        "Debes citar al menos un [product_id] de los productos que uses.\n\n"
        f"CONTEXTO:\n{chr(10).join(ctx_lines)}\n\n"
        f"PREGUNTA: {user_text}\n"
        "RESPUESTA:"
    )

# -----------------------------------------------------------------------------
# Small-talk
# -----------------------------------------------------------------------------
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
class AskReq(BaseModel):
    question: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = None
    abstain_threshold: Optional[float] = None
    session_id: Optional[str] = None

class ListReq(BaseModel):
    filters: Optional[Dict] = None
    limit: Optional[int] = 100
    page: Optional[int] = 1

class AggregateReq(BaseModel):
    filters: Optional[Dict] = None
    group_by: Optional[Literal["store", "category", "country"]] = None
    operation: Optional[Literal["min", "max", "avg"]] = None

class Plan(BaseModel):
    intent: Literal["lookup","list","aggregate","compare","count"]
    filters: Optional[Dict] = Field(default_factory=dict)
    product_name: Optional[str] = None
    product_name_b: Optional[str] = None
    group_by: Optional[Literal["store","category","country"]] = None
    operation: Optional[Literal["min","max","avg"]] = None
    top_k: Optional[int] = 5
    limit: Optional[int] = 100

class ChatReq(BaseModel):
    message: str
    limit: Optional[int] = 100
    mode: Optional[Literal["lookup","list","aggregate","compare","count"]] = None
    session_id: Optional[str] = None

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
        ("muéstrame los lácteos en peru", {"intent":"list","filters":{"country":"PE","category":"lacteos"}}),
        ("aceite vegetal 900ml en argentina", {"intent":"lookup","filters":{"country":"AR","category":"aceite"}}),
        ("¿cuántos productos hay en chile?", {"intent":"count","filters":{"country":"CL"}}),
        ("promedio de precios por país para arroz", {"intent":"aggregate","group_by":"country","filters":{"category":"arroz"}}),
    ]
    prompt = (
        "Devuelve SOLO un JSON que cumpla exactamente este esquema, sin texto extra.\n"
        f"Esquema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        "- Normaliza country a ISO2 entre: MX, BR, AR, CO, CL, PE, EC, CR, PA, PY.\n"
        "- category usa canónicos: azucar, arroz, lacteos, tomate, aceite, huevo, pan, atun, bebidas, pasta, legumbres, gaseosa, papa, banano, cebolla, manzana, harina, cerveza, queso.\n"
        "- Si el usuario pide \"leche\", normaliza a category=\"lacteos\".\n"
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
# Facetas
# -----------------------------------------------------------------------------
def _facet_suggestions(filters: Dict, top_n: int = 5) -> Dict[str, List[str]]:
    try:
        by_cat = aggregate_prices(sanitize_filters(filters), by="category").get("groups", [])
        by_store = aggregate_prices(sanitize_filters(filters), by="store").get("groups", [])
        cats = [g["group"] for g in by_cat if g.get("group")][:top_n]
        stores = [g["group"] for g in by_store if g.get("group")][:top_n]
        return {"categories": cats, "stores": stores}
    except Exception:
        return {"categories": [], "stores": []}

# -----------------------------------------------------------------------------
# Endpoints principales
# -----------------------------------------------------------------------------
@app.post("/ask", tags=["rag"])
def ask(req: AskReq):
    try:
        top_k = req.top_k or getattr(S, "top_k", 5)

        # Filtros con fusión + memoria (si session_id)
        base_f = build_filters_smart(req.question, req.filters)

        # ¿El usuario mencionó categoría explícita en este turno?
        heur_now = _guess_filters(req.question)
        prefer_last_cat = not bool(heur_now.get("category"))

        # Aplica memoria con regla "la categoría de sesión manda" si no se mencionó
        f = merge_with_memory(base_f, req.session_id, prefer_last_cat=prefer_last_cat)

        # Texto efectivo para el dense search (hereda el 'tema' si hace falta)
        effective_q = pick_effective_query(req.question, req.session_id, prefer_last_cat)

        hits: List[Dict] = retrieve(effective_q, f)
        if not hits:
            return {"answer": "No tengo esa información en la base", "evidence": [], "filters": f}

        ctx = "\n".join(
            f"[{h['product_id']}] {h['name']} | Marca: {h['brand']} | "
            f"Pres: {h['size']}{h['unit']} | Precio: {h['price']} {h['currency']} | "
            f"Tienda: {h['store']} | País: {h['country']}"
            for h in hits[:top_k]
        )
        prompt = _prompt_answer_friendly(req.question, ctx)
        txt = llm_chat.generate(prompt)
        ids = re.findall(r"\[(.*?)\]", txt)
        if not txt or not ids:
            return {"answer": "No tengo esa información en la base", "evidence": [], "filters": f}

        ids = list(dict.fromkeys(ids))
        ev = [h for h in hits if h["product_id"] in ids]
        if not ev:
            return {"answer": "No tengo esa información en la base", "evidence": [], "filters": f}

        remember_session(req.session_id, filters=f, intent="lookup", query=req.question, hits=hits)
        return {"answer": txt, "evidence": ev, "top_k_used": top_k, "filters": f}
    except Exception as e:
        return {"answer": "No puedo acceder a la base en este momento.", "evidence": [], "error": str(e)[:200]}

@app.post("/list", tags=["products"])
def list_products(req: ListReq):
    try:
        lim = max(1, min(req.limit or 100, 1000))
        page = max(1, req.page or 1)
        rows_all = list_by_filter(sanitize_filters(req.filters), limit=lim * page)
        if not rows_all and req.filters and req.filters.get("category"):
            f2 = dict(req.filters); f2.pop("category", None)
            rows_all = list_by_filter(sanitize_filters(f2), limit=lim * page)
        start = (page - 1) * lim
        rows = rows_all[start:start + lim]
        reply = "Te muestro los productos que encontré con ese filtro." if rows else "No tengo esa información en la base"
        chips = _facet_suggestions(req.filters or {})
        return {
            "count": len(rows_all),
            "page": page,
            "page_size": lim,
            "items": rows,
            "reply": reply,
            "suggestions": chips
        }
    except Exception as e:
        return {"count": 0, "items": [], "reply": "No puedo acceder a la base en este momento.", "error": str(e)[:200], "suggestions": {"categories": [], "stores": []}}

@app.post("/aggregate", tags=["products"])
def aggregate(req: AggregateReq):
    try:
        result = aggregate_prices(sanitize_filters(req.filters), by=req.group_by)
        if req.operation and result.get("groups"):
            for g in result["groups"]:
                k = req.operation
                for m in ["min", "max", "avg"]:
                    if m != k and m in g:
                        del g[m]
        if result.get("groups"):
            ctx = json.dumps(result.get("groups", [])[:10], ensure_ascii=False)
            prompt = (
                f"Eres {ASSISTANT_NAME}. Redacta un resumen claro (2–3 frases) "
                f"de estas agregaciones (JSON): {ctx}. No inventes valores."
            )
            summary = llm_chat.generate(prompt) or "Aquí tienes un resumen de precios por grupo."
            result["reply"] = summary
        else:
            result["reply"] = "No tengo esa información en la base"
        return result
    except Exception as e:
        return {"groups": [], "reply": "No puedo acceder a la base en este momento.", "error": str(e)[:200]}

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

@app.post("/chat", tags=["chat"])
def chat(req: ChatReq):
    text = req.message.strip()

    # Small-talk
    small = _is_smalltalk(text)
    if small:
        prompt = _prompt_smalltalk(text, small)
        reply = llm_chat.generate(prompt)
        return {"type":"text", "reply": reply, "evidence": [], "model": llm_chat.model}

    # Planner + fusión
    plan = _plan_from_llm(text)
    heur = _guess_filters(text)
    base_filters = plan.filters if plan else heur

    merged = build_filters_smart(text, base_filters)
    # Preferir categoría de sesión si el turno NO menciona una
    heur_now = _guess_filters(text)
    prefer_last_cat = not bool(heur_now.get("category"))
    merged = merge_with_memory(merged, req.session_id, prefer_last_cat=prefer_last_cat)

    if not plan:
        plan = Plan(intent=_classify_intent_heuristic(text),
                    filters=merged,
                    top_k=getattr(S, "top_k", 5),
                    limit=min(max(req.limit or 100, 1), 1000))
    else:
        plan.filters = merged

    if req.mode:
        plan.intent = req.mode

    # ---- EXECUTOR ----
    try:
        if plan.intent == "list":
            items = list_by_filter(plan.filters or None, limit=min(max(plan.limit or 100, 1), 1000))
            if not items and plan.filters and plan.filters.get("category"):
                f2 = dict(plan.filters); f2.pop("category", None)
                items = list_by_filter(f2, limit=min(max(plan.limit or 100, 1), 1000))
            remember_session(req.session_id, filters=plan.filters, intent="list", query=text, hits=items)
            if not items:
                return {"type":"table","reply":"No tengo esa información en la base","count":0,"items":[],"suggestions":{"categories":[],"stores":[]}}
            chips = _facet_suggestions(plan.filters or {})
            return {"type":"table","reply":f"Aquí tienes {len(items)} producto(s).","count":len(items),"items":items,"suggestions":chips}

        if plan.intent == "count":
            items = list_by_filter(plan.filters or None, limit=1000)
            remember_session(req.session_id, filters=plan.filters, intent="count", query=text, hits=items)
            return {"type":"text","reply":f"Tengo {len(items)} registro(s) que cumplen ese filtro.","evidence":[]}

        if plan.intent == "aggregate":
            if not plan.group_by:
                nt2 = _norm(text)
                if "tienda" in nt2: plan.group_by = "store"
                elif "categor" in nt2: plan.group_by = "category"
                elif "pais" in nt2 or "país" in nt2: plan.group_by = "country"
            agg = aggregate_prices(plan.filters or None, by=plan.group_by)
            _log_event("chat_stream_aggregate", {
            "sid": req.session_id,
            "filters": plan.filters,
            "group_by": plan.group_by or "category",
            "rows": (agg or {}).get("groups", [])[:10],
            })

            remember_session(req.session_id, filters=plan.filters, intent="aggregate", query=text, hits=[])
            if not agg.get("groups"):
                return {"type":"text","reply":"No tengo esa información en la base","evidence":[]}
            if plan.operation:
                for g in agg["groups"]:
                    for m in ["min","max","avg"]:
                        if m != plan.operation and m in g:
                            del g[m]
            return {"type":"aggregate","reply":"Aquí tienes un resumen de precios.","result":agg}

        if plan.intent == "compare":
            # Soportar comparación con 1 o 0 nombres: top-2 del retrieve
            effective_q = pick_effective_query(text, req.session_id, prefer_last_cat)
            hits = retrieve(effective_q, plan.filters or None)[: max(plan.top_k or 5, 5)]
            if len(hits) < 2:
                return {"type":"text","reply":"No tengo esa información en la base para comparar.","evidence":[]}
            ctx_lines = []
            for h in hits[:2]:
                ctx_lines.append(f"[{h['product_id']}] {h['name']} | {h['price']} {h['currency']} | {h['store']} | {h['country']}")
            prompt = _prompt_compare_friendly(text, ctx_lines)
            txt = llm_chat.generate(prompt)
            ids = re.findall(r"\[(.*?)\]", txt)
            ev = [h for h in hits if h["product_id"] in ids]
            remember_session(req.session_id, filters=plan.filters, intent="compare", query=text, hits=hits)
            if not txt or not ev:
                return {"type":"text","reply":"No tengo esa información en la base","evidence":[]}
            return {"type":"text","reply":txt,"evidence":ev}

        # default: lookup
        effective_q = pick_effective_query(text, req.session_id, prefer_last_cat)
        hits = retrieve(effective_q if not plan.product_name else plan.product_name,
                        plan.filters or None)[: plan.top_k or getattr(S, "top_k", 5)]
        if not hits and plan.filters and plan.filters.get("category"):
            f2 = dict(plan.filters); f2.pop("category", None)
            hits = retrieve(effective_q if not plan.product_name else plan.product_name, f2)[: plan.top_k or getattr(S, "top_k", 5)]
        remember_session(req.session_id, filters=plan.filters, intent="lookup", query=text, hits=hits)
        _log_event("chat_stream_lookup", {
        "sid": req.session_id,
        "filters": plan.filters,
        "ids": [h.get("product_id") for h in (hits or [])],
        })

        if not hits:
            rows = list_by_filter(plan.filters or {}, limit=min(max(plan.limit or 20, 1), 100))
            _log_event("chat_stream_list", {
            "sid": req.session_id,
            "filters": plan.filters,
            "count": len(rows) if rows else 0,
            "sample_ids": [r.get("product_id") for r in (rows[:10] if rows else [])],
            })

            if rows:
                chips = _facet_suggestions(plan.filters or {})
                msg = (f"Encontré {len(rows)} producto(s) con tu filtro. "
                       "¿Quieres afinar por categoría/marca/unidad?")
                return {"type":"table","reply":msg,"count":len(rows),"items":rows[:10],"suggestions":chips}
            return {"type":"text","reply":"No tengo esa información en la base","evidence":[]}

        ctx = _build_ctx(hits, plan.top_k or getattr(S, "top_k", 5))
        prompt = _prompt_answer_friendly(text, ctx)
        txt = llm_chat.generate(prompt)
        ids = re.findall(r"\[(.*?)\]", txt)
        ev = [h for h in hits if h["product_id"] in ids]
        if not txt or not ev:
            return {"type":"text","reply":"No tengo esa información en la base","evidence":[]}
        return {"type":"text","reply":txt,"evidence":ev}
    except Exception as e:
        return {"type":"text","reply":"No puedo acceder a la base en este momento.","evidence":[],"error":str(e)[:200]}

# -----------------------------------------------------------------------------
# /chat/stream (memoria + encabezado de filtros)
# -----------------------------------------------------------------------------
@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatReqStream):
    text = (req.message or "").strip()
    t_req0 = _now_ms()
    
    def _sse_no_data():
        yield "data: No tengo esa información en la base\n\n"

    # Small-talk corto
    small = _is_smalltalk(text)
    if small:
        prompt = _prompt_smalltalk(text, small)
        _log_event("chat_stream_smalltalk", {
            "sid": req.session_id, "message": text, "intent": small
        })
        return StreamingResponse(llm_chat.stream(prompt), media_type="text/event-stream")

    # 1) Planner LLM (si aplica) + heurística directa  (TIMED)
    planner_ms = None                     # ← NUEVO
    t_pl0 = _now_ms()                     # ← NUEVO
    try:
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
            limit=min(max(req.limit or 100, 1), 1000),
        )
    else:
        plan.filters = merged

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
            rows = list_by_filter(plan.filters or None, limit=min(max(plan.limit or 100, 1), 1000))
            if not rows and plan.filters and plan.filters.get("category"):
                f2 = dict(plan.filters); f2.pop("category", None)
                rows = list_by_filter(f2, limit=min(max(plan.limit or 100, 1), 1000))
        except Exception as e:
            _log_event("chat_stream_list_error", {"sid": req.session_id, "err": str(e)[:200]})
            return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

        remember_session(req.session_id, filters=plan.filters, intent="list", query=text, hits=rows)
        _log_event("chat_stream_list", {
            "sid": req.session_id,
            "filters": plan.filters,
            "count": len(rows),
            "sample_ids": [r.get("product_id") for r in (rows[:10] or [])],
        })
        if not rows:
            return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

        def gen():
            # --- VIZ_PROMPT ---
            try:
                vizp = _maybe_viz_prompt("list", plan.filters or {}, rows=rows)
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"
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
            yield "data: [FIN]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    # ---- COMPARE → stream de resumen comparativo ----
    # ---- COMPARE → N países (2..10) usando SOLO columnas de la BD ----
    if plan.intent == "compare":
        try:
            countries = _extract_countries(text, max_n=10)
            cat = (plan.filters or {}).get("category")

            # Si el usuario mencionó >=2 países y hay categoría → multi-compare
            if len(countries) >= 2 and cat:
                per_country_rows: list[tuple[str, list[dict]]] = []
                top_per_country = max(min(plan.top_k or 3, 5), 1)  # 1..5 por país
                # No arrastramos tienda al comparar países
                for code in countries:
                    f = dict(plan.filters or {})
                    f["country"] = code
                    f.pop("store", None)
                    rows = list_by_filter(f, limit=min(max(plan.limit or 100, 1), 1000)) or []
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

                # Política: si hay menos de 2 países con datos, no podemos comparar
                if len(with_data) < 2:
                    return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

                # Stream determinista (sin LLM), SOLO columnas permitidas
                def gen():
                    # --- VIZ_PROMPT agregado por país (avg/min/max) ---
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
                        agg_for_viz = {"groups": groups}
                        vizp = _maybe_viz_prompt(
                            "aggregate",
                            {"category": cat, "country": [c for c, _ in with_data]},
                            agg=agg_for_viz,
                            group_by="country",
                        )
                    except NameError:
                        vizp = None
                    if vizp:
                        yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                    head = " | ".join(countries)
                    yield f"data: Filtros → categoría: {cat} | países: {head} | tienda: -\n\n"
                    if without_data:
                        miss = ", ".join(without_data)
                        yield f"data: Aviso: sin registros en la base para: {miss}\n\n"

                    for c, rows in with_data:
                        yield f"data: — País {c}: mostrando hasta {top_per_country} producto(s)\n\n"
                        for i, r in enumerate(rows[:top_per_country], start=1):
                            yield f"data: {_fmt_row(r, i)}\n\n"
                    yield "data: [FIN]\n\n"

                return StreamingResponse(
                    gen(), media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )

            # Fallback: comparación simple (cuando no hay N países + categoría)
            effective_q = pick_effective_query(
                text, req.session_id, prefer_last_cat=not bool((plan.filters or {}).get("category"))
            )
            hits = retrieve(effective_q, plan.filters or None)[: max(plan.top_k or 5, 5)]
            _log_event("chat_stream_compare_single", {
                "sid": req.session_id,
                "filters": plan.filters,
                "effective_query": effective_q,
                "ids": [h.get("product_id") for h in (hits or [])],
            })
            if len(hits) < 2:
                return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

            def gen_single():
                # --- VIZ_PROMPT con los 2 primeros resultados ---
                try:
                    vizp = _maybe_viz_prompt("compare", plan.filters or {}, rows=hits[:2])
                except NameError:
                    vizp = None
                if vizp:
                    yield f"data: [VIZ_PROMPT] {vizp}\n\n"

                yield f"data: Filtros → país: {(plan.filters or {}).get('country') or '-'} | categoría: {(plan.filters or {}).get('category') or '-'} | tienda: {(plan.filters or {}).get('store') or '-'}\n\n"
                yield "data: Comparativa simple (primeros 2 resultados):\n\n"
                for i, h in enumerate(hits[:2], start=1):
                    yield f"data: {_fmt_row(h, i)}\n\n"
                yield "data: [FIN]\n\n"

            return StreamingResponse(gen_single(), media_type="text/event-stream")

        except Exception as e:
            _log_event("chat_stream_compare_error", {"sid": req.session_id, "err": str(e)[:200]})
            return StreamingResponse(_sse_no_data(), media_type="text/event-stream")


    # ---- AGGREGATE → stream de resumen agregado ----
    if plan.intent == "aggregate":
        try:
            t_agg0 = _now_ms()
            agg = aggregate_prices(plan.filters or None, by=plan.group_by or "category")
            t_agg1 = _now_ms()
        except Exception as e:
            _log_event("chat_stream_aggregate_error", {"sid": req.session_id, "err": str(e)[:200]})
            return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

        remember_session(req.session_id, filters=plan.filters, intent="aggregate", query=text, hits=[])
        _log_event("chat_stream_aggregate", {
            "sid": req.session_id,
            "filters": plan.filters,
            "group_by": plan.group_by or "category",
            "rows": agg.get("groups", [])[:10],
        })
        if not agg.get("groups"):
            return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

        ctx = json.dumps({"group_by": plan.group_by or "category",
                        "rows": agg.get("groups", [])}, ensure_ascii=False)
        prompt = (f"Eres {ASSISTANT_NAME}. Redacta en 2–4 frases un resumen del agregado (JSON): {ctx}. "
                "No inventes valores no listados.")

        def gen():
            # VIZ_PROMPT (no cuenta para TTFB del LLM)
            try:
                vizp = _maybe_viz_prompt("aggregate", plan.filters or {}, agg=agg, group_by=plan.group_by or "category")
            except NameError:
                vizp = None
            if vizp:
                yield f"data: [VIZ_PROMPT] {vizp}\n\n"

            yield f"data: {_filters_head(plan.filters)}\n\n"

            # ---- LLM STREAM con TTFB y duración total ----
            t_llm0 = _now_ms()
            first_token_ms = None
            total_chars = 0
            for chunk in llm_chat.stream(prompt):
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

        return StreamingResponse(gen(), media_type="text/event-stream")


   # ---- LOOKUP (por defecto) → stream respuesta amable + contexto ----
    try:
        t_ret0 = _now_ms()
        effective_q = pick_effective_query(text, req.session_id, prefer_last_cat)
        hits = retrieve(effective_q, plan.filters or None)[: plan.top_k or getattr(S, "top_k", 5)]
        if not hits and plan.filters and plan.filters.get("category"):
            f2 = dict(plan.filters); f2.pop("category", None)
            hits = retrieve(effective_q, f2)[: plan.top_k or getattr(S, "top_k", 5)]
        t_ret1 = _now_ms()
    except Exception as e:
        _log_event("chat_stream_lookup_error", {"sid": req.session_id, "err": str(e)[:200]})
        return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

    remember_session(req.session_id, filters=plan.filters, intent="lookup", query=text, hits=hits)
    _log_event("chat_stream_lookup", {
        "sid": req.session_id,
        "filters": plan.filters,
        "effective_query": effective_q,
        "ids": [h.get("product_id") for h in (hits or [])],
    })

    if not hits:
        return StreamingResponse(_sse_no_data(), media_type="text/event-stream")

    ctx = _build_ctx(hits, plan.top_k or getattr(S, "top_k", 5))
    prompt = _prompt_answer_friendly(text, ctx)

    def gen_lookup():
        # VIZ_PROMPT + encabezado (no cuentan para TTFB del LLM)
        try:
            vizp = _maybe_viz_prompt("lookup", plan.filters or {}, rows=hits)
        except NameError:
            vizp = None
        if vizp:
            yield f"data: [VIZ_PROMPT] {vizp}\n\n"

        yield f"data: {_filters_head(plan.filters)}\n\n"

        # ---- LLM STREAM con TTFB y duración total ----
        t_llm0 = _now_ms()
        first_token_ms = None
        total_chars = 0
        for chunk in llm_chat.stream(prompt):
            if first_token_ms is None:
                first_token_ms = _now_ms()
            total_chars += len(chunk)
            yield chunk
        t_llm1 = _now_ms()

        _log_perf("chat_stream_lookup_perf", {
            "gen_model": llm_chat.model,
            "planner_ms": planner_ms, 
            "retrieve_ms": t_ret1 - t_ret0,
            "ttfb_ms": (first_token_ms - t_llm0) if first_token_ms else None,
            "llm_stream_ms": t_llm1 - t_llm0,
            "total_ms": _now_ms() - t_req0,
            "hits": len(hits),
            "ctx_len_chars": len(ctx),
            "top_k": plan.top_k or getattr(S, "top_k", 5),
            "filters": plan.filters,
            "q": text[:120],
            "sid": req.session_id
        })

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
