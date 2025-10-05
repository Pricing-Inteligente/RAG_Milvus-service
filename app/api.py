# app/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal
import requests, re, json, unicodedata

# === Config ===
from settings import get_settings
S = get_settings()

# === Milvus helpers (tus utilidades) ===
from retrieve import retrieve, list_by_filter, aggregate_prices

# -----------------------------------------------------------------------------
# Utilidades de normalización y alias (tildes/mayúsculas → canónico)
# -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    # minúsculas + sin tildes/diacríticos
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
    "CR": ["cr", "costa rica", "costarica"],
    "PA": ["pa", "panama", "panamá"],
    "PY": ["py", "paraguay"],
}
NCOUNTRIES = { _norm(alias): code
               for code, aliases in COUNTRY_ALIASES.items()
               for alias in aliases }

CAT_ALIASES = {
    "azucar":   ["azucar", "azúcar"],
    "arroz":    ["arroz"],
    "leche":    ["leche"],
    "tomate":   ["tomate"],
    "aceite":   ["aceite"],
    "huevo":    ["huevo", "huevos"],
    "pan":      ["pan"],
    "atun":     ["atun", "atún"],
    "galletas": ["galletas"],
    "bebidas":  ["bebidas"],
    "lacteos":  ["lacteos", "lácteos"],
    "pasta":    ["pasta"],
    "legumbres":["legumbres"],
    "aseo":     ["aseo"],
}
NCATEGORIES = { _norm(alias): canon
                for canon, aliases in CAT_ALIASES.items()
                for alias in aliases }

STORE_ALIASES = {
    "Exito": ["exito", "éxito"],
    "Jumbo": ["jumbo"],
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
    "Assai": ["assai"],
    "PaoDeAcucar": ["pao de acucar", "pao de açúcar", "paodeacucar"],
}
NSTORES = { _norm(alias): canon
            for canon, aliases in STORE_ALIASES.items()
            for alias in aliases }

def sanitize_filters(f: Dict | None) -> Dict:
    """Normaliza filtros (country ISO, category y store canónicos) e ignora tildes/mayúsculas."""
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

    # deja pasar otros filtros tal cual (p.ej., brand, name)
    for k in ["brand", "name"]:
        if k in f:
            out[k] = f[k]
    return out

# -----------------------------------------------------------------------------
# CORS
# -----------------------------------------------------------------------------

app = FastAPI(title="RAG Pricing API", version="1.3.0")

origins = [
    "http://localhost:5173",  # tu frontend local
    "http://127.0.0.1:5173",  # a veces Vite usa 127.0.0.1
    "http://localhost:8080",  # Lovable local
]

# Ahora S.cors_origins ya es lista (gracias a settings.py)
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
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.1,
        num_ctx: int = 2048,
        num_predict: int = 256,
        timeout: int = 120,
    ):
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
            return ""  # activa abstención

    def stream(self, prompt: str):
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
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "response" in chunk:
                    yield f"data: {chunk['response']}\n\n"
                if chunk.get("done"):
                    break
            except Exception:
                continue

# LLM para respuesta (redacción)
llm = OllamaLLM(
    model=getattr(S, "gen_model", "phi3:mini"),
    base_url=getattr(S, "ollama_host", "http://127.0.0.1:11434"),
    temperature=0.1,
    num_ctx=1024,
    num_predict=128,
)

# Helper: llamada al LLM con temp=0 para *planner*
def _llm_json(prompt: str) -> str:
    old = llm.temperature
    try:
        llm.temperature = 0.0
        return llm.generate(prompt)
    finally:
        llm.temperature = old

# -----------------------------------------------------------------------------
# Raíz / salud
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
# /ask  (QA con RAG, read-only)
# -----------------------------------------------------------------------------
class AskReq(BaseModel):
    question: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = None
    abstain_threshold: Optional[float] = None  # por si lo quieres tunear por request

def _build_ctx(hits: List[Dict], k: int) -> str:
    return "\n".join(
        f"[{h['product_id']}] {h['name']} | Marca: {h['brand']} | "
        f"Pres: {h['size']}{h['unit']} | Precio: {h['price']} {h['currency']} | "
        f"Tienda: {h['store']} | País: {h['country']}"
        for h in hits[:k]
    )

def _prompt_answer(question: str, ctx: str) -> str:
    return (
        "Eres un asistente de retail. SOLO puedes usar los datos del CONTEXTO.\n"
        "Si el CONTEXTO no contiene la respuesta exacta, responde exactamente:\n"
        "\"No tengo esa información en la base\".\n"
        "Responde en español, breve y conversacional. Cita el/los [product_id] usados.\n\n"
        f"CONTEXTO:\n{ctx}\n\n"
        f"PREGUNTA: {question}\n"
        "RESPUESTA:"
    )

@app.post("/ask", tags=["rag"])
def ask(req: AskReq):
    top_k = req.top_k or getattr(S, "top_k", 5)

    # Normaliza filtros por si vienen desde el front con mayúsculas/tildes
    flt = sanitize_filters(req.filters)

    # Recupera evidencia (tu retrieve usa Milvus)
    hits: List[Dict] = retrieve(req.question, flt)
    if not hits:
        return {"answer": "No tengo esa información en la base", "evidence": []}

    ctx = _build_ctx(hits, top_k)
    prompt = _prompt_answer(req.question, ctx)

    txt = llm.generate(prompt)
    ids = re.findall(r"\[(.*?)\]", txt)  # exige citar product_id
    if not txt or not ids:
        return {"answer": "No tengo esa información en la base", "evidence": []}

    # Evidencia solo de los IDs citados (orden único)
    ids = list(dict.fromkeys(ids))
    ev = [h for h in hits if h["product_id"] in ids]
    if not ev:
        return {"answer": "No tengo esa información en la base", "evidence": []}

    return {"answer": txt, "evidence": ev, "top_k_used": top_k}

# Streaming (SSE) para UX de chat
@app.post("/ask/stream", tags=["rag"])
def ask_stream(req: AskReq):
    top_k = req.top_k or getattr(S, "top_k", 5)
    flt = sanitize_filters(req.filters)
    hits: List[Dict] = retrieve(req.question, flt)
    if not hits:
        def gen_no_data():
            yield "data: No tengo esa información en la base\n\n"
        return StreamingResponse(gen_no_data(), media_type="text/event-stream")

    ctx = _build_ctx(hits, top_k)
    prompt = _prompt_answer(req.question, ctx)
    return StreamingResponse(llm.stream(prompt), media_type="text/event-stream")

# -----------------------------------------------------------------------------
# /list  (consulta directa sin LLM)
# -----------------------------------------------------------------------------
class ListReq(BaseModel):
    filters: Optional[Dict] = None
    limit: Optional[int] = 100

@app.post("/list", tags=["products"])
def list_products(req: ListReq):
    lim = max(1, min(req.limit or 100, 1000))
    rows = list_by_filter(sanitize_filters(req.filters), limit=lim)
    return {"count": len(rows), "items": rows}

# -----------------------------------------------------------------------------
# /aggregate  (min/máx/promedio) — solo lectura
# -----------------------------------------------------------------------------
class AggregateReq(BaseModel):
    filters: Optional[Dict] = None
    group_by: Optional[Literal["store", "category", "country"]] = None
    operation: Optional[Literal["min", "max", "avg"]] = None

@app.post("/aggregate", tags=["products"])
def aggregate(req: AggregateReq):
    result = aggregate_prices(sanitize_filters(req.filters), by=req.group_by)
    if req.operation and result.get("groups"):
        for g in result["groups"]:
            k = req.operation
            for m in ["min", "max", "avg"]:
                if m != k and m in g:
                    del g[m]
    return result

# -----------------------------------------------------------------------------
# /chat  (Planner → Executor → Answerer). LLM-First + normalización + fallback.
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

def _contains_any(nt: str, patterns: List[str]) -> bool:
    for p in patterns:
        if " " in p:
            if p in nt:
                return True
        else:
            if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", nt):
                return True
    return False

def _classify_intent_heuristic(text: str) -> str:
    nt = _norm(text)
    LIST_SYNS = [
        "lista", "listado", "listar", "listame", "muestrame", "ensename",
        "enviame", "pasame", "traeme", "quiero ver", "mostrar", "muestra",
        "ver todos", "todos los productos", "dame todos", "dame todo"
    ]
    AGG_SYNS = [
        "promedio", "media", "minimo", "maximo", "promedio por", "promedios por",
        "agrupa", "distribucion", "rango", "por tienda", "por pais", "por categoria"
    ]
    COUNT_SYNS = ["cuantos", "cuantos hay", "numero de", "cantidad", "total de", "cuantas", "cuantos productos"]
    COMPARE_SYNS = ["comparar", "compara", "comparacion", "vs", "contra", "frente a"]

    if _contains_any(nt, COMPARE_SYNS): return "compare"
    if _contains_any(nt, AGG_SYNS):     return "aggregate"
    if _contains_any(nt, LIST_SYNS):    return "list"
    if _contains_any(nt, COUNT_SYNS):   return "count"
    return "lookup"

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
        ("Envíame todos los productos de México",
         {"intent":"list","filters":{"country":"MX"}}),
        ("muéstrame los lácteos en peru",
         {"intent":"list","filters":{"country":"PE","category":"lacteos"}}),
        ("aceite vegetal 900ml en argentina",
         {"intent":"lookup","filters":{"country":"AR","category":"aceite"}}),
        ("¿cuántos productos hay en chile?",
         {"intent":"count","filters":{"country":"CL"}}),
        ("promedio de precios por país para arroz",
         {"intent":"aggregate","group_by":"country","filters":{"category":"arroz"}}),
        ("compara leche entera 1l vs arroz blanco 1kg en ecuador",
         {"intent":"compare","product_name":"leche entera 1l","product_name_b":"arroz blanco 1kg","filters":{"country":"EC"}}),
    ]

    prompt = (
        "Devuelve SOLO un JSON que cumpla exactamente este esquema, sin texto extra.\n"
        f"Esquema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        "Reglas:\n"
        "- Normaliza el país a códigos ISO de esta lista: MX, BR, AR, CO, CL, PE, EC, CR, PA, PY.\n"
        "- category usa estos canónicos: azucar, arroz, leche, tomate, aceite, huevo, pan, atun, galletas, bebidas, lacteos, pasta, legumbres, aseo.\n"
        "- store devuelve el nombre canónico si lo reconoces (Exito, Jumbo, Olimpica, Carulla, Ara, D1, Walmart, Soriana, Chedraui, Lider, Wong, Metro, Tottus, Carrefour, Assai, PaoDeAcucar); si no, déjalo vacío o omítelo.\n"
        "- Si no estás seguro de algún campo, pon null o deja filters vacío.\n\n"
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
        # Normaliza filtros a canónico
        data["filters"] = sanitize_filters(data["filters"])
        return Plan(**data)
    except Exception:
        return None

class ChatReq(BaseModel):
    message: str
    limit: Optional[int] = 100

# --- Helper para adjuntar metadata de modelo y plan en todas las respuestas de /chat
def with_meta(payload: dict, plan: Plan) -> dict:
    payload["model"] = llm.model
    payload["planner"] = {
        "intent": plan.intent,
        "filters": plan.filters,
        "model": llm.model,
    }
    # 👇 Alias para el front: si hay 'reply' y no hay 'message', cópialo.
    if "reply" in payload and "message" not in payload:
        payload["message"] = payload["reply"]
    return payload


@app.post("/chat", tags=["chat"])
def chat(req: ChatReq):
    text = req.message.strip()

    # 1) Planner LLM (JSON) + 2) Heurística + 3) Normalización + Fallback
    plan = _plan_from_llm(text)
    heur = _guess_filters(text)

    if plan:
        plan.filters = plan.filters or {}
        for k, v in heur.items():
            plan.filters.setdefault(k, v)
        plan.filters = sanitize_filters(plan.filters)
    else:
        plan = Plan(
            intent=_classify_intent_heuristic(text),
            filters=sanitize_filters(heur),
            top_k=getattr(S, "top_k", 5),
            limit=min(max(req.limit or 100, 1), 1000),
        )

    # ---- EXECUTOR ----
    if plan.intent == "list":
        items = list_by_filter(plan.filters or None, limit=min(max(plan.limit or 100, 1), 1000))
        if not items:
            return with_meta({"type":"table","reply":"No tengo esa información en la base","count":0,"items":[]}, plan)
        return with_meta({"type":"table","reply":f"Encontré {len(items)} producto(s).","count":len(items),"items":items}, plan)

    if plan.intent == "count":
        items = list_by_filter(plan.filters or None, limit=1000)
        return with_meta({"type":"text","reply":f"Tengo {len(items)} registro(s) que cumplen ese filtro.","evidence":[]}, plan)

    if plan.intent == "aggregate":
        if not plan.group_by:
            nt = _norm(text)
            if "tienda" in nt: plan.group_by = "store"
            elif "categor" in nt: plan.group_by = "category"
            elif "pais" in nt: plan.group_by = "country"
        agg = aggregate_prices(plan.filters or None, by=plan.group_by)
        if not agg.get("groups"):
            return with_meta({"type":"text","reply":"No tengo esa información en la base","evidence":[]}, plan)
        if plan.operation:
            for g in agg["groups"]:
                for m in ["min","max","avg"]:
                    if m != plan.operation and m in g:
                        del g[m]
        return with_meta({"type":"aggregate","reply":"Resumen de precios.","result":agg}, plan)

    if plan.intent == "compare":
        if not (plan.product_name and plan.product_name_b):
            return with_meta({"type":"text","reply":"Necesito dos productos para comparar.","evidence":[]}, plan)
        hits_a = retrieve(plan.product_name, plan.filters or None)[:3]
        hits_b = retrieve(plan.product_name_b, plan.filters or None)[:3]
        if not hits_a or not hits_b:
            return with_meta({"type":"text","reply":"No tengo esa información en la base para comparar.","evidence":[]}, plan)
        ctx_lines = []
        for h in hits_a[:2] + hits_b[:2]:
            ctx_lines.append(f"[{h['product_id']}] {h['name']} | {h['price']} {h['currency']} | {h['store']} | {h['country']}")
        prompt = (
            "Compara SOLO los productos del CONTEXTO (precio y presentación). "
            "Si no es concluyente, responde exactamente: \"No tengo esa información en la base\".\n"
            "Responde en español y cita [product_id].\n\n"
            f"CONTEXTO:\n{chr(10).join(ctx_lines)}\n\nPREGUNTA: {text}\nRESPUESTA:"
        )
        txt = llm.generate(prompt)
        ids = re.findall(r"\[(.*?)\]", txt)
        ev = [h for h in (hits_a + hits_b) if h["product_id"] in ids]
        if not txt or not ev:
            return with_meta({"type":"text","reply":"No tengo esa información en la base","evidence":[]}, plan)
        return with_meta({"type":"text","reply":txt,"evidence":ev}, plan)

    # default: lookup
    hits = retrieve(text if not plan.product_name else plan.product_name,
                    plan.filters or None)[: plan.top_k or getattr(S, "top_k", 5)]
    if not hits:
        return with_meta({"type":"text","reply":"No tengo esa información en la base","evidence":[]}, plan)
    ctx = _build_ctx(hits, plan.top_k or getattr(S, "top_k", 5))
    prompt = (
        "Eres un asistente de retail. SOLO puedes usar el CONTEXTO.\n"
        "Si el CONTEXTO no contiene la respuesta exacta, responde exactamente:\n"
        "\"No tengo esa información en la base\".\n"
        "Responde en español y cita [product_id].\n\n"
        f"CONTEXTO:\n{ctx}\n\nPREGUNTA: {text}\nRESPUESTA:"
    )
    txt = llm.generate(prompt)
    ids = re.findall(r"\[(.*?)\]", txt)
    ev = [h for h in hits if h["product_id"] in ids]
    if not txt or not ev:
        return with_meta({"type":"text","reply":"No tengo esa información en la base","evidence":[]}, plan)
    return with_meta({"type":"text","reply":txt,"evidence":ev}, plan)
