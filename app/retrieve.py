# app/retrieve.py — Recuperación híbrida (Milvus + BM25) con HF u Ollama para embeddings
from __future__ import annotations
import os, json, re, sqlite3, ast
from typing import Dict, List, Optional, Tuple

from settings import get_settings
from synonyms import detect_category                    # ← sinónimos (rápido/determinista)
from category_resolver import resolve_category_semantic # ← fallback semántico (embeddings)
S = get_settings()

# ---------- Milvus ----------
from pymilvus import connections, Collection

def _milvus() -> Collection:
    host = getattr(S, "milvus_host", "127.0.0.1")
    port = str(getattr(S, "milvus_port", "19530"))
    alias = "default"
    if alias not in connections.list_connections():
        connections.connect(alias=alias, host=host, port=port)
    col_name = getattr(S, "milvus_collection", "products_latam")
    return Collection(col_name)

# ---------- Embeddings ----------
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_HF_TOKENIZER = None
_HF_MODEL = None

def _embed_hf(text: str) -> List[float]:
    """HF: intfloat/multilingual-e5-base (768D) + mean pooling + L2-normalize."""
    global _HF_TOKENIZER, _HF_MODEL
    model_id = getattr(S, "embed_model", "intfloat/multilingual-e5-base")
    if _HF_TOKENIZER is None or _HF_MODEL is None:
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _HF_MODEL = AutoModel.from_pretrained(model_id)
        _HF_MODEL.eval()
    batch = _HF_TOKENIZER([f"query: {text}"], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = _HF_MODEL(**batch)
        last = out.last_hidden_state
        attn = batch["attention_mask"].unsqueeze(-1)
        masked = last.masked_fill(~attn.bool(), 0.0)
        emb = masked.sum(dim=1) / attn.sum(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
    return emb[0].tolist()

def _embed_ollama(text: str) -> List[float]:
    """Ollama embeddings endpoint."""
    try:
        model = getattr(S, "embed_model", "nomic-embed-text")
        base = getattr(S, "ollama_host", "http://127.0.0.1:11434").rstrip("/")
        r = requests.post(f"{base}/api/embeddings", json={"model": model, "prompt": text}, timeout=60)
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception:
        return []

def _embed(text: str) -> List[float]:
    backend = getattr(S, "embed_backend", "hf").lower()
    if backend == "ollama":
        return _embed_ollama(text)
    # default HF
    return _embed_hf(text)

# ---------- Campos ----------
VECTOR_FIELD   = getattr(S, "vector_field", "embedding")
PK_FIELD       = getattr(S, "pk_field", "product_id")
NAME_FIELD     = getattr(S, "name_field", "name")
BRAND_FIELD    = getattr(S, "brand_field", "brand")
CAT_FIELD      = getattr(S, "category_field", "category")
STORE_FIELD    = getattr(S, "store_field", "store")
COUNTRY_FIELD  = getattr(S, "country_field", "country")
PRICE_FIELD    = getattr(S, "price_field", "price")
CURRENCY_FIELD = getattr(S, "currency_field", "currency")
SIZE_FIELD     = getattr(S, "size_field", "size")
UNIT_FIELD     = getattr(S, "unit_field", "unit")
URL_FIELD      = getattr(S, "url_field", "url")
CANON_TEXT     = getattr(S, "canonical_text_field", "canonical_text")

RETURN_FIELDS = [
    PK_FIELD, NAME_FIELD, BRAND_FIELD, CAT_FIELD, STORE_FIELD, COUNTRY_FIELD,
    PRICE_FIELD, CURRENCY_FIELD, SIZE_FIELD, UNIT_FIELD, URL_FIELD, CANON_TEXT
]

# ---------- País: permitir ISO2 o nombres en la BD ----------
_COUNTRY_SYNONYMS = {
    "AR": ["AR", "Argentina"],
    "BR": ["BR", "Brasil", "Brazil"],
    "CO": ["CO", "Colombia"],
    "CR": ["CR", "Costa Rica"],
    "MX": ["MX", "México", "Mexico"],
    "PA": ["PA", "Panamá", "Panama"],
    "PY": ["PY", "Paraguay"],
    "PE": ["PE", "Perú", "Peru"],
    "CL": ["CL", "Chile"],
    "UY": ["UY", "Uruguay"],
}
def _expand_countries(value):
    vals = value if isinstance(value, (list, tuple, set)) else [value]
    out = []
    for v in vals:
        key = str(v).upper()
        out.extend(_COUNTRY_SYNONYMS.get(key, [v]))
    # dedup preservando orden
    seen, res = set(), []
    for v in out:
        if v not in seen:
            seen.add(v); res.append(v)
    return res

# ---------- Utils ----------
def _norm(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")

def _as_list_maybe(val):
    """Acepta 'CO' | ['CO','BR'] | "['CO','BR']" y devuelve lista normalizada."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        return [x for x in val if x not in (None, "", "null")]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                return [x for x in arr if x not in (None, "", "null")]
            except Exception:
                return [s]
        return [s]
    return [val]

def _milvus_expr(filters: Optional[Dict]) -> Optional[str]:
    """
    Construye una expresión Milvus válida, soportando valores str o lista.
    Omite campos vacíos/None. Para listas usa 'in [..]'.
    Además, expande país a sinónimos (ISO2 + nombres).
    """
    if not filters:
        return None

    def _lit(x):
        if isinstance(x, str):
            return '"' + x.replace('"', '\\"') + '"'
        return str(x)

    def _mk(field_name: str, value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            vals = [v for v in value if v not in (None, "", "null")]
            if not vals:
                return None
            vals = list(dict.fromkeys(vals))  # dedup
            if len(vals) == 1:
                v = vals[0]
                return f'{field_name} == {_lit(v)}'
            return f'{field_name} in [{", ".join(_lit(v) for v in vals)}]'
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            return f'{field_name} == {_lit(v)}'
        return f'{field_name} == {value}'

    # Expandir país: acepta lista o string con pinta de lista
    if filters.get("country"):
        filters = dict(filters)  # no mutar el original
        c_list = _as_list_maybe(filters["country"])
        filters["country"] = _expand_countries(c_list)

    exprs = []
    expr = _mk(COUNTRY_FIELD,  filters.get("country"))
    if expr: exprs.append(expr)
    expr = _mk(CAT_FIELD,      filters.get("category"))
    if expr: exprs.append(expr)
    expr = _mk(STORE_FIELD,    filters.get("store"))
    if expr: exprs.append(expr)
    expr = _mk(BRAND_FIELD,    filters.get("brand"))
    if expr: exprs.append(expr)

    return " and ".join(exprs) if exprs else None

def _entity_get(entity, field: str):
    """Compat para PyMilvus Hit.entity: get(field) sin default; fallback a indexing."""
    try:
        return entity.get(field)
    except TypeError:
        return entity.get(field)
    except Exception:
        try:
            return entity[field]
        except Exception:
            return None

# ---------- BM25 / Keyword (opcional) ----------
def _fts_conn() -> Optional[sqlite3.Connection]:
    db_path = getattr(S, "fts_db_path", None)
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        return sqlite3.connect(db_path)
    except Exception:
        return None

def _first_scalar(v):
    """Para FTS/BM25: si el filtro viene como lista, toma el primero."""
    if isinstance(v, (list, tuple, set)):
        return next(iter(v), None)
    return v

def _bm25_search_sqlite(query: str, filters: Optional[Dict], top_k: int=50) -> List[Tuple[str, float]]:
    conn = _fts_conn()
    if not conn:
        return []
    nt = _norm(query)
    q = " ".join(t for t in re.split(r"\W+", nt) if t)
    if not q:
        return []
    where = " text MATCH ? "; args = [q]
    if filters:
        ctry = _first_scalar(filters.get("country"))
        cat  = _first_scalar(filters.get("category"))
        store= _first_scalar(filters.get("store"))
        brand= _first_scalar(filters.get("brand"))
        if ctry:  where += " AND country = ?";  args.append(ctry)
        if cat:   where += " AND category = ?"; args.append(cat)
        if store: where += " AND store = ?";    args.append(store)
        if brand: where += " AND brand = ?";    args.append(brand)
    sql = f"""
        SELECT docid, bm25(products_fts) AS score
        FROM products_fts
        WHERE {where}
        ORDER BY score ASC
        LIMIT {int(top_k)}
    """
    try:
        cur = conn.cursor()
        cur.execute(sql, args)
        return [(str(docid), float(1.0 / (1e-9 + score))) for (docid, score) in cur.fetchall()]
    except Exception:
        return []

_CORPUS_CACHE = None
def _load_corpus_cache() -> List[Dict]:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE
    path = getattr(S, "catalog_jsonl", "data/catalog.jsonl")
    items = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try: items.append(json.loads(line))
                except Exception: continue
    _CORPUS_CACHE = items
    return _CORPUS_CACHE

def _match_val(doc_v, flt_v):
    if isinstance(flt_v, (list, tuple, set)):
        return doc_v in flt_v
    return doc_v == flt_v

def _bm25_search_fallback(query: str, filters: Optional[Dict], top_k: int=50) -> List[Tuple[str, float]]:
    docs = _load_corpus_cache()
    if not docs:
        return []
    nt = _norm(query)
    tokens = [t for t in re.split(r"\W+", nt) if t]
    if not tokens:
        return []
    scored = []
    for d in docs:
        if filters:
            ctry = filters.get("country")
            cat  = filters.get("category")
            store= filters.get("store")
            brand= filters.get("brand")
            if ctry  and not _match_val(d.get(COUNTRY_FIELD), ctry): continue
            if cat   and not _match_val(d.get(CAT_FIELD),      cat):  continue
            if store and not _match_val(d.get(STORE_FIELD),    store):continue
            if brand and not _match_val(d.get(BRAND_FIELD),    brand):continue
        text = " ".join([str(d.get(NAME_FIELD,"")), str(d.get(BRAND_FIELD,"")), str(d.get(CAT_FIELD,"")), str(d.get(STORE_FIELD,""))])
        ntxt = _norm(text)
        tf = sum(ntxt.count(tok) for tok in tokens)
        if tf > 0:
            scored.append((str(d.get(PK_FIELD)), float(tf)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def _bm25_search(query: str, filters: Optional[Dict], top_k: int=50) -> List[Tuple[str, float]]:
    hits = _bm25_search_sqlite(query, filters, top_k)
    if hits: return hits
    return _bm25_search_fallback(query, filters, top_k)

# ---------- Dense (Milvus vector) ----------
def _dense_search(query: str, filters: Optional[Dict], top_k: int = 50) -> List[Dict]:
    vec = _embed(query)
    if not vec:
        return []
    col = _milvus()
    col.load()

    # Descubrir métrica del índice (antes de loggear)
    metric = getattr(S, "metric_type", "IP")
    try:
        for idx in col.indexes:
            if getattr(idx, "field_name", None) == VECTOR_FIELD:
                params = getattr(idx, "params", {}) or {}
                if isinstance(params, str):
                    try: params = json.loads(params)
                    except Exception: params = {}
                mt = params.get("metric_type")
                if mt:
                    metric = str(mt).upper()
                break
    except Exception:
        pass

    # Validación de dimensión
    dim = 0
    try:
        for f in col.schema.fields:
            if f.name == VECTOR_FIELD and hasattr(f, "params") and isinstance(f.params, dict):
                dim = int(f.params.get("dim") or f.params.get("max_length") or 0)
                break
    except Exception:
        dim = 0
    if dim and len(vec) != dim:
        # log dimension mismatch
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/retrieve.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "dim_mismatch", "expected": dim, "got": len(vec),
                    "collection": col.name, "vector_field": VECTOR_FIELD
                }) + "\n")
        except Exception:
            pass
        return []

    # Expr final (con expansión de país)
    expr = _milvus_expr(filters)

    # --- Pre-chequeos: ¿qué filtro está vaciando el conjunto? ---
    has_country = has_category = has_both = False
    try:
        if filters and filters.get("country"):
            expr_country = _milvus_expr({"country": filters.get("country")})
            rows = col.query(expr=expr_country, output_fields=[PK_FIELD], limit=1)
            has_country = len(rows) > 0
    except Exception:
        pass
    try:
        if filters and filters.get("category"):
            expr_category = _milvus_expr({"category": filters.get("category")})
            rows = col.query(expr=expr_category, output_fields=[PK_FIELD], limit=1)
            has_category = len(rows) > 0
    except Exception:
        pass
    try:
        if expr:
            rows = col.query(expr=expr, output_fields=[PK_FIELD], limit=1)
            has_both = len(rows) > 0
    except Exception:
        pass

    # Log de pre-chequeo
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/retrieve.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "kind": "dense_precheck",
                "collection": col.name,
                "metric": metric,
                "dim": len(vec),
                "query": query,
                "filters_in": filters,
                "expr_final": expr,
                "has_country": has_country,
                "has_category": has_category,
                "has_both": has_both
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # Relajar categoría si país tiene datos pero la intersección queda vacía
    if (filters and filters.get("category")
        and has_country and not has_both):
        relaxed = dict(filters)
        relaxed.pop("category", None)
        expr = _milvus_expr(relaxed)
        try:
            with open("logs/retrieve.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": "dense_relax_category", "relaxed_expr": expr,
                    "relaxed_filters": relaxed
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # --- Búsqueda vectorial ---
    try:
        # Usa ef para HNSW y nprobe para IVF/IVF_FLAT
        index_type = str(getattr(S, "index_type", "")).upper()  # pon "HNSW" en .env si aplica
        if index_type == "HNSW":
            search_param = {"metric_type": metric, "params": {"ef": int(getattr(S, "ef_search", 96))}}
        else:
            search_param = {"metric_type": metric, "params": {"nprobe": int(getattr(S, "nprobe", 48))}}

        res = col.search(
            data=[vec],
            anns_field=VECTOR_FIELD,
            param=search_param,
            limit=top_k,
            expr=expr,
            output_fields=RETURN_FIELDS
        )

    except Exception:
        return []

    out = []
    # Para L2, menor distancia = mejor; para IP/COSINE, mayor = mejor. Invertimos L2 para homogeneizar “score”.
    invert = (metric.upper() == "L2")
    for h in res[0]:
        drow = {}
        for f in RETURN_FIELDS:
            drow[f] = _entity_get(h.entity, f)
        raw = float(h.distance)
        drow["_score"] = (-raw if invert else raw)
        out.append(drow)

    # Log de hits
    try:
        with open("logs/retrieve.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "kind":"dense_hits","count": len(out),
                "sample_ids":[d.get(PK_FIELD) for d in out[:5]]
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return out

# ---------- Fetch ----------
def _fetch_by_ids(ids: List[str]) -> List[Dict]:
    if not ids:
        return []
    col = _milvus()
    expr = f'{PK_FIELD} in {ids!r}'
    try:
        ret = col.query(expr=expr, output_fields=RETURN_FIELDS)
    except Exception:
        return []
    m = {str(r[PK_FIELD]): r for r in ret}
    out = []
    for i in ids:
        r = m.get(str(i))
        if r:
            out.append({
                "score": None,
                "product_id": str(r.get(PK_FIELD)),
                "name": r.get(NAME_FIELD),
                "brand": r.get(BRAND_FIELD),
                "category": r.get(CAT_FIELD),
                "store": r.get(STORE_FIELD),
                "country": r.get(COUNTRY_FIELD),
                "price": r.get(PRICE_FIELD),
                "currency": r.get(CURRENCY_FIELD),
                "unit": r.get(UNIT_FIELD),
                "size": r.get(SIZE_FIELD),
                "url": r.get(URL_FIELD),
                "canonical_text": r.get(CANON_TEXT),
            })
    return out

# ---------- RRF ----------
def _rrf_merge(
    dense: List[Tuple[str, float]],
    bm25: List[Tuple[str, float]],
    rrf_k: float = 60.0,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    final_top_k: int = 10
) -> List[Tuple[str, float]]:

    ranks_dense = {pid: i for i, (pid, _) in enumerate(dense, start=1)}
    ranks_bm25  = {pid: i for i, (pid, _) in enumerate(bm25,  start=1)}
    pids = set(ranks_dense) | set(ranks_bm25)
    fused = []
    for pid in pids:
        rd = ranks_dense.get(pid)
        rb = ranks_bm25.get(pid)
        s = 0.0
        if rd is not None: s += w_dense * (1.0 / (rrf_k + rd))
        if rb is not None: s += w_bm25 * (1.0 / (rrf_k + rb))
        fused.append((pid, s))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:final_top_k]

# ---------- LLM re-rank (opcional) ----------
def _llm_rerank(pairs: List[Tuple[str, Dict]], query: str) -> List[Tuple[str, float]]:
    if not pairs:
        return []
    try:
        base_url = getattr(S, "ollama_host", "http://127.0.0.1:11434").rstrip("/")
        model = getattr(S, "gen_model", "phi3:mini")
        items_txt = "\n".join(
            f"- [{pid}] {d.get('name','')} | Marca: {d.get('brand','')} | {d.get('size','')}{d.get('unit','')} | {d.get('price','')} {d.get('currency','')} | {d.get('store','')} | {d.get('country','')}"
            for pid, d in pairs
        )
        prompt = (
            "Califica de 0 a 3 (0 = nada relevante, 3 = muy relevante) cada producto respecto a la consulta.\n"
            "Devuelve SOLO JSON con lista de objetos: [{\"product_id\":\"...\",\"score\":N}, ...]\n"
            f"Consulta: {query}\nProductos:\n{items_txt}\n\nJSON:"
        )
        r = requests.post(f"{base_url}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 1024, "num_predict": 128}
        }, timeout=60)
        r.raise_for_status()
        txt = (r.json().get("response") or "").strip()
        m = re.search(r"\[.*\]", txt, re.S)
        if not m:
            return [(pid, 0.0) for pid, _ in pairs]
        data = json.loads(m.group(0))
        mapp = {str(x["product_id"]): float(x.get("score", 0.0)) for x in data if "product_id" in x}
        out = []
        for pid, _ in pairs:
            out.append((pid, mapp.get(pid, 0.0)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception:
        return [(pid, float(len(pairs)-i)) for i, (pid, _) in enumerate(pairs)]

# =========================
#  API PÚBLICA
# =========================
def retrieve(query: str, filters: Optional[Dict]=None) -> List[Dict]:
    # -------- Normalización de intención → categoría canónica --------
    if filters is None:
        filters = {}

    # ⚠️ No infieras categoría cuando la consulta sea genérica (“precios” + “productos”)
    ntq = _norm(query)
    is_generic = (("precio" in ntq or "precios" in ntq) and ("producto" in ntq or "productos" in ntq))

    # 1) Sinónimos rápidos (determinista)
    if not is_generic and ("category" not in filters or not filters.get("category")):
        inferred = detect_category(query)
        if inferred:
            filters["category"] = inferred

    # 2) Fallback semántico por embeddings si aún no hay categoría
    if not is_generic and ("category" not in filters or not filters.get("category")):
        cat_sem, score_sem, _ = resolve_category_semantic(
            query, min_cosine=float(getattr(S, "category_min_cosine", 0.60))
        )
        if cat_sem:
            filters["category"] = cat_sem

    top_k = int(getattr(S, "hybrid_top_k", 30))
    final_k = int(getattr(S, "final_top_k", 10))
    rrf_k  = float(getattr(S, "rrf_k", 60.0))
    w_dense = float(getattr(S, "dense_weight", 1.0))
    w_bm25  = float(getattr(S, "bm25_weight", 1.0))
    bm25_on = bool(getattr(S, "bm25_enabled", False))
    llm_rerank_on = bool(getattr(S, "llm_rerank_enabled", False))
    llm_rerank_top_n = int(getattr(S, "llm_rerank_top_n", 12))

    # 1) Dense
    dense_dicts = _dense_search(query, filters, top_k=top_k)
    dense_pairs = [(str(d.get(PK_FIELD)), float(d.get("_score", 0.0)))
                   for d in dense_dicts if d.get(PK_FIELD) is not None]

    # 2) BM25
    bm25_pairs = _bm25_search(query, filters, top_k=top_k) if bm25_on else []

    # 3) Fusión
    fused = _rrf_merge(dense_pairs, bm25_pairs, rrf_k=rrf_k, w_dense=w_dense, w_bm25=w_bm25,
                       final_top_k=max(final_k, llm_rerank_top_n))

    # 4) Recuperar filas completas
    fused_ids = [pid for pid, _ in fused]
    rows = _fetch_by_ids(fused_ids)
    m = {r["product_id"]: r for r in rows}
    pairs_full = [(pid, m.get(pid)) for pid in fused_ids if m.get(pid)]

    # 5) Rerank opcional
    if llm_rerank_on and pairs_full:
        subset = pairs_full[:llm_rerank_top_n]
        rr = _llm_rerank(subset, query)
        order = {pid: i for i, (pid, _) in enumerate(rr)}
        pairs_full.sort(key=lambda x: order.get(x[0], 1e9))

    # 6) Salida final con score de fusión
    score_by = {pid: s for pid, s in fused}
    out = []
    for pid, d in pairs_full[:final_k]:
        r = dict(d)
        r["score"] = float(score_by.get(pid, 0.0))
        out.append(r)
    return out

def list_by_filter(filters: Optional[Dict]=None, limit: int=100) -> List[Dict]:
    col = _milvus()
    col.load()
    expr = _milvus_expr(filters)
    try:
        ret = col.query(expr=expr, output_fields=RETURN_FIELDS, limit=limit)
    except Exception:
        return []
    rows = []
    for r in ret:
        rows.append({
            "score": None,
            "product_id": str(r.get(PK_FIELD)),
            "name": r.get(NAME_FIELD),
            "brand": r.get(BRAND_FIELD),
            "category": r.get(CAT_FIELD),
            "store": r.get(STORE_FIELD),
            "country": r.get(COUNTRY_FIELD),
            "price": r.get(PRICE_FIELD),
            "currency": r.get(CURRENCY_FIELD),
            "unit": r.get(UNIT_FIELD),
            "size": r.get(SIZE_FIELD),
            "url": r.get(URL_FIELD),
            "canonical_text": r.get(CANON_TEXT),
        })
    return rows[:limit]

def aggregate_prices(filters: Optional[Dict]=None, by: Optional[str]=None) -> Dict:
    rows = list_by_filter(filters, limit=int(getattr(S, "aggregate_limit", 5000)))
    if not rows:
        return {"groups": []}
    by = by or "category"
    groups: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        key = str(r.get(by)) if r.get(by) is not None else "N/A"
        price = r.get("price")
        if price is None:
            continue
        groups.setdefault(key, {"min": float(price), "max": float(price), "avg": 0.0, "group": key})
        counts[key] = counts.get(key, 0) + 1
        g = groups[key]
        if float(price) < g["min"]: g["min"] = float(price)
        if float(price) > g["max"]: g["max"] = float(price)
        g["avg"] += float(price)
    out = []
    for k, g in groups.items():
        n = counts[k]
        g["avg"] = g["avg"] / max(n, 1)
        g["n"] = n
        out.append(g)
    out.sort(key=lambda x: x["group"])
    return {"groups": out}
