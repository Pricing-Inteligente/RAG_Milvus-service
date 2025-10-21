# app/category_resolver.py — Resolver de categoría por similitud semántica (E5)
from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from settings import get_settings
S = get_settings()

from pymilvus import connections, Collection
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

_HF_TOKENIZER = None
_HF_MODEL = None

def _embed(texts: List[str]) -> torch.Tensor:
    global _HF_TOKENIZER, _HF_MODEL
    model_id = getattr(S, "embed_model", "intfloat/multilingual-e5-base")
    if _HF_TOKENIZER is None or _HF_MODEL is None:
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _HF_MODEL = AutoModel.from_pretrained(model_id)
        _HF_MODEL.eval()
    batch = _HF_TOKENIZER([f"query: {t}" for t in texts], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = _HF_MODEL(**batch)
        last = out.last_hidden_state
        attn = batch["attention_mask"].unsqueeze(-1)
        masked = last.masked_fill(~attn.bool(), 0.0)
        emb = masked.sum(dim=1) / attn.sum(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
    return emb  # [n, 768]

def _milvus_col() -> Collection:
    host = getattr(S, "milvus_host", "127.0.0.1")
    port = str(getattr(S, "milvus_port", "19530"))
    alias = "default"
    if alias not in connections.list_connections():
        connections.connect(alias=alias, host=host, port=port)
    return Collection(getattr(S, "milvus_collection", "retail_products"))

def list_distinct_categories(limit: int = 1000) -> List[str]:
    """Obtiene las categorías distintas presentes en la colección (best-effort)."""
    col = _milvus_col()
    col.load()
    rows = col.query(expr="", output_fields=[getattr(S, "category_field", "category")], limit=limit)
    cats = sorted({r.get(getattr(S,"category_field","category")) for r in rows if r.get(getattr(S,"category_field","category"))})
    return cats

def resolve_category_semantic(query: str, cats: Optional[List[str]] = None, min_cosine: float = 0.35) -> Tuple[Optional[str], float, Dict[str,float]]:
    """Devuelve (categoria, score, scores_por_categoria). Usa cosine con E5."""
    cats = cats or list_distinct_categories()
    if not cats: 
        return (None, 0.0, {})
    em_q = _embed([query])[0]          # [768]
    em_c = _embed(cats)                # [C, 768]
    scores = (em_q @ em_c.T).squeeze(0)  # cosine porque ya están normalizados
    best_idx = int(torch.argmax(scores).item())
    best_cat = cats[best_idx]
    best_score = float(scores[best_idx].item())
    details = {cats[i]: float(scores[i].item()) for i in range(len(cats))}
    if best_score < min_cosine:
        return (None, best_score, details)
    return (best_cat, best_score, details)
