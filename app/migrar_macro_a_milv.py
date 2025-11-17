# migrar_macro_a_milv.py  — PG -> Milvus con embeddings E5 reales (768D)
from __future__ import annotations
import os, hashlib
from datetime import datetime
from typing import List, Tuple

from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import psycopg2

# --------- Config ---------
def env(k, d=None): return os.getenv(k, d)

MILVUS_HOST = env("MILVUS_HOST", env("milvus_host", "127.0.0.1"))
MILVUS_PORT = env("MILVUS_PORT", env("milvus_port", "19530"))
COLL_NAME   = env("MILVUS_MACRO_COLLECTION", "macro_latam")

PG_HOST = env("PG_HOST"); PG_PORT = int(env("PG_PORT", "5432"))
PG_DB   = env("PG_DB");   PG_USER = env("PG_USER"); PG_PASS = env("PG_PASSWORD")

TAB       = env("PG_MACRO_TABLE", "tabla_macroeconomicas")
F_PAIS    = env("PG_MACRO_COUNTRY", "pais")
F_VAR     = env("PG_MACRO_VARIABLE", "variables")
F_UNIDAD  = env("PG_MACRO_UNIT", "unidad")
F_FECHA   = env("PG_MACRO_DATE", "fecha")
F_VALOR   = env("PG_MACRO_VALUE", "valor_interp")
F_ANIO    = env("PG_MACRO_YEAR", "anio")
F_MES     = env("PG_MACRO_MONTH", "mes")

EMBED_MODEL = env("MACRO_EMBED_MODEL", "intfloat/multilingual-e5-base")  # mismo que productos por defecto

COUNTRY_TO_ISO = {
    "argentina":"AR","colombia":"CO","mexico":"MX","méxico":"MX","brasil":"BR","brazil":"BR",
    "chile":"CL","paraguay":"PY","peru":"PE","perú":"PE","ecuador":"EC","panama":"PA","panamá":"PA",
    "costa rica":"CR","uruguay":"UY","bolivia":"BO"
}

# --------- Helpers ---------
def to_iso2(raw: str) -> str:
    if not raw: return ""
    k = (raw or "").strip().lower()
    return COUNTRY_TO_ISO.get(k, raw.strip())

def norm_date(date_val, year, month) -> str:
    if date_val:
        s = str(date_val)
        for fmt in ("%Y-%m-%d","%Y/%m/%d","%Y%m%d","%Y-%m","%Y/%m","%Y%m"):
            try:
                d = datetime.strptime(s[:10], fmt)
                return d.strftime("%Y-%m-%d" if fmt in ("%Y-%m-%d","%Y/%m/%d","%Y%m%d") else "%Y-%m")
            except Exception:
                pass
        return s
    if year and month:
        try:    return f"{int(year):04d}-{int(month):02d}"
        except: return f"{year}-{month}"
    return ""

def make_id(iso2: str, var: str, d: str) -> str:
    base = f"{iso2}|{(var or '').strip().lower()}|{d}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def fetch_rows() -> List[Tuple]:
    cols = f'"{F_PAIS}","{F_VAR}","{F_UNIDAD}","{F_FECHA}","{F_VALOR}","{F_ANIO}","{F_MES}"'
    sql  = f'SELECT {cols} FROM "{TAB}"'
    with psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

def ensure_collection(dim: int = 768) -> Collection:
    if utility.has_collection(COLL_NAME):
        utility.drop_collection(COLL_NAME)

    fields = [
        FieldSchema(name="macro_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),

        FieldSchema(name="pais",     dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="variables",dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="unidad",   dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="fecha",    dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="anio",     dtype=DataType.INT64),
        FieldSchema(name="mes",      dtype=DataType.INT64),
        FieldSchema(name="valor",    dtype=DataType.DOUBLE),
        FieldSchema(name="canonical_text", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Macroeconomic variables LATAM (PG -> Milvus, E5 embeddings)")
    col = Collection(COLL_NAME, schema=schema)

    # Índice vectorial real
    col.create_index(field_name="embedding", index_params={
        "index_type": "HNSW", "metric_type": "IP", "params": {"M": 32, "efConstruction": 200}
    })

    try:
        col.create_index(field_name="pais", index_params={"index_type":"INVERTED","metric_type":"L2","params":{}})
    except Exception:
        pass
    try:
        col.create_index(field_name="variables", index_params={"index_type":"INVERTED","metric_type":"L2","params":{}})
    except Exception:
        pass
    return col

# --------- Embedding (E5) ---------
def embed_texts(texts: List[str]):
    # E5: usar prefijo "passage: "
    from sentence_transformers import SentenceTransformer
    import torch, numpy as np
    model_name = EMBED_MODEL
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    embs = model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
    return embs.astype("float32").tolist()

def migrate():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    col = ensure_collection(dim=768)

    rows = fetch_rows()
    print(f"[INFO] PG -> {len(rows)} filas")

    # preparar buffers
    ids, embs, paises, vars_, units, fechas, anios, meses, valores, canons = ([] for _ in range(10))
    texts_for_embed: List[str] = []

    for (pais, var, uni, fec, val, anio, mes) in rows:
        iso2 = to_iso2(str(pais) if pais is not None else "")
        dstr = norm_date(fec, anio, mes)
        mid  = make_id(iso2, str(var or ""), dstr or f"{anio}-{mes}")
        # Texto canónico SIN el valor numérico (para que el embedding represente concepto, no magnitud)
        canon = " ".join([x for x in [iso2 or (pais or ""), str(var or ""), f"({uni})" if uni else "", dstr] if x])

        ids.append(mid)
        paises.append(iso2 or (pais or ""))
        vars_.append(var or "")
        units.append(uni or "")
        fechas.append(dstr)
        anios.append(int(anio) if anio is not None else 0)
        meses.append(int(mes) if mes is not None else 0)
        try:
            valores.append(float(val) if val is not None else 0.0)
        except Exception:
            valores.append(0.0)
        canons.append(canon)
        texts_for_embed.append(canon)

    # Embeddings reales
    embs = embed_texts(texts_for_embed)

    # insert por lotes
    B = 5000
    total = len(ids)
    for i in range(0, total, B):
        sl = slice(i, i+B)
        data = [
            ids[sl], embs[sl], paises[sl], vars_[sl], units[sl],
            fechas[sl], anios[sl], meses[sl], valores[sl], canons[sl]
        ]
        col.insert(data)
        print(f"[OK] insertados {min(i+B, total)}/{total}")
    col.flush()
    col.load()
    print("[DONE] macro_latam creado (con embeddings reales) y cargado.")

if __name__ == "__main__":
    migrate()
