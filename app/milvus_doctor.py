# app/milvus_doctor.py — Doctor robusto para Milvus
import os, sys, json
from pymilvus import connections, utility, Collection

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
DB_NAME     = os.getenv("MILVUS_DB", "default")
COLLECTION  = os.getenv("MILVUS_COLLECTION", "retail_products")
VECTOR_ENV  = os.getenv("MILVUS_VECTOR_FIELD", "").strip()
TOPK        = int(os.getenv("MILVUS_TOPK", "5"))

FLOAT_VECTOR = 101
BINARY_VECTOR = 100

def _parse_idx(idx):
    """
    PyMilvus v2.3.x no expone index_type/metric_type como atributos.
    Esta función los extrae desde idx.params (dict o JSON string).
    """
    params = getattr(idx, "params", {}) or {}
    if isinstance(params, str):
        try: params = json.loads(params)
        except Exception: params = {}
    return {
        "field": getattr(idx, "field_name", None),
        "index_name": getattr(idx, "index_name", None),
        "index_type": params.get("index_type"),
        "metric_type": params.get("metric_type"),
        "raw_params": params.get("params") or params
    }

def main():
    print(f"[INFO] Conectando a Milvus {MILVUS_HOST}:{MILVUS_PORT} ...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("[OK ] Conectado.")

    # DB (best-effort)
    try:
        if DB_NAME in utility.list_databases():
            utility.set_database(DB_NAME)
    except Exception:
        pass

    cols = utility.list_collections()
    print(f"[INFO] Colecciones existentes: {cols}")
    if COLLECTION not in cols:
        print(f"[ERR] No existe la colección '{COLLECTION}'.")
        sys.exit(2)

    col = Collection(COLLECTION)
    col.load()
    print(f"[OK ] Colección '{COLLECTION}' cargada.")

    print("[INFO] Esquema:")
    vector_field_name, vec_dim, vec_dtype = None, None, None
    for f in col.schema.fields:
        p = getattr(f, "params", {}) or {}
        print(f"  - {f.name}: {f.dtype}, is_primary={getattr(f,'is_primary',False)}, params={p}")
    # Campo vector (por env o autodetección)
    if VECTOR_ENV:
        for f in col.schema.fields:
            if f.name == VECTOR_ENV:
                vector_field_name = f.name
                vec_dtype = int(getattr(f, "dtype", 0))
                p = getattr(f, "params", {}) or {}
                vec_dim = int(p.get("dim") or p.get("max_length") or 0)
                break
        if not vector_field_name:
            print(f"[ERR] No encontré el campo vector '{VECTOR_ENV}'."); sys.exit(3)
    else:
        for f in col.schema.fields:
            dt = int(getattr(f, "dtype", 0))
            if dt in (FLOAT_VECTOR, BINARY_VECTOR):
                vector_field_name = f.name
                vec_dtype = dt
                p = getattr(f, "params", {}) or {}
                vec_dim = int(p.get("dim") or p.get("max_length") or 0)
                break
        if not vector_field_name:
            print("[ERR] No hay campo de tipo vector en el schema."); sys.exit(3)

    print(f"[INFO] Campo vector: '{vector_field_name}' dtype={vec_dtype} dim={vec_dim}")

    print("[INFO] Índices:")
    idxs = col.indexes or []
    if not idxs:
        print("  (no hay índices: búsqueda será FLAT/naive si procede)")
    metric = "COSINE"
    for idx in idxs:
        info = _parse_idx(idx)
        print(f"  - field={info['field']} index_name={info['index_name']} "
              f"index_type={info['index_type']} metric_type={info['metric_type']} params={info['raw_params']}")
        if info["field"] == vector_field_name and info.get("metric_type"):
            metric = info["metric_type"]

    n = col.num_entities
    print(f"[INFO] Registros en '{COLLECTION}': {n}")
    if n == 0:
        print("[WARN] La colección está vacía. Reingesta necesaria.")
        sys.exit(0)

    # Sample para smoke test
    expr = ""
    try:
        has_pk = any(f.name == "product_id" for f in col.schema.fields)
        expr = "product_id != ''" if has_pk else ""
    except Exception:
        pass

    sample = col.query(expr=expr, output_fields=[vector_field_name], limit=1)
    if not sample:
        print("[WARN] No pude samplear una entidad para el smoke test."); sys.exit(0)
    qvec = sample[0][vector_field_name]
    if vec_dim and len(qvec) != vec_dim:
        print(f"[ERR] Longitud del vector sample={len(qvec)} != dim colección={vec_dim}"); sys.exit(4)

    print(f"[INFO] Ejecutando search TOPK={TOPK} metric={metric} ...")
    res = col.search(
        data=[qvec],
        anns_field=vector_field_name,
        param={"metric_type": metric, "params": {"nprobe": 16}},
        limit=TOPK,
        output_fields=["product_id","name","brand","category","store","country","price","currency","url"]
    )
    hits = res[0]
    print(f"[OK ] Hits={len(hits)}")
    for i, h in enumerate(hits, 1):
        print(f"  #{i} distance={h.distance} id={h.id} {h.entity.get('name')} | {h.entity.get('brand')} | {h.entity.get('price')} {h.entity.get('currency')}")
    print("[DONE] Doctor finalizado sin errores críticos.")

if __name__ == "__main__":
    main()
