# migrar_pg_a_milvus.py – versión robusta para columnas flexibles
# • Soporta colecciones split (productos/macro) o unificadas
# • Mapea columnas de Postgres vía .env y genera IDs cuando no existen
# • Crea/esquema Milvus con nombres de campo configurables por .env
# • Tolera columnas ausentes (unidad, cantidad, etc.)

import os, re, json, hashlib
from datetime import datetime
from dateutil import parser as dtparser
from dotenv import load_dotenv
from tqdm import tqdm
import psycopg2, psycopg2.extras as pgx

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# -------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------

def _parse_date_int(s: str) -> int:
    if not s:
        return 0
    s = str(s).strip()
    try:
        dt = dtparser.parse(s, dayfirst=True, fuzzy=True)
        return int(dt.strftime("%Y%m%d"))
    except Exception:
        pass
    # patrón especial d.M.y.H
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})\.(\d{1,2})$", s)
    if m:
        d, M, y, H = map(int, m.groups())
        if y < 100:
            y += 2000
        try:
            return int(datetime(y, M, d, H).strftime("%Y%m%d"))
        except Exception:
            return 0
    return 0


def _parse_number(text: str):
    if text is None:
        return None, None
    s = str(text).strip()
    sym = None
    m = re.search(r"(USD|EUR|ARS|COP|MXN|R\$|\$|€|£|₡)", s, re.I)
    if m:
        sym = m.group(1)
    raw = re.sub(r"[^0-9\.,-]", "", s)

    if raw.count(",") > 0 and raw.count(".") > 0:
        last = max(raw.rfind(","), raw.rfind("."))
        dec = raw[last]
        grp = "," if dec == "." else "."
        num = raw.replace(grp, "").replace(dec, ".")
    elif raw.count(",") > 0:
        parts = raw.split(",")
        if len(parts[-1]) in (1, 2):
            num = raw.replace(".", "").replace(",", ".")
        else:
            num = raw.replace(",", "")
    else:
        parts = raw.split(".")
        if len(parts[-1]) in (1, 2):
            num = raw
        else:
            num = raw.replace(".", "")
    try:
        return float(num), sym
    except Exception:
        return None, sym


def to_str(x, maxlen=None):
    if x is None:
        s = ""
    elif isinstance(x, (list, tuple, set)):
        s = " ".join("" if v is None else str(v) for v in x)
    elif isinstance(x, dict):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)
    s = s.strip()
    if maxlen and len(s) > maxlen:
        s = s[:maxlen]
    return s

# -------------------------------------------------------------
# Embeddings (E5 por defecto)
# -------------------------------------------------------------
_embedder = None

def embed_passages(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
        _embedder = SentenceTransformer(model_name)
    texts = [f"passage: {t}" for t in texts]
    vecs = _embedder.encode(texts, normalize_embeddings=True)
    return vecs.tolist(), len(vecs[0])

# -------------------------------------------------------------
# Milvus helpers
# -------------------------------------------------------------

def _schema_field_names(col: Collection):
    return [f.name for f in col.schema.fields]


def reorder_to_schema(col: Collection, field_order, columns):
    schema_names = _schema_field_names(col)
    idx_map = {name: i for i, name in enumerate(field_order)}
    try:
        return [columns[idx_map[name]] for name in schema_names]
    except KeyError as e:
        raise KeyError(
            f"Campo ausente para reordenar: {e}. Schema={schema_names} / field_order={field_order}"
        )


def ensure_database():
    db_name = os.getenv("MILVUS_DB", "default")
    try:
        from pymilvus import db
        if db_name != "default":
            if db_name not in db.list_database():
                db.create_database(db_name)
            db.using_database(db_name)
            print(f"[INFO] Using Milvus database '{db_name}'")
        else:
            print("[INFO] Using Milvus database 'default'")
    except Exception as e:
        print(f"[WARN] Database namespaces not available or error: {e}")


def ensure_collection(dim, *, name, description, value_field_name="price"):
    vec_field = os.getenv("VECTOR_FIELD", "embedding")
    metric = os.getenv("MILVUS_METRIC", "cosine").upper()
    if metric in ("COSINE", "COS"):
        metric = "IP"

    # Campos comunes configurables
    PK = os.getenv("PRIMARY_KEY_FIELD", "product_id")
    TEXT = os.getenv("TEXT_FIELD", "canonical_text")
    URL = os.getenv("URL_FIELD", "url")
    NAME = os.getenv("NAME_FIELD", "name")
    BRAND = os.getenv("BRAND_FIELD", "brand")
    CAT = os.getenv("CATEGORY_FIELD", "category")
    STORE = os.getenv("STORE_FIELD", "store")
    COUNTRY = os.getenv("COUNTRY_FIELD", "country")
    UNIT = os.getenv("UNIT_FIELD", "unit")
    SIZE = os.getenv("SIZE_FIELD", "size")
    CURR = os.getenv("CURRENCY_FIELD", "currency")
    LAST = os.getenv("LAST_SEEN_FIELD", "last_seen")

    recreate = os.getenv("RECREATE_COLLECTION", "0") == "1"
    if recreate and utility.has_collection(name):
        print(f"[INFO] Dropping existing collection '{name}'…")
        utility.drop_collection(name)

    if not utility.has_collection(name):
        fields = [
            FieldSchema(name=PK, dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name=vec_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name=TEXT, dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name=URL, dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name=NAME, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=value_field_name, dtype=DataType.DOUBLE),
            FieldSchema(name=BRAND, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=CAT, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=STORE, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=COUNTRY, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=UNIT, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=SIZE, dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name=CURR, dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name=LAST, dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description=description)
        col = Collection(name, schema)
        col.create_index(
            field_name=vec_field,
            index_params={
                "metric_type": metric,
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        print(f"[INFO] Created collection '{name}' (metric={metric})")
    else:
        col = Collection(name)
        print(f"[INFO] Using existing collection '{name}'")

    col.load()
    if os.getenv("DEBUG_SCHEMA", "0") == "1":
        print("[DEBUG] schema_order =", _schema_field_names(col))
    return col


# Coerción segura de columnas a tipos del schema

def coerce_by_schema(col, field_order, columns):
    dtypes = {f.name: f.dtype for f in col.schema.fields}
    coerced = []
    for name, vals in zip(field_order, columns):
        dt = dtypes[name]
        if dt == DataType.VARCHAR:
            coerced.append([to_str(v) for v in vals])
        elif dt == DataType.DOUBLE:
            def to_float(x):
                if x is None:
                    return None
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, list):
                    return float(x[0]) if x else None
                s = str(x).strip()
                if s.count(",") and s.count("."):
                    last = max(s.rfind(","), s.rfind("."))
                    dec = s[last]
                    grp = "," if dec == "." else "."
                    s = s.replace(grp, "").replace(dec, ".")
                elif s.count(",") == 1 and s.count(".") == 0:
                    s = s.replace(".", "").replace(",", ".")
                else:
                    s = s.replace(",", "")
                try:
                    return float(s)
                except:
                    return None
            coerced.append([to_float(v) for v in vals])
        elif dt == DataType.INT64:
            def to_int(x):
                if x is None:
                    return 0
                if isinstance(x, (int, float)):
                    return int(x)
                try:
                    return int(str(x))
                except:
                    return 0
            coerced.append([to_int(v) for v in vals])
        elif dt == DataType.FLOAT_VECTOR:
            coerced.append([[float(f) for f in v] for v in vals])
        else:
            coerced.append(list(vals))
    return coerced


def safe_insert(col, rows, field_order):
    if not rows:
        return 0, 0
    columns = [list(c) for c in zip(*rows)]
    columns = coerce_by_schema(col, field_order, columns)
    try:
        columns = reorder_to_schema(col, field_order, columns)
        col.insert(columns)
        return len(rows), 0
    except Exception as e:
        if len(rows) == 1:
            rid = rows[0][0]
            print(f"[SKIP] Row {rid} descartada: {e}")
            return 0, 1
        mid = len(rows) // 2
        ok1, sk1 = safe_insert(col, rows[:mid], field_order)
        ok2, sk2 = safe_insert(col, rows[mid:], field_order)
        return ok1 + ok2, sk1 + sk2


def insert_rows(col, rows, field_order):
    if not rows:
        return
    ok, skipped = safe_insert(col, rows, field_order)
    if skipped:
        print(f"[WARN] Insertadas OK: {ok}, saltadas: {skipped}")

# -------------------------------------------------------------
# Lectura Postgres (configurable por .env)
# -------------------------------------------------------------

def get_env_map(prefix, defaults):
    out = {}
    for k, v in defaults.items():
        out[k] = os.getenv(f"{prefix}_{k}", v)
    return out

# Defaults ajustados a tu tabla_precios
PRODUCTS_MAP_DEFAULT = {
    "TABLE": "tabla_precios",
    "ID": "__auto__",       # genera ID estable si no hay columna
    "NAME": "nombre",
    "BRAND": "marca",
    "PRICE": "precio_simulado",
    "DESC": "nombre",       # si no hay descripción, reutiliza nombre
    "DATE": "fecha",
    "CATEGORY": "producto", # en tu tabla es 'producto'
    "COUNTRY": "pais",
    "STORE": "retail",
    "URL": "url",
    "CURRENCY": "moneda",
}

MACRO_MAP_DEFAULT = {
    "TABLE": "variables_clean_fields",
    "ID": "__auto__",       # genera ID si no hay columna
    "VARIABLE": "nombre",
    "VALUE": "valor",
    "COUNTRY": "pais",
    "DATE": "fecha",
    "PREVIOUS": "previous",
    "UNIT": "unit",
}

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    load_dotenv()

    # Conectar Milvus y Postgres
    connections.connect(host=os.getenv("MILVUS_HOST", "127.0.0.1"),
                        port=os.getenv("MILVUS_PORT", "19530"))
    ensure_database()

    pg = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=int(os.getenv("PG_PORT", 5432)),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )
    cur = pg.cursor(cursor_factory=pgx.DictCursor)

    # Campos Milvus (nombres lógicos)
    PK = os.getenv("PRIMARY_KEY_FIELD", "product_id")
    VEC = os.getenv("VECTOR_FIELD", "embedding")
    TEXT = os.getenv("TEXT_FIELD", "canonical_text")
    URL = os.getenv("URL_FIELD", "url")
    NAME = os.getenv("NAME_FIELD", "name")
    VALUE = os.getenv("VALUE_FIELD", os.getenv("PRICE_FIELD", "price"))  # compatibilidad
    BRAND = os.getenv("BRAND_FIELD", "brand")
    CAT = os.getenv("CATEGORY_FIELD", "category")
    STORE = os.getenv("STORE_FIELD", "store")
    COUNTRY = os.getenv("COUNTRY_FIELD", "country")
    UNIT = os.getenv("UNIT_FIELD", "unit")
    SIZE = os.getenv("SIZE_FIELD", "size")
    CURR = os.getenv("CURRENCY_FIELD", "currency")
    LAST = os.getenv("LAST_SEEN_FIELD", "last_seen")

    # Modo colecciones: unified | split
    mode = os.getenv("MILVUS_COLLECTION_MODE", "unified").lower()
    coll_products_name = os.getenv("MILVUS_COLLECTION_PRODUCTS",
                                   os.getenv("MILVUS_COLLECTION", "products_latam"))
    coll_macro_name = os.getenv("MILVUS_COLLECTION_MACRO", "macro_latam")

    # dimensión de embeddings
    _, dim = embed_passages(["probar"])

    # Crear/usar colecciones según modo
    if mode == "split":
        col_products = ensure_collection(dim, name=coll_products_name,
                                         description="LatAm products",
                                         value_field_name=VALUE)
        col_macro = ensure_collection(dim, name=coll_macro_name,
                                      description="LatAm macro variables",
                                      value_field_name=VALUE)
    else:
        col_unified = ensure_collection(dim, name=coll_products_name,
                                        description="LatAm products + macro (unificado)",
                                        value_field_name=VALUE)

    field_order = [PK, VEC, TEXT, URL, NAME, VALUE, BRAND, CAT, STORE,
                   COUNTRY, UNIT, SIZE, CURR, LAST]

    # ------------------------- Productos -------------------------
    P = get_env_map("PG_PRODUCTS", PRODUCTS_MAP_DEFAULT)
    id_expr_p = "NULL" if P["ID"] in (None, "", "__auto__") else P["ID"]

    cur.execute(
        f"""
        SELECT {id_expr_p} AS id,
               {P['NAME']} AS name,
               {P['BRAND']} AS brand,
               {P['PRICE']} AS price,
               {P['DESC']} AS desc,
               {P['DATE']} AS date,
               {P['CATEGORY']} AS category,
               {P['COUNTRY']} AS country,
               {P['STORE']} AS store,
               {P['URL']} AS url,
               {P['CURRENCY']} AS currency
        FROM {P['TABLE']}
        """
    )
    productos = cur.fetchall()

    buf = []
    for r in tqdm(productos, desc="Migrando productos"):
        pid = to_str(r.get("id"))[:128]
        if not pid:
            # Genera ID estable cuando no existe en la tabla
            key = "|".join(
                [
                    to_str(r.get("country")), to_str(r.get("store")),
                    to_str(r.get("brand")), to_str(r.get("category")),
                    to_str(r.get("name")), to_str(r.get("url")),
                ]
            )
            pid = hashlib.md5(key.encode("utf-8")).hexdigest()

        name = to_str(r.get("name"), 512)
        brand = to_str(r.get("brand"), 256)
        category = to_str(r.get("category"), 256)
        unit = to_str(r.get("unit"), 64)          # puede no existir ⇒ ""
        size = to_str(r.get("qty"), 128)          # puede no existir ⇒ ""
        store = to_str(r.get("store"), 256)
        country = to_str(r.get("country"), 64)
        url = to_str(r.get("url"), 1024)
        desc = to_str(r.get("desc"), 2000)
        last_seen = _parse_date_int(r.get("date"))
        value_num, currency_sym = _parse_number(r.get("price"))
        if value_num is None:
            # No insertamos filas sin número válido
            continue
        if not currency_sym:
            currency_sym = to_str(r.get("currency"))

        canonical = " ".join(filter(None, [country, store, category, brand, name, desc]))
        vec, _ = embed_passages([canonical])

        row = [
            pid, vec[0], canonical, url, name, value_num, brand, category,
            store, country, unit, size, currency_sym or "", last_seen,
        ]
        buf.append(row)

        if len(buf) >= 1000:
            if mode == "split":
                insert_rows(col_products, buf, field_order)
            else:
                insert_rows(col_unified, buf, field_order)
            buf = []

    if mode == "split":
        insert_rows(col_products, buf, field_order)
    else:
        insert_rows(col_unified, buf, field_order)

    # ------------------------- Macro -------------------------
    M = get_env_map("PG_MACRO", MACRO_MAP_DEFAULT)
    id_expr_m = "NULL" if M["ID"] in (None, "", "__auto__") else M["ID"]

    cur.execute(
        f"""
        SELECT {id_expr_m} AS id,
               {M['VARIABLE']} AS variable,
               {M['VALUE']} AS value,
               {M['COUNTRY']} AS country,
               {M['DATE']} AS date,
               {M['PREVIOUS']} AS previous,
               {M['UNIT']} AS unit
        FROM {M['TABLE']}
        """
    )
    macros = cur.fetchall()

    buf = []
    for r in tqdm(macros, desc="Migrando macro"):
        mid_raw = to_str(r.get("id"))
        if not mid_raw:
            base = f"{to_str(r.get('country'))}|{to_str(r.get('variable'))}|{to_str(r.get('date'))}"
            mid_raw = hashlib.md5(base.encode("utf-8")).hexdigest()
        mid = f"macro:{mid_raw[:118]}"

        variable = to_str(r.get("variable"), 256)
        unit_m = to_str(r.get("unit"), 64)
        country = to_str(r.get("country"), 64)
        val_raw = to_str(r.get("value"), 64)
        fecha_raw = to_str(r.get("date"), 64)
        value_num, curr = _parse_number(r.get("value"))
        if value_num is None:
            continue
        last_seen = _parse_date_int(r.get("date"))

        name = variable
        category = "macro"
        brand = ""
        store = ""
        size = ""
        url = ""

        canonical = f"{country} {variable} {val_raw} {unit_m} fecha {fecha_raw}".strip()
        vec, _ = embed_passages([canonical])

        row = [
            mid, vec[0], canonical, url, name, value_num, brand, category,
            store, country, unit_m, size, curr or "", last_seen,
        ]
        buf.append(row)

        if len(buf) >= 1000:
            if mode == "split":
                insert_rows(col_macro, buf, field_order)
            else:
                insert_rows(col_unified, buf, field_order)
            buf = []

    if mode == "split":
        insert_rows(col_macro, buf, field_order)
    else:
        insert_rows(col_unified, buf, field_order)

    # flush & compact
    if mode == "split":
        for c in (col_products, col_macro):
            c.flush()
            try:
                utility.compact(c.name)
            except Exception:
                pass
            c.load()
    else:
        col_unified.flush()
        try:
            utility.compact(col_unified.name)
        except Exception:
            pass
        col_unified.load()

    cur.close()
    pg.close()
    print("✅ Migración completada.")


if __name__ == "__main__":
    main()
