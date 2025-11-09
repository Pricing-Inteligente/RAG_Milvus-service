# build_fts_from_pg.py
import os, sqlite3, psycopg2, hashlib, math

# Lee variables del .env (tu entorno debe tenerlas cargadas)
def env(k, d=None): return os.getenv(k, d)

PG_HOST = env("PG_HOST"); PG_PORT = int(env("PG_PORT", "5432"))
PG_DB   = env("PG_DB");   PG_USER = env("PG_USER"); PG_PASS = env("PG_PASSWORD")

TAB     = env("PG_PRODUCTS_TABLE", "tabla_precios")
FID     = env("PG_PRODUCTS_ID", "__auto__")
FNAME   = env("PG_PRODUCTS_NAME", None)      # opcional
FBRAND  = env("PG_PRODUCTS_BRAND", None)     # opcional
FCAT    = env("PG_PRODUCTS_CATEGORY", "category")
FSTORE  = env("PG_PRODUCTS_STORE", None)     # opcional
FCOUN   = env("PG_PRODUCTS_COUNTRY", None)   # opcional
FDESC   = env("PG_PRODUCTS_DESC", None)      # opcional
FREF    = env("PG_PRODUCTS_REF", None)       # opcional

OUT_DB  = env("OUT_FTS_DB", r"C:\RAG-FTS\fts.db")

# Arma la lista de columnas que intentaremos leer
cols = []
id_col = None if FID == "__auto__" else FID
if id_col: cols.append(id_col)
for c in [FNAME, FBRAND, FCAT, FSTORE, FCOUN, FDESC, FREF]:
    if c: cols.append(c)
# Evita duplicados conservando orden
seen = set(); safe_cols = []
for c in cols:
    if c not in seen:
        safe_cols.append(c); seen.add(c)

if not FCAT:
    raise SystemExit("Necesito al menos PG_PRODUCTS_CATEGORY (p.ej. 'producto' en tu .env).")

# Conéctate a PG y trae datos por lotes
conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)
cur  = conn.cursor(name="fts_cursor")  # server-side cursor
cur.itersize = 5000

select_cols = ", ".join([f'"{c}"' for c in safe_cols])
cur.execute(f'SELECT {select_cols} FROM "{TAB}"')

# Prepara SQLite FTS5
os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)
if os.path.exists(OUT_DB):
    os.remove(OUT_DB)
sconn = sqlite3.connect(OUT_DB)
scur  = sconn.cursor()
# Tabla virtual FTS5
scur.execute("CREATE VIRTUAL TABLE products_fts USING fts5(text, country, category, store, brand, content='');")

def make_id(name, brand, store, country, cat):
    base = "|".join([name or "", brand or "", store or "", country or "", cat or ""])
    return hashlib.md5(base.encode("utf-8")).hexdigest()

rows = 0
while True:
    batch = cur.fetchmany(5000)
    if not batch: break
    for r in batch:
        pos = 0
        val = {}
        if id_col:
            val["id"] = r[pos]; pos += 1
        for c in [FNAME, FBRAND, FCAT, FSTORE, FCOUN, FDESC, FREF]:
            if c:
                val[c] = r[pos]; pos += 1
        name    = (val.get(FNAME)  or "").strip() if FNAME else ""
        brand   = (val.get(FBRAND) or "").strip() if FBRAND else ""
        cat     = (val.get(FCAT)   or "").strip()
        store   = (val.get(FSTORE) or "").strip() if FSTORE else ""
        country = (val.get(FCOUN)  or "").strip() if FCOUN else ""
        desc_   = (val.get(FDESC)  or "").strip() if FDESC else ""
        ref_    = (val.get(FREF)   or "").strip() if FREF else ""
        pid = (val.get("id") if id_col else make_id(name, brand, store, country, cat))

        # Texto que BM25 va a rankear:
        text = " ".join(x for x in [name, brand, cat, store, country, desc_, ref_] if x)

        scur.execute(
            "INSERT INTO products_fts (text, country, category, store, brand) VALUES (?,?,?,?,?)",
            (text, country, cat, store, brand)
        )
        rows += 1
        if rows % 10000 == 0:
            sconn.commit()
sconn.commit()
cur.close(); conn.close(); sconn.close()
print(f"OK: {rows} filas indexadas en {OUT_DB}")
