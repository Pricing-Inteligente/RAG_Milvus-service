# build_fts_sqlite.py
import os, csv, sqlite3, argparse
from typing import Dict

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def build_from_csv(csv_path: str, db_path: str):
    ensure_dir(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # FTS5 con columnas filtrables en where
    cur.execute("CREATE VIRTUAL TABLE products_fts USING fts5(text, country, category, store, brand, content='');")
    # Opcional: índices auxiliares en tabla shadow no aplican a FTS; las col. extra van aquí para filtrado
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = 0
        for r in reader:
            text = " ".join(
                str(x) for x in [
                    r.get("canonical_text") or "",
                    r.get("name") or "",
                    r.get("brand") or "",
                    r.get("category") or "",
                    r.get("store") or "",
                    r.get("country") or "",
                ]
                if x
            )
            vals = (
                text.strip(),
                (r.get("country") or "").strip(),
                (r.get("category") or "").strip(),
                (r.get("store") or "").strip(),
                (r.get("brand") or "").strip(),
            )
            cur.execute("INSERT INTO products_fts (text, country, category, store, brand) VALUES (?,?,?,?,?)", vals)
            rows += 1
            if rows % 10000 == 0:
                conn.commit()
        conn.commit()
    conn.close()
    print(f"OK: {rows} filas indexadas en {db_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV con columnas: product_id,name,brand,category,store,country,canonical_text")
    ap.add_argument("--db",  required=True, help="ruta del archivo SQLite a crear, ej: /abs/fts.db")
    args = ap.parse_args()
    build_from_csv(args.csv, args.db)

