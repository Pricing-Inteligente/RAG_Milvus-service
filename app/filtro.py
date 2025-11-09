# count_categories_milvus.py
from __future__ import annotations
import os, argparse, csv
from collections import Counter
from typing import Dict, Any, List
from pymilvus import connections, Collection

def env(k, d=None): 
    return os.getenv(k, d)

def build_expr(country: str | None, store: str | None) -> str:
    clauses = []
    if country:
        # Acepta ISO2 y nombre
        vals = [country, country.upper(), country.capitalize()]
        quoted = ",".join([f"'{v}'" for v in vals])
        clauses.append(f"country in [{quoted}]")
    if store:
        clauses.append(f"store == '{store}'")
    return " and ".join(clauses) if clauses else ""

def iter_all_categories(col: Collection, category_field: str, expr: str, batch_size: int = 8192):
    """
    Recorre toda la colección (o el subset filtrado) trayendo solo 'category' por lotes.
    En PyMilvus 2.6.x se usa .next() (no es iterable con 'for').
    """
    it = col.query_iterator(expr=expr, output_fields=[category_field], batch_size=batch_size)
    try:
        while True:
            batch: List[Dict[str, Any]] = it.next()
            if not batch:
                break
            yield batch
    finally:
        try:
            it.close()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser(description="Cuenta valores de 'category' en Milvus.")
    ap.add_argument("--host", default=env("MILVUS_HOST", "127.0.0.1"))
    ap.add_argument("--port", default=env("MILVUS_PORT", "19530"))
    ap.add_argument("--collection", default=env("MILVUS_COLLECTION", "products_latam"))
    ap.add_argument("--category_field", default=env("CATEGORY_FIELD", "category"))
    ap.add_argument("--country", help="Filtrar por country (ej. CO)", default=None)
    ap.add_argument("--store", help="Filtrar por store (ej. Olimpica)", default=None)
    ap.add_argument("--csv", help="Ruta para exportar CSV con conteo", default=None)
    args = ap.parse_args()

    print(f"[connect] Milvus {args.host}:{args.port}")
    connections.connect(host=args.host, port=args.port)

    col = Collection(args.collection)
    try:
        col.load()
    except Exception:
        pass

    expr = build_expr(args.country, args.store)
    if expr:
        print(f"[expr] {expr}")

    counter = Counter()
    total = 0

    for batch in iter_all_categories(col, args.category_field, expr):
        for r in batch:
            cat = r.get(args.category_field) or "<NULL>"
            counter[cat] += 1
            total += 1

    print("\n=== Conteo por categoría ===")
    for cat, cnt in counter.most_common():
        print(f"{cat:20s}  {cnt:>8d}")
    print(f"\nTotal filas contadas: {total}")
    print(f"Categorías distintas: {len(counter)}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "count"])
            for cat, cnt in counter.most_common():
                w.writerow([cat, cnt])
        print(f"[csv] escrito: {os.path.abspath(args.csv)}")

if __name__ == "__main__":
    main()
