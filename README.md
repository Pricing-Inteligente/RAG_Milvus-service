### RAG_Milvus-service

Orquestación y utilidades para la capa vectorial de Pricing Inteligente basada en Milvus. Provee:
- Milvus Standalone para almacenamiento vectorial (embeddings de productos, textos y resultados LASSO).
- MinIO como backend de objetos.
- etcd como servicio de metadatos.
- Attu como UI web para explorar colecciones, índices y datos.

## Contexto dentro de Pricing Inteligente
Pricing Inteligente unifica datos de precios y variables macro de 10 países y múltiples retailers/marcas. Esta capa:
- Indexa embeddings de documentos/tablas para RAG (chatbot).
- Aloja colecciones derivadas (p. ej., `lasso_models`) para filtrar y enriquecer respuestas.
- Sirve al Frontend (chatbot) y a dashboards (Plotly Dash + Grok API) para consultas semánticas y analíticas.

## Servicios (docker-compose)
Definidos en `docker-compose.yml`:
- `milvus-etcd`: metadatos (puertos internos 2379/2380).
- `milvus-minio`: almacenamiento de objetos (consola opcional en 9001; en este compose se publica en `19001:9001`).
- `milvus-standalone`: Milvus v2.3.16 (gRPC `19530`, HTTP `9091`).
- `attu`: UI de Milvus (expuesta en `18000:3000`).

Red: `milvus-net` (interna).  
Volúmenes persistentes: `./volumes/etcd`, `./volumes/minio`, `./volumes/milvus`.

Puertos publicados (host → contenedor):
- 19530 → 19530 (Milvus gRPC)
- 9091 → 9091 (Milvus HTTP/health)
- 18000 → 3000 (Attu UI)
- 19001 → 9001 (MinIO console; opcional)

Variables relevantes:
- `MINIO_ROOT_USER=minioadmin`
- `MINIO_ROOT_PASSWORD=minioadmin`
- `ETCD_ENDPOINTS=milvus-etcd:2379`
- `MINIO_ADDRESS=milvus-minio:9000`
- `COMMON_STORAGETYPE=minio`

Healthcheck:
- Milvus se valida vía `http://localhost:9091/healthz` (retries e intervalos definidos).

## Requisitos
- Docker y Docker Compose.
- 4 GiB de RAM libres recomendadas para levantar Milvus + MinIO + Attu cómodamente.

## Instalación y arranque
1) Crear las carpetas de volúmenes si aún no existen:
   - `mkdir -p volumes/etcd volumes/minio volumes/milvus`
2) Levantar el stack:
   - `docker compose up -d`
3) Verificar salud:
   - `curl http://localhost:9091/healthz` → `{"status":"OK"}`
4) Abrir Attu:
   - `http://localhost:18000`
   - Conectar a `milvus-standalone:19530` (service name interno).

Apagar:
- `docker compose down` (no borra volúmenes).

Logs:
- `docker compose logs -f milvus-standalone`
- `docker compose logs -f attu`

## Uso desde Python
Ejemplo básico de conexión (pymilvus):

```python
from pymilvus import connections, utility

connections.connect(alias="default", host="127.0.0.1", port="19530")
print(utility.get_server_version())
```

Creación de colección y carga de vectores:

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields, description="Embeddings de productos")
col = Collection("products_embeddings", schema)
col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
col.load()
```

Consulta vectorial:

```python
query_vec = [[0.0]*768]  # reemplazar por embedding real
res = col.search(query_vec, "embedding", param={"nprobe": 16}, limit=10, output_fields=["product_id"])
for hits in res:
    for h in hits:
        print(h.id, h.distance, h.entity.get("product_id"))
```

## Integración con otros servicios
- Ingesta LASSO: cargar tabla `lasso_models` con features/coeficientes para enriquecer respuestas.
- Chatbot: usa RAG con colecciones de documentos (productos, catálogos, notas metodológicas).
- Dashboards: consultas agregadas a colecciones para filtros/segmentaciones.

## Personalización
- Cambiar credenciales de MinIO en `docker-compose.yml` si se expone públicamente.
- Ajustar `index_type`, `metric_type` y parámetros (`nlist`, `nprobe`) según volumen y latencia.
- Para despliegue remoto, revisar mapeos de puertos y seguridad (firewall, TLS, secretos).

## Troubleshooting
- Attu no conecta: usar `milvus-standalone:19530` como host desde Attu (service name).
- `healthz` falla: esperar a que etcd y minio estén listos; revisar `depends_on` y logs.
- Espacio en disco: verificar volúmenes `./volumes/*` y políticas de MinIO.
- Conflicto de puertos: cambiar `18000`, `19001`, `19530`, `9091` en el compose del host.

## Archivos
- `docker-compose.yml`: definición principal de servicios (etcd, minio, milvus, attu).
- `docker-compose.override.yml`: ajustes locales opcionales (volúmenes/command).
- `volumes/`: persistencia de datos (creados en runtime).


