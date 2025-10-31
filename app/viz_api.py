# viz_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re, json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    prompt: str

def extract_dataset(prompt: str):
    m = re.search(r"Datos\s*\(JSON\)\s*:\s*(\[.*\])\s*$", prompt, re.S | re.I)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except Exception:
        return []

@app.post("/generate-graph")
def generate_graph(req: Req):
    rows = extract_dataset(req.prompt)
    if not rows:
        return {"error": "NO_DATA"}

    x = [str(r.get("label", "")) for r in rows]
    y = [float(r.get("value", 0)) for r in rows]

    # Especificación “style-less” compatible con Plotly del front
    figure = {
        "data": [{
            "type": "bar",
            "x": x,
            "y": y,
            "text": [
                f"{r.get('value')} {r.get('currency','')}".strip() for r in rows
            ],
            "textposition": "auto",
            "hovertemplate": "%{x}<br>Precio: %{y}<extra></extra>",
        }],
        "layout": {
            "title": {"text": "Visualización generada"},
            "xaxis": {"automargin": True},
            "yaxis": {"automargin": True},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 70},
        }
    }
    return {"figure": figure}
