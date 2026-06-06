"""
Hardware Eye — local HTTP server.

Start:   python server.py
Then open docs/index.html in a browser (or visit http://127.0.0.1:8000).
"""

from __future__ import annotations

import base64
import json

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from generate import generate_image, init_pipeline

app = FastAPI(title="Hardware Eye")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise pipeline once at startup ──
print("Initialising pipeline (this may take a minute on first run) …")
CTX = init_pipeline()
print(f"Ready — perf_index={CTX['perf_index']:.3f}")


# ═══════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════

@app.post("/api/generate")
def api_generate(
    seed: int | None = Query(None),
    steps: int | None = Query(None),
    guidance_scale: float | None = Query(None),
    landscape: bool = Query(False),
    empty: bool = Query(False),
    no_negative: bool = Query(False),
    cpu_brand: str | None = Query(None),
    gpu_brand: str | None = Query(None),
):
    result = generate_image(
        CTX,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        landscape=landscape,
        empty=empty,
        no_negative=no_negative,
        cpu_brand=cpu_brand,
        gpu_brand=gpu_brand,
    )
    return JSONResponse({
        "image_base64": base64.b64encode(result["image_bytes"]).decode(),
        "cpu_brand": result["cpu_brand"],
        "gpu_brand": result["gpu_brand"],
        "seed": result["seed"],
        "steps": result["steps"],
        "guidance_scale": result["guidance_scale"],
        "perf_index": result["perf_index"],
        "resolution": result["resolution"],
    })


@app.get("/api/status")
def api_status():
    return JSONResponse({
        "perf_index": CTX["perf_index"],
        "hw_profile": CTX["hw_profile"],
    })


# ═══════════════════════════════════════════════════
#  serve the frontend from /docs/ folder
# ═══════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def index():
    import os as _os
    path = _os.path.join(_os.path.dirname(__file__), "docs", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
