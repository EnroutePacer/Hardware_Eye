"""
Hardware Eye — local HTTP server.

Start:   python server.py
Then open docs/index.html in a browser (or visit http://127.0.0.1:8000).
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.generate import generate_image, init_pipeline

app = FastAPI(title="Hardware Eye")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize pipeline once at startup ──
print("Initializing pipeline (this may take a minute on first run) …")
CTX = init_pipeline()
print(f"Ready — perf_index={CTX['perf_index']:.3f}")

_gen_lock = threading.Lock()  # serialise generation requests


# ═══════════════════════════════════════════════════
#  SSE-based generation (progress + result)
# ═══════════════════════════════════════════════════

async def _sse_generator(q: queue.Queue):
    """Yield SSE events from the queue until 'done' sentinel."""
    while True:
        try:
            data = await asyncio.to_thread(q.get, timeout=30)
        except queue.Empty:
            yield "event: ping\ndata: {}\n\n"
            continue
        if data is None:
            break  # sentinel
        kind = data.get("kind", "progress")
        yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        if kind == "done":
            break


@app.get("/api/generate")
async def api_generate(
    seed: int | None = Query(None),
    steps: int | None = Query(None),
    guidance_scale: float | None = Query(None),
    landscape: bool = Query(False),
    empty: bool = Query(False),
    no_negative: bool = Query(False),
    cpu_brand: str | None = Query(None),
    gpu_brand: str | None = Query(None),
):
    """Stream generation progress via SSE, final event contains base64 image."""
    q: queue.Queue = queue.Queue()

    def _run():
        with _gen_lock:
            def on_progress(stage: str, pct: int, msg: str):
                q.put({"kind": "progress", "stage": stage, "pct": pct, "msg": msg})

            try:
                result = generate_image(
                    CTX,
                    seed=seed, steps=steps, guidance_scale=guidance_scale,
                    landscape=landscape, empty=empty, no_negative=no_negative,
                    cpu_brand=cpu_brand, gpu_brand=gpu_brand,
                    progress_callback=on_progress,
                )
                q.put({
                    "kind": "progress", "stage": "done", "pct": 100,
                    "msg": f"seed={result['seed']}  {result['cpu_brand']}{' + ' + result['gpu_brand'] if result['gpu_brand'] else ''}",
                })
                q.put({
                    "kind": "done",
                    "image_base64": base64.b64encode(result["image_bytes"]).decode(),
                    "cpu_brand": result["cpu_brand"],
                    "gpu_brand": result["gpu_brand"],
                    "seed": result["seed"],
                    "steps": result["steps"],
                    "guidance_scale": result["guidance_scale"],
                    "perf_index": result["perf_index"],
                    "resolution": result["resolution"],
                })
            except Exception as e:
                q.put({"kind": "error", "msg": str(e)})
                q.put(None)
            finally:
                q.put(None)

    threading.Thread(target=_run, daemon=True).start()
    return StreamingResponse(_sse_generator(q), media_type="text/event-stream")


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
