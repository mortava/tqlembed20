import asyncio
import functools
import os
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel

from src.query import FileSearchQuery
from src.store import FileSearchStoreManager
from src.uploader import FileUploader

load_dotenv()

app = FastAPI(title="Quinn — Gemini RAG Agent")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

# Quinn talks to exactly one File Search Store. Pinning the name server-side
# means a buggy or stale frontend can never spawn a stray store or query the
# wrong KB. Override via env var if you ever need to point at a staging store.
PINNED_STORE_NAME = os.getenv(
    "QUINN_STORE_NAME",
    "fileSearchStores/quinn-knowledge-base-geg6qsemdjxo",
)
PINNED_STORE_DISPLAY = os.getenv("QUINN_STORE_DISPLAY", "Quinn Knowledge Base")

client = genai.Client(api_key=api_key)
store_mgr = FileSearchStoreManager(client)
uploader = FileUploader(client)
query_engine = FileSearchQuery(client)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── Pinned store ──────────────────────────────────────────────────────────────

@app.get("/api/store")
async def pinned_store():
    """Return the single pinned File Search Store the frontend should use."""
    return {"name": PINNED_STORE_NAME, "display_name": PINNED_STORE_DISPLAY}


# ── Stores (diagnostic only — frontend uses /api/store) ──────────────────────

@app.get("/api/stores")
async def list_stores():
    stores = await asyncio.to_thread(store_mgr.list)
    return {
        "stores": [
            {"name": s.name, "display_name": getattr(s, "display_name", s.name)}
            for s in stores
        ]
    }


@app.get("/api/stores/documents")
async def list_documents():
    """List documents in the pinned store."""
    docs = await asyncio.to_thread(
        functools.partial(store_mgr.list_documents, PINNED_STORE_NAME)
    )
    return {
        "store_name": PINNED_STORE_NAME,
        "documents": [
            {
                "name": d.name,
                "display_name": getattr(d, "display_name", d.name),
            }
            for d in docs
        ],
    }


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    display_name: Optional[str] = Form(None),
    # store_name is accepted for backward compat but ignored — uploads always
    # land in the pinned store.
    store_name: Optional[str] = Form(None),
):
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    resolved_display = display_name or file.filename or "upload"
    try:
        await asyncio.to_thread(
            functools.partial(
                uploader.upload_direct,
                tmp_path,
                PINNED_STORE_NAME,
                resolved_display,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return {"ok": True, "display_name": resolved_display, "store_name": PINNED_STORE_NAME}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    # store_names accepted for backward compat but ignored — queries always
    # hit the pinned store.
    store_names: Optional[list[str]] = None
    metadata_filter: Optional[str] = None


@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        result = await asyncio.to_thread(
            functools.partial(
                query_engine.ask_with_citations,
                req.question,
                [PINNED_STORE_NAME],
                req.metadata_filter,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
