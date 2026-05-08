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

client = genai.Client(api_key=api_key)
store_mgr = FileSearchStoreManager(client)
uploader = FileUploader(client)
query_engine = FileSearchQuery(client)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── Stores ────────────────────────────────────────────────────────────────────

class CreateStoreRequest(BaseModel):
    display_name: str
    embedding_model: str = "models/gemini-embedding-2"


@app.get("/api/stores")
async def list_stores():
    stores = await asyncio.to_thread(store_mgr.list)
    return {
        "stores": [
            {"name": s.name, "display_name": getattr(s, "display_name", s.name)}
            for s in stores
        ]
    }


@app.post("/api/stores")
async def create_store(req: CreateStoreRequest):
    store = await asyncio.to_thread(
        functools.partial(store_mgr.create, req.display_name, req.embedding_model)
    )
    return {"name": store.name, "display_name": getattr(store, "display_name", store.name)}


@app.delete("/api/stores")
async def delete_store(name: str):
    await asyncio.to_thread(functools.partial(store_mgr.delete, name))
    return {"ok": True}


@app.get("/api/stores/documents")
async def list_documents(store_name: str):
    docs = await asyncio.to_thread(
        functools.partial(store_mgr.list_documents, store_name)
    )
    return {"documents": [{"name": d.name} for d in docs]}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    store_name: str = Form(...),
    display_name: str = Form(...),
):
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        await asyncio.to_thread(
            functools.partial(uploader.upload_direct, tmp_path, store_name, display_name)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return {"ok": True, "display_name": display_name}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    store_names: list[str]
    metadata_filter: Optional[str] = None


@app.post("/api/query")
async def query(req: QueryRequest):
    try:
        result = await asyncio.to_thread(
            functools.partial(
                query_engine.ask_with_citations,
                req.question,
                req.store_names,
                req.metadata_filter,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
