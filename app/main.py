from contextlib import asynccontextmanager

from fastapi import FastAPI

# Patch ChromaDB RustBindingsAPI shutdown crash (library bug: bindings attr missing on stop)
import chromadb.api.rust as _rust_api

_original_stop = getattr(_rust_api.RustBindingsAPI, "stop", None)


def _patched_stop(self):
    try:
        if _original_stop:
            _original_stop(self)
    except AttributeError:
        pass


if _original_stop:
    _rust_api.RustBindingsAPI.stop = _patched_stop

from app.api import routes_chat, routes_sessions, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import init_db
from app.obs.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")
app.add_exception_handler(HelixError, helix_error_handler)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
