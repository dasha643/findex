import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional
from pathlib import Path  # <--- ДОДАНО

from fastapi import Depends, FastAPI, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from findex.numpy_index import NumpyInvertedIndex
from findex.embedder import SemanticIndex
# Імпортуємо generate_snippet
from findex.search import search_numpy_bm25, reciprocal_rank_fusion, generate_snippet
from findex.indexer import tokenize
from .config import settings

logger = logging.getLogger("findex.web")

# Глобальні змінні
_index_instance: Optional[NumpyInvertedIndex] = None
_semantic_instance: Optional[SemanticIndex] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Завантаження індексів при старті."""
    global _index_instance, _semantic_instance
    try:
        logger.info(f"Loading NumPy index from {settings.index_path}")
        idx = NumpyInvertedIndex()
        idx.load_pickle(settings.index_path)
        _index_instance = idx
        logger.info("NumPy Index loaded successfully")

        try:
            logger.info("Loading embeddings...")
            sem = SemanticIndex()
            sem.load("embeddings.npz")
            _semantic_instance = sem
            logger.info("Embeddings loaded successfully")
        except Exception as e:
            logger.warning(f"Embeddings not loaded (semantic search disabled): {e}")
            _semantic_instance = None

    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        _index_instance = None

    yield

    logger.info("Shutting down, releasing memory")
    _index_instance = None
    _semantic_instance = None


app = FastAPI(title="findex Web API", version="1.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory="src/findex/web/templates")


# --- Залежності ---
def get_index() -> NumpyInvertedIndex:
    if _index_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not loaded. Please check server logs."
        )
    return _index_instance


# --- Pydantic Моделі ---
class SearchResult(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    page: int
    k: int
    total: int
    results: List[SearchResult]


class DocResponse(BaseModel):
    doc_id: int
    title: str
    text: str


class StatsResponse(BaseModel):
    num_docs: int
    num_terms: int


# --- Обробник помилок ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = str(uuid.uuid4())
    logger.error(f"Request {req_id} failed: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": req_id}
    )


# --- Маршрути API ---
@app.get("/health")
def health(index: NumpyInvertedIndex = Depends(get_index)):
    return {"status": "ok", "semantic_enabled": _semantic_instance is not None}


@app.get("/stats", response_model=StatsResponse)
def stats(index: NumpyInvertedIndex = Depends(get_index)):
    return StatsResponse(num_docs=len(index.docs), num_terms=len(index.index))


@app.get("/search", response_model=SearchResponse)
def search(
        q: str = Query(..., min_length=1),
        page: int = Query(1, ge=1),
        k: int = Query(10, ge=1, le=100),
        mode: str = Query("bm25", description="bm25, semantic, or hybrid"),
        index: NumpyInvertedIndex = Depends(get_index),
):
    # 1. BM25 Результати
    bm25_res = []
    if mode in ("bm25", "hybrid"):
        query_terms = list(tokenize(q))
        bm25_res = search_numpy_bm25(index, query_terms, top_k=k * 2)

    # 2. Семантичні результати
    semantic_res = []
    if mode in ("semantic", "hybrid"):
        if _semantic_instance is None:
            logger.warning("Semantic search requested but model not loaded")
            if mode == "semantic":
                raise HTTPException(status_code=503, detail="Semantic search is not available (embeddings not loaded)")
        else:
            semantic_res = _semantic_instance.search(q, k=k * 2)

    # 3. Злиття
    if mode == "bm25":
        final_res = bm25_res
    elif mode == "semantic":
        final_res = semantic_res
    else:
        final_res = reciprocal_rank_fusion(bm25_res, semantic_res)

    # 4. Пагінація
    total = len(final_res)
    start = (page - 1) * k
    paginated = final_res[start:start + k]

    # 5. Формування відповіді з реальними сніпетами
    results = []
    # Беремо перше слово з запиту для пошуку в тексті
    highlight_term = q.split()[0] if q.split() else q

    for doc_id, score in paginated:
        doc_meta = index.docs.get(doc_id)
        title = doc_meta.title if doc_meta else f"Документ {doc_id}"

        snippet = "Текст недоступний."
        if doc_meta and doc_meta.title:
            try:
                # Читаємо файл (перші 10000 символів, щоб не вантажити пам'ять)
                text_content = Path(doc_meta.title).read_text(encoding="utf-8")[:10000]
                snippet = generate_snippet(text_content, highlight_term)
            except Exception as e:
                logger.warning(f"Could not read snippet for {title}: {e}")

        results.append(SearchResult(
            doc_id=doc_id,
            title=title,
            score=float(score),
            snippet=snippet
        ))

    return SearchResponse(query=q, page=page, k=k, total=total, results=results)


@app.get("/docs/{doc_id}", response_model=DocResponse)
def get_document(doc_id: int, index: NumpyInvertedIndex = Depends(get_index)):
    doc_meta = index.docs.get(doc_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found")

    # Читаємо повний текст документа
    text = "Текст документа недоступний."
    if doc_meta.title:
        try:
            text = Path(doc_meta.title).read_text(encoding="utf-8")
        except Exception:
            pass

    return DocResponse(
        doc_id=doc_id,
        title=doc_meta.title,
        text=text
    )


# --- Фронтенд ---
@app.get("/")
def index_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")