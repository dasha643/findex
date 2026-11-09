import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from findex.inverted import InvertedIndex
from findex.search import parse_query
from .config import settings

logger = logging.getLogger("findex.web")

_index_instance: Optional[InvertedIndex] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Завантаження індексу при старті та очищення при зупинці."""
    global _index_instance
    try:
        logger.info(f"Loading index from {settings.index_path}")
        idx = InvertedIndex()
        idx.load_pickle(settings.index_path)
        _index_instance = idx
        logger.info("Index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        _index_instance = None

    yield

    logger.info("Shutting down, releasing index")
    _index_instance = None


app = FastAPI(title="findex Web API", version="0.7.0", lifespan=lifespan)
templates = Jinja2Templates(directory="src/findex/web/templates")


def get_index() -> InvertedIndex:
    if _index_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not loaded"
        )
    return _index_instance


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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = str(uuid.uuid4())
    logger.error(f"Request {req_id} failed: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "request_id": req_id}
    )


@app.get("/health")
def health(index: InvertedIndex = Depends(get_index)):
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats(index: InvertedIndex = Depends(get_index)):
    return StatsResponse(num_docs=index.num_docs, num_terms=len(index.index))


@app.get("/search", response_model=SearchResponse)
def search(
        q: str = Query(..., min_length=1),
        page: int = Query(1, ge=1),
        k: int = Query(10, ge=1, le=100),
        scorer: str = "default",
        index: InvertedIndex = Depends(get_index),
):
    query_node = parse_query(q)
    results_set = query_node.evaluate(index)
    sorted_results = sorted(results_set)

    total = len(sorted_results)
    start = (page - 1) * k
    end = start + k
    paginated = sorted_results[start:end]

    results = []
    for doc_id in paginated:
        doc_meta = index.docs.get(doc_id)
        results.append(SearchResult(
            doc_id=doc_id,
            title=doc_meta.title if doc_meta else "Unknown",
            score=1.0,
            snippet="..."
        ))

    return SearchResponse(query=q, page=page, k=k, total=total, results=results)


@app.get("/docs/{doc_id}", response_model=DocResponse)
def get_document(doc_id: int, index: InvertedIndex = Depends(get_index)):
    doc_meta = index.docs.get(doc_id)
    if not doc_meta:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocResponse(
        doc_id=doc_id,
        title=doc_meta.title,
        text="Тут має бути текст документа. Якщо ви зберігали його в DocMeta, поверніть його."
    )


@app.get("/")
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})