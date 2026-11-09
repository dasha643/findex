import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import patch

from findex.web.app import app, get_index
from findex.inverted import InvertedIndex
from findex.models import DocMeta, Posting


@pytest.fixture
def mock_index():
    """Створюємо фейковий індекс для тестів (без реального файлу)."""
    idx = InvertedIndex()
    idx.docs[1] = DocMeta(doc_id=1, doc_len=10, title="Test Document 1")
    idx.index["hello"] = [Posting(doc_id=1, term_freq=1)]
    return idx


@pytest.fixture
def client(mock_index):
    """Підміняємо залежність get_index на наш фейковий індекс."""
    app.dependency_overrides[get_index] = lambda: mock_index
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_search_success(client):
    class FakeNode:
        def evaluate(self, index):
            return {1}

    with patch("findex.web.app.search_numpy_bm25", return_value=[(1, 1.0)]):
        response = client.get("/search?q=hello")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "hello"
        assert data["total"] == 1
        assert data["results"][0]["doc_id"] == 1

def test_search_validation_error(client):
    response = client.get("/search")
    assert response.status_code == 422


def test_get_document_not_found(client):
    response = client.get("/docs/999")
    assert response.status_code == 404


def test_health_no_index():
    def mock_get_index():
        raise HTTPException(status_code=503, detail="Index not loaded")

    app.dependency_overrides[get_index] = mock_get_index
    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 503
    app.dependency_overrides.clear()