from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import AppConfig, RAGConfig
from app.server import create_app


def build_client() -> TestClient:
    config = AppConfig(rag=RAGConfig(vector_store="in_memory"))
    app = create_app(config)
    return TestClient(app)


def test_health_endpoint():
    client = build_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_path_success():
    client = build_client()
    response = client.post("/ingest-path", json={"path": "data/docs"})
    assert response.status_code == 200
    assert response.json()["ingested"] >= 1


def test_ingest_path_outside_project():
    client = build_client()
    response = client.post("/ingest-path", json={"path": "/tmp"})
    assert response.status_code == 200
    assert response.json()["error"] == "path_outside_project"


def test_ingest_path_missing():
    client = build_client()
    response = client.post("/ingest-path", json={"path": "data/does_not_exist"})
    assert response.status_code == 200
    assert response.json()["error"] == "path_not_found"
