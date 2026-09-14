from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == "BharatTrade AI"
    assert data["version"] == "0.1.0"
    assert data["documentation"] == "/docs"


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["application"] == "BharatTrade AI"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "development"
    assert "timestamp" in data