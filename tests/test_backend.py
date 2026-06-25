from fastapi.testclient import TestClient

from backend.main import app, CHUNKS, COMPANIES, DOCUMENTS


client = TestClient(app)


def test_company_list_seeded():
    response = client.get("/companies")
    assert response.status_code == 200
    assert any(row["ticker"] == "NVDA" for row in response.json())


def test_chat_returns_citations():
    company_id = next(cid for cid, c in COMPANIES.items() if c["ticker"] == "NVDA")
    response = client.post("/research/chat", json={"company_ids": [company_id], "question": "What drove Data Center revenue growth?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert "answer" in payload


def test_delete_document_removes_chunks():
    doc_id = next(iter(DOCUMENTS))
    chunk_ids = [cid for cid, c in CHUNKS.items() if c["document_id"] == doc_id]
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert all(cid not in CHUNKS for cid in chunk_ids)
