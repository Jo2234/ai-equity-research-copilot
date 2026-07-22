"""The former prototype entry point delegates to the tested package backend."""

from fastapi.testclient import TestClient

from backend.main import create_app


def test_legacy_entry_point_uses_packaged_storage_and_contract(tmp_path):
    app = create_app(tmp_path, seed=False)
    client = TestClient(app)
    company_id = client.post("/companies", json={"ticker": "TEST", "name": "Test Company"}).json()["id"]
    uploaded = client.post(f"/companies/{company_id}/documents", json={
        "title": "Revenue note", "document_type": "manual_note", "filename": "note.txt",
        "text": "Revenue growth was supported by stronger software subscription demand.",
    })
    assert uploaded.status_code == 201
    document = uploaded.json()
    assert document["status"] == "ready"
    assert document["chunk_count"] == 1
    # Reopening storage and deleting through the packaged API replaces the old
    # globals-only tests; the canonical endpoint tests cover chat and citations.
    reopened = TestClient(create_app(tmp_path, seed=False))
    assert reopened.get(f"/documents/{document['id']}").status_code == 200
    assert reopened.delete(f"/documents/{document['id']}").status_code == 204
    assert reopened.get(f"/documents/{document['id']}").status_code == 404
