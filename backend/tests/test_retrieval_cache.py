from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.retrieval import RetrievalService
from ai_equity_research_copilot_backend.schemas import (
    CompanyCreate,
    DocumentChunk,
    DocumentCreate,
    DocumentStatus,
    DocumentType,
    Message,
    StoredCitation,
)
from ai_equity_research_copilot_backend.storage import JsonRepository


def add_document(repo, company, title="Annual", text="Revenue grew 14 percent due to stronger enterprise demand."):
    document = repo.create_document(company.id, DocumentCreate(
        title=title, fiscal_year=2025, document_type=DocumentType.ten_k,
    ), None)
    chunk = DocumentChunk(document_id=document.id, company_id=company.id,
                          chunk_index=0, text=text, token_count=12,
                          embedding=HashingEmbedder().embed(text))
    repo.replace_chunks(document.id, [chunk])
    document.status = DocumentStatus.ready
    repo.update_document(document)
    return document, chunk


def setup_repo(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    first = repo.create_company(CompanyCreate(ticker="ONE", name="Company One"))
    second = repo.create_company(CompanyCreate(ticker="TWO", name="Company Two"))
    add_document(repo, first)
    add_document(repo, second, "Foreign", "Revenue grew 99 percent due to stronger enterprise demand.")
    return repo, first, second, RetrievalService(repo, HashingEmbedder())


def test_warm_cache_preserves_scores_filters_and_company_isolation(tmp_path):
    repo, first, second, service = setup_repo(tmp_path)
    params = {"query": "Revenue growth", "company_ids": [first.id],
              "document_types": [DocumentType.ten_k], "fiscal_years": [2025]}
    cold = service.search(**params)
    assert cold
    warm = service.search(**params)
    uncached = RetrievalService(repo, HashingEmbedder()).search(**params)
    assert [r.model_dump() for r in cold] == [r.model_dump() for r in warm] == [r.model_dump() for r in uncached]
    assert {r.company.id for r in warm} == {first.id}
    assert {r.company.id for r in service.search("Revenue growth", [second.id])} == {second.id}
    assert not service.search("Revenue growth", [first.id], fiscal_years=[2024])
    assert service.prepare([first.id, second.id]) is service.prepare([second.id, first.id])


def test_history_writes_keep_prepared_corpus_and_passage_cache(tmp_path, monkeypatch):
    repo, company, _, service = setup_repo(tmp_path)
    results = service.search("Revenue growth", [company.id])
    prepared = service.prepare([company.id])
    assert prepared.passage_terms
    conversation = repo.create_conversation("Research")
    message = repo.create_message(Message(conversation_id=conversation.id, role="user", content="Revenue growth"))
    repo.create_citations([StoredCitation(message_id=message.id, document_chunk_id=results[0].chunk.id,
        citation_label="S1", excerpt="Revenue grew", relevance_score=1)])
    def unexpected_read():
        raise AssertionError("Warm preparation reread the persisted corpus")
    monkeypatch.setattr(repo, "_read_state", unexpected_read)
    assert service.prepare([company.id]) is prepared
    assert service.search("Revenue growth", [company.id]) == results


def test_upload_replacement_status_and_deletion_invalidate_next_request(tmp_path):
    repo, company, _, service = setup_repo(tmp_path)
    before = service.prepare([company.id])
    added, chunk = add_document(repo, company, "New")
    assert added.id in {r.document.id for r in service.search("Revenue growth", [company.id])}
    assert service.prepare([company.id]) is not before
    chunk.text = "Margins were negative following unprecedented litigation."
    chunk.embedding = HashingEmbedder().embed(chunk.text)
    repo.replace_chunks(added.id, [chunk])
    assert added.id not in {r.document.id for r in service.search("Revenue growth", [company.id])}
    assert added.id in {r.document.id for r in service.search("Margins litigation", [company.id])}
    added.status = DocumentStatus.failed
    repo.update_document(added)
    assert not service.search("Margins litigation", [company.id])
    repo.delete_document(before.snapshot.chunks[0].document_id)
    assert not service.search("Revenue growth", [company.id])
    # A request's explicitly retained snapshot remains coherent after mutation.
    assert service.search("Revenue growth", [company.id], corpus=before)


def test_external_replacement_is_not_hidden_by_a_local_history_write(tmp_path):
    repo, company, _, service = setup_repo(tmp_path)
    before = service.prepare([company.id])
    external = JsonRepository(repo.state_file)
    external.delete_document(before.snapshot.chunks[0].document_id)
    repo.create_conversation("New history after external deletion")
    assert not service.search("Revenue growth", [company.id])


def test_concurrent_queries_share_coherent_cache_and_refresh_after_deletion(tmp_path):
    repo, company, _, service = setup_repo(tmp_path)
    barrier = Barrier(6)
    def query(_):
        barrier.wait()
        return service.search("Revenue growth", [company.id])
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(query, range(6)))
    assert results[0] and all(rows == results[0] for rows in results)
    repo.delete_document(results[0][0].document.id)
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(not rows for rows in pool.map(query, range(6)))


def test_prepared_scope_cache_is_bounded(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    companies = [repo.create_company(CompanyCreate(ticker=f"C{i}", name=f"Company {i}")) for i in range(9)]
    service = RetrievalService(repo, HashingEmbedder())
    oldest = service.prepare([companies[0].id])
    for company in companies[1:]:
        service.prepare([company.id])
    assert len(service._prepared) == 8
    assert service.prepare([companies[0].id]) is not oldest


def test_external_replacement_during_history_publication_invalidates_cache(tmp_path, monkeypatch):
    repo, company, _, service = setup_repo(tmp_path)
    before = service.prepare([company.id])
    external = JsonRepository(repo.state_file)
    replace = Path.replace
    injected = False

    def replace_then_delete(path, target):
        nonlocal injected
        result = replace(path, target)
        if target == repo.state_file and not injected:
            injected = True
            external.delete_document(before.snapshot.chunks[0].document_id)
        return result

    monkeypatch.setattr(Path, "replace", replace_then_delete)
    repo.create_conversation("History racing with another repository's deletion")
    assert injected
    assert not service.search("Revenue growth", [company.id])
