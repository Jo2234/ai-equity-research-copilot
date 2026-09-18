
import pytest
from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.retrieval import RetrievalService
from ai_equity_research_copilot_backend.retrieval_query import (
    company_clause_queries,
    contextual_passages,
)
from ai_equity_research_copilot_backend.schemas import (
    Company,
    CompanyCreate,
    DocumentChunk,
    DocumentCreate,
    DocumentStatus,
    DocumentType,
)
from ai_equity_research_copilot_backend.storage import JsonRepository


def test_separate_topics_require_two_distinct_unambiguous_company_clauses():
    maple = Company(name="Maple Finance Corporation", ticker="MPL")
    cedar = Company(name="Cedar Energy Corporation", ticker="CDR")
    companies = [maple, cedar]
    query = "Compare Maple's sensitivity to interest rates with Cedar’s sensitivity to commodities in annual 2023 filings"
    clauses = company_clause_queries(query, companies)
    assert "interest rates" in clauses[maple.id] and "commodities" not in clauses[maple.id]
    assert "commodities" in clauses[cedar.id] and "interest" not in clauses[cedar.id]
    for ambiguous in (
        "Compare Maple with Cedar on revenue growth",
        "Compare Maple and Cedar sensitivity to rates",
        "Compare Maple revenue with Cedar revenue in annual 2023 filings",
        "Compare Maple sensitivity to rates with its sensitivity to commodities",
        "Compare Maple and Cedar revenue with Cedar income",
        "Compare Maple sensitivity with Cedar sensitivity to commodities",
        "Compare Maple sensitivity to rates with Cedar exposure with respect to commodities",
    ):
        assert company_clause_queries(ambiguous, companies) == {c.id: ambiguous for c in companies}
    other = Company(name="Maple Energy Corporation", ticker="MLE")
    ambiguous = "Compare Maple sensitivity to rates with MLE commodity exposure"
    assert set(company_clause_queries(ambiguous, [maple, other]).values()) == {ambiguous}


def add(repo, company, text, *, year=2023, metadata=None, document_type=DocumentType.ten_k):
    doc = repo.create_document(company.id, DocumentCreate(
        title=f"{document_type.value} {year}", document_type=document_type, fiscal_year=year), None)
    chunk = DocumentChunk(company_id=company.id, document_id=doc.id, chunk_index=0,
                          text=text, token_count=30, embedding=HashingEmbedder().embed(text),
                          metadata=metadata or {})
    repo.replace_chunks(doc.id, [chunk])
    doc.status = DocumentStatus.ready
    repo.update_document(doc)
    return chunk


def test_comparison_ranking_keeps_owner_topics_and_original_period_policy(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    maple = repo.create_company(CompanyCreate(name="Maple", ticker="MPL"))
    cedar = repo.create_company(CompanyCreate(name="Cedar", ticker="CDR"))
    rate = add(repo, maple, "Interest rate sensitivity reduced net interest income by $2 million as deposit balances declined.")
    add(repo, maple, "Commodity sensitivity increased earnings by $3 million as commodity prices rose.")
    commodity = add(repo, cedar, "Commodity sensitivity reduced earnings by $7 million as production volume and commodity prices declined.")
    add(repo, cedar, "Interest rate sensitivity increased net interest income by $8 million as interest rates and deposit balances rose.")
    add(repo, cedar, commodity.text, year=2024)
    service = RetrievalService(repo, HashingEmbedder())
    query = "Compare Maple sensitivity to interest rates with Cedar sensitivity to commodities in annual 2023 filings"
    results = service.search(query, [maple.id, cedar.id], top_k=2)
    assert {r.chunk.id for r in results} == {rate.id, commodity.id}
    assert all(r.document.fiscal_year == 2023 and r.query == query for r in results)
    assert service.search(query, [maple.id, cedar.id], top_k=2) == results


@pytest.mark.parametrize("metadata", [False, True])
def test_heading_context_promotes_actual_segment_driver_without_cross_section_leak(tmp_path, metadata):
    repo = JsonRepository(tmp_path / "state.json")
    company = repo.create_company(CompanyCreate(name="Cedar", ticker="CDR"))
    heading = "2023 Upstream Earnings Driver Analysis"
    body = "Lower realizations decreased earnings by $9 million, primarily driven by lower crude prices."
    text = body if metadata else heading + "\n\n" + body
    fields = {"section_spans": [{"start": 0, "end": len(body), "section": heading}]} if metadata else {}
    relevant = add(repo, company, text, metadata=fields)
    add(repo, company, "Upstream commodity operational earnings depend on commodity prices, production volumes, and general economic conditions.")
    unrelated = add(repo, company, "2023 Downstream Earnings Driver Analysis\n\nMargins increased earnings by $2 million from higher refining prices.")
    service = RetrievalService(repo, HashingEmbedder())
    results = service.search("What commodity factors drove upstream earnings?", [company.id], top_k=3)
    assert results[0].chunk.id == relevant.id
    prepared = service.prepare([company.id])
    assert "upstream" not in prepared.terms[unrelated.id]
    assert all("upstream" not in terms for terms in prepared.passage_terms[unrelated.id])
    assert results[0].chunk.text == text  # Discovery features never rewrite evidence.


def test_infrastructure_investment_discovers_quantified_ppe_without_relabeling_it(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    company = repo.create_company(CompanyCreate(name="Birch", ticker="BRCH"))
    capital = add(repo, company, "Cash used in investing increased by $8 billion, including a $3 billion increase in additions to property and equipment.")
    add(repo, company, "We invest in AI and cloud infrastructure investment to meet customer demand and support future capacity.")
    service = RetrievalService(repo, HashingEmbedder())
    results = service.search("How did AI cloud infrastructure investment change?", [company.id])
    discovered = next(result for result in results[:2] if result.chunk.id == capital.id)
    assert discovered.chunk.text == capital.text and "AI" not in capital.text
    # The bonus is limited to this discovery intent, and never invents a quantity.
    prepared = service.prepare([company.id])
    assert prepared.capital_investment[capital.id]
    assert capital.id not in {r.chunk.id for r in service.search("Supplier inventory manufacturing", [company.id])}


def test_invalid_section_spans_and_final_chunk_heading_are_not_borrowed():
    company = Company(name="Cedar", ticker="CDR")
    chunk = DocumentChunk(company_id=company.id, document_id=company.id, text="Cash rose by $2 million.",
                          chunk_index=0, token_count=5, embedding=[], metadata={
                              "section_title": "Upstream", "section_spans": [
                                  {"start": -1, "end": 999, "section": "Upstream"},
                                  {"start": 0, "end": 5, "section": None}]})
    assert not contextual_passages(chunk)
    assert not contextual_passages(chunk.model_copy(update={"metadata": {"section_title": "Upstream"}}))


def test_quantified_sensitivity_under_financial_heading_beats_generic_price_exposure(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    company = repo.create_company(CompanyCreate(name="Cedar", ticker="CDR"))
    body = "Lower crude prices decreased earnings by $6 million, due to lower realizations and production volume."
    actual = add(repo, company, body, metadata={"section_spans": [
        {"start": 0, "end": len(body), "section": "2023 Upstream Earnings Driver Analysis"}]})
    add(repo, company, "Commodity price sensitivity may affect earnings, production volume, depreciation and costs in future periods.")
    service = RetrievalService(repo, HashingEmbedder())
    results = service.search("Commodity sensitivity", [company.id])
    assert results[0].chunk.id == actual.id


def test_direct_financial_aliases_remain_specific():
    from ai_equity_research_copilot_backend.query import content_terms
    from ai_equity_research_copilot_backend.query_topics import query_topics

    topics = query_topics("Liquidity")
    assert topics.matched_terms(content_terms("Capital Resources")) == {"liquidity"}
    topics = query_topics("Commodity sensitivity")
    assert topics.matched_terms(content_terms("lower crude prices")) == {"commodity"}
    assert not topics.matched_terms(content_terms("lower interest rates"))
    assert not topics.matched_terms(content_terms("lower share prices"))


def test_source_family_comparison_ranks_filing_geography_and_release_demand_separately(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    company = repo.create_company(CompanyCreate(name="Maple", ticker="MPL"))
    foreign = repo.create_company(CompanyCreate(name="Birch", ticker="BRCH"))
    geography = "Geographic sales increased 12 percent in northern regions, driven by stronger regional sales."
    demand = "Customer demand strengthened for new products as retail customers upgraded their devices."
    filing = add(repo, company, geography)
    release = add(repo, company, demand, document_type=DocumentType.eight_k)
    add(repo, company, demand)
    add(repo, company, geography, document_type=DocumentType.eight_k)
    add(repo, foreign, geography)
    add(repo, company, geography, year=2024)
    service = RetrievalService(repo, HashingEmbedder())
    query = "Compare Maple geographic sales in the 2023 annual filing with customer demand in the earnings release"
    results = service.search(query, [company.id], top_k=2)
    assert {r.chunk.id for r in results} == {filing.id, release.id}
    assert all(r.document.fiscal_year == 2023 and r.query == query for r in results)
    filtered = service.search(query, [company.id], document_types=[DocumentType.ten_k])
    assert filtered and filtered[0].chunk.id == filing.id
    assert all(r.document.document_type == DocumentType.ten_k for r in filtered)
    assert service.search(query, [company.id], top_k=2) == results


def test_source_family_comparison_ranks_quarter_deposits_and_management_behavior(tmp_path):
    repo = JsonRepository(tmp_path / "state.json")
    company = repo.create_company(CompanyCreate(name="Cedar", ticker="CDR"))
    balance = "Deposit balances increased by $11 million following growth in average account balances."
    behavior = "Customer behavior shifted toward longer duration investments as customers sought greater flexibility."
    filing = add(repo, company, balance, document_type=DocumentType.ten_q)
    transcript = add(repo, company, behavior, document_type=DocumentType.earnings_transcript)
    add(repo, company, behavior, document_type=DocumentType.ten_q)
    add(repo, company, balance, document_type=DocumentType.earnings_transcript)
    service = RetrievalService(repo, HashingEmbedder())
    query = "Compare quarterly deposit balance disclosures with management commentary on customer behavior in 2023"
    results = service.search(query, [company.id], top_k=2)
    assert {r.chunk.id for r in results} == {filing.id, transcript.id}
    # A later ordinary query must not retain the per-source comparison hint.
    ordinary = service.search("Customer behavior", [company.id], document_types=[DocumentType.ten_q])
    assert ordinary and ordinary[0].chunk.text == behavior
