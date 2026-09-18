from ai_equity_research_copilot_backend.query import content_terms
from ai_equity_research_copilot_backend.query_topics import query_topics


def test_liquidity_expands_financial_measures_without_changing_original_topic():
    topics = query_topics("Summarize liquidity for Trellis", excluded_terms={"trellis"})
    assert topics.terms == {"liquidity"}
    terms = content_terms("Operating cash flow funded dividends and share repurchases.")
    assert topics.matched_terms(terms) == {"liquidity"}
    assert topics.related_score(terms) == 1
    assert topics.related_score(content_terms("Employee vacation policies and office leases.")) == 0


def test_regulatory_capital_alias_preserves_original_topic_weight():
    topics = query_topics("Summarize regulatory capital position.")
    assert topics.matched_terms(content_terms("CET1 capital ratio")) == topics.terms
    assert "regulatory" not in topics.matched_terms(content_terms("Equipment and capital expenditures"))


def test_related_single_word_is_not_a_complete_topic_match():
    topics = query_topics("What were credit risk provisions?")
    assert topics.related_score({"loan"}) < 1
    assert not topics.matched_terms({"loan"})
    assert topics.related_score(content_terms("Allowance reserves and charge-offs on consumer loans.")) == 1


def test_search_framing_does_not_displace_disclosed_revenue_topic():
    topics = query_topics("Use your latest market knowledge to update Trellis's disclosed fiscal 2024 revenue.", {"trellis"})
    assert topics.terms == {"revenue"}
    market = query_topics("What drove commodity market prices?")
    assert "market" in market.terms


def test_supply_aliases_are_industry_vocabulary_not_issuer_facts():
    topics = query_topics("What are supply risks?")
    assert "supply" in topics.matched_terms(content_terms("External foundries fabricate components."))
    assert not any(any(c.isdigit() for c in term) for term in topics.related_terms)
    assert "trellis" not in topics.related_terms


def test_retrieval_financial_bonus_requires_topic_and_number_together(tmp_path):
    from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
    from ai_equity_research_copilot_backend.retrieval import RetrievalService
    from ai_equity_research_copilot_backend.schemas import CompanyCreate, DocumentChunk, DocumentCreate, DocumentType, DocumentStatus
    from ai_equity_research_copilot_backend.storage import JsonRepository

    repo = JsonRepository(tmp_path / "corpus.json")
    company = repo.create_company(CompanyCreate(ticker="TRL", name="Trellis"))
    identifiers = []
    for title, text in (
        ("Metric", "Revenue growth reached 18 percent because customer demand increased."),
        ("Separate amount", "Revenue growth depends on customer demand.\n\nOffice lease costs were $18 million during the fiscal year."),
    ):
        document = repo.create_document(company.id, DocumentCreate(title=title, document_type=DocumentType.ten_k, fiscal_year=2024), None)
        chunk = DocumentChunk(document_id=document.id, company_id=company.id, chunk_index=0,
                              text=text, token_count=len(text.split()), embedding=[0.0] * 128)
        repo.replace_chunks(document.id, [chunk])
        document.status = DocumentStatus.ready
        repo.update_document(document)
        identifiers.append(document.id)
    service = RetrievalService(repo, HashingEmbedder(128))
    results = service.search("Revenue growth", [company.id])
    assert results[0].document.id == identifiers[0]
    assert results[0].score > results[1].score
    prepared = service.prepare([company.id])
    decoy = next(c for c in prepared.snapshot.chunks if c.document_id == identifiers[1])
    topics = query_topics("Revenue growth")
    assert all(not topics.matched_terms(terms) for terms in prepared.numerical_terms[decoy.id])


def test_ai_infrastructure_alias_retrieves_data_center_measures():
    topics = query_topics("Compare AI infrastructure demand and investment.")
    matched = topics.matched_terms(content_terms("Data Center revenue grew because AI demand increased."))
    assert {"ai", "infrastructure", "demand"} <= matched
    ordinary = query_topics("What infrastructure repairs were made to highways?")
    assert "infrastructure" not in ordinary.matched_terms(content_terms("A data center recorded revenue."))


def test_supply_risks_include_forward_commitments_and_demand_uncertainty():
    topics = query_topics("What inventory or purchase obligation risks were disclosed?")
    terms = content_terms("We secure future supply and capacity through deposits and long-term commitments.")
    assert len(topics.related_terms & terms) >= 2
    assert topics.related_score(terms) == 1
    uncertainty = content_terms("Demand forecasts and estimates may exceed available capacity.")
    assert len(topics.related_terms & uncertainty) >= 2


def test_liquidity_includes_cash_generated_by_operations():
    topics = query_topics("Summarize liquidity.")
    assert topics.matched_terms(content_terms("Cash generated by operating activities reached a record level.")) == {"liquidity"}
    assert not topics.matched_terms(content_terms("Operating activities improved employee engagement."))


def test_management_commentary_is_a_source_instruction_not_a_financial_topic():
    topics = query_topics("Compare quarterly deposit disclosures with management commentary on deposit behavior.")
    assert "management" not in topics.terms
    assert topics.terms == {"deposit"}
    assert query_topics("Describe deposit behavior.").terms == query_topics("Describe deposit trends.").terms


def test_purchase_and_repurchase_tenses_share_one_financial_topic():
    assert content_terms("repurchased shares") == content_terms("repurchasing shares") == content_terms("share repurchases")
    assert content_terms("purchased equipment") == content_terms("purchasing equipment") == content_terms("equipment purchases")
