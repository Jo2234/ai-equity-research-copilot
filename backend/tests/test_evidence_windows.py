from ai_equity_research_copilot_backend.evidence import evidence_passages


RESULT = "Subscription revenue increased 28 percent to $8.4 billion this year."
CAUSE = "The increase was driven by enterprise subscriptions and higher renewals."
OFFSET = "This growth was partly offset by lower usage among smaller customers."


def test_three_sentence_financial_chain_retains_source_order_and_smaller_windows():
    source = f"{RESULT} {CAUSE} {OFFSET}"
    passages = evidence_passages(source)
    assert source in passages
    assert RESULT in passages
    assert f"{RESULT} {CAUSE}" in passages
    assert all(passage in source for passage in passages)


def test_three_sentence_window_never_crosses_paragraph_boundary():
    source = f"{RESULT} {CAUSE}\n\n{OFFSET}"
    assert f"{RESULT} {CAUSE} {OFFSET}" not in evidence_passages(source)
    assert f"{RESULT} {CAUSE}" in evidence_passages(source)


def test_no_cross_document_state_is_retained():
    evidence_passages(f"{RESULT} {CAUSE}")
    assert evidence_passages(OFFSET) == [OFFSET]


def test_unlabelled_numeric_row_does_not_become_third_sentence_context():
    row = "Enterprise subscriptions $8,400 $6,562 28%."
    source = f"{RESULT} {row} {OFFSET}"
    assert source not in evidence_passages(source)


def test_incomplete_final_sentence_is_not_a_three_sentence_window():
    fragment = "The remaining increase was primarily driven by"
    source = f"{RESULT} {CAUSE} {fragment}"
    assert source not in evidence_passages(source)


def test_three_sentence_candidates_require_explanation_not_just_three_metrics():
    source = (f"{RESULT} "
              "Operating income increased 18 percent during the current fiscal year. "
              "Cash generated from operations was $2.4 billion in the current year.")
    assert source not in evidence_passages(source)


def test_three_sentence_windows_are_bounded_but_can_retain_longer_causal_chain():
    detail = "Enterprise clients expanded adoption across regions and renewed their subscriptions. "
    # Each sentence remains shorter than the existing 650-character limit;
    # the complete contiguous chain crosses that limit without reaching 1000.
    first = "Subscription revenue increased 28 percent to $8.4 billion " + "through enterprise adoption " * 7 + "."
    second = "The growth was driven by " + "expanded usage of hosted services across customer groups " * 5 + "."
    third = "This was partly offset by " + "lower transaction volume in smaller customer accounts " * 4 + "."
    source = f"{first} {second} {third}"
    assert 650 < len(source) <= 1000
    assert source in evidence_passages(source)
    oversized = f"{first} {second} {third[:-1]} {'additional customer migration effects ' * 10}."
    assert len(oversized) > 1000
    assert oversized not in evidence_passages(oversized)
    # Never grow a fourth-sentence candidate to retain unrelated nearby prose.
    assert source + " " + detail.strip() not in evidence_passages(source + " " + detail)


def test_bullet_lists_do_not_form_three_sentence_narratives():
    source = f"• {RESULT} • {CAUSE} • {OFFSET}"
    assert source not in evidence_passages(source)


def test_qualitative_supplier_dependency_keeps_explicit_mitigation_continuation():
    dependency = "Certain specialized components are sourced from single suppliers at overseas locations."
    mitigation = "Where multiple sources are available, we qualify alternative suppliers to minimize production risks."
    source = dependency + " " + mitigation
    assert source in evidence_passages(source)
    assert dependency in evidence_passages(source)
    assert mitigation in evidence_passages(source)
    assert source not in evidence_passages(dependency + "\n\n" + mitigation)


def test_qualitative_dependency_keeps_referential_risk_explanation():
    first = "The company relies on external suppliers for specialized manufacturing capacity."
    second = "Disruptions at these suppliers may delay deliveries and interrupt production."
    assert first + " " + second in evidence_passages(first + " " + second)


def test_qualitative_financial_cause_does_not_require_a_number():
    first = "Operating margins declined during the current fiscal period."
    second = "The decline was driven by higher freight charges and manufacturing expenses."
    assert first + " " + second in evidence_passages(first + " " + second)


def test_unrelated_qualitative_neighbor_is_not_automatically_joined():
    first = "The company relies on external suppliers for specialized manufacturing capacity."
    second = "The board appointed a new chief executive during the current fiscal period."
    assert first + " " + second not in evidence_passages(first + " " + second)
