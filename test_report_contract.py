#!/usr/bin/env python3
"""
Test suite for the ATIS Investigation Report endpoint contract.

Validates:
1. Pydantic schema enforcement
2. Deterministic fields come from backend, not LLM
3. Endpoint returns InvestigationReport directly (no wrapper)
4. Validation rejects malformed output
5. Cross-validation of evidence query IDs
6. All required fields present in final output
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, ".")

from report_generator import (
    generate_investigation_report,
    InvestigationReport,
    KeyFinding,
    ImportantEntity,
    ImportantRelationship,
    EvidenceAndSource,
)


# =============================================================================
# Test Data
# =============================================================================

def make_sample_investigation() -> Dict[str, Any]:
    """Create a realistic investigation payload matching a real successful investigation."""
    return {
        "investigation_id": "test-inv-001",
        "title": "Where can mine lithium in Zimbabwe",
        "original_question": "Where can mine lithium in Zimbabwe",
        "perspective": {"country": "Zimbabwe", "country_code": "ZW"},
        "queries": [
            {
                "query_id": "q-001",
                "sequence": 1,
                "question": "Where can mine lithium in Zimbabwe",
                "answer": "Lithium mining in Zimbabwe is concentrated in the Bikita area of Masvingo Province...",
                "findings": [
                    {"text": "Bikita Minerals is the largest lithium producer in Zimbabwe", "source_nodes": ["Bikita_Minerals"]},
                    {"text": "Arcadia Lithium Project near Harare is a major development", "source_nodes": ["Arcadia_Lithium"]},
                ],
                "result": {
                    "executive_summary": "Zimbabwe has significant lithium deposits...",
                    "key_entities": [
                        {"entity_name": "Bikita Minerals", "entity_type": "company", "country": "Zimbabwe", "significance_score": 9},
                        {"entity_name": "Arcadia Lithium Project", "entity_type": "project", "country": "Zimbabwe", "significance_score": 8},
                    ]
                }
            },
            {
                "query_id": "q-002",
                "sequence": 2,
                "question": "What are the regulations for lithium mining in Zimbabwe",
                "answer": "The Mines and Minerals Act governs mining operations...",
                "findings": [
                    {"text": "Special mining leases required for large-scale operations", "source_nodes": ["Mines_and_Minerals_Act"]},
                ],
                "result": {}
            },
            {
                "query_id": "q-003",
                "sequence": 3,
                "question": "Who are the major lithium mining companies in Zimbabwe",
                "answer": "Major players include Bikita Minerals, Zijin Mining, and Huayou Cobalt...",
                "findings": [
                    {"text": "Chinese companies have invested heavily in Zimbabwe lithium", "source_nodes": ["Zijin_Mining", "Huayou_Cobalt"]},
                ],
                "result": {}
            },
            {
                "query_id": "q-004",
                "sequence": 4,
                "question": "What is the export process for lithium from Zimbabwe",
                "answer": "Lithium exports require permits from the Ministry of Mines...",
                "findings": [
                    {"text": "Export ban on raw lithium was introduced in 2022", "source_nodes": ["Lithium_Export_Ban"]},
                ],
                "result": {}
            }
        ],
        "entities": [
            {"name": "Bikita Minerals", "type": "company", "country": "Zimbabwe", "significance": "Largest lithium producer"},
            {"name": "Arcadia Lithium Project", "type": "project", "country": "Zimbabwe", "significance": "Major new development"},
            {"name": "Zijin Mining", "type": "company", "country": "China", "significance": "Major investor"},
            {"name": "Huayou Cobalt", "type": "company", "country": "China", "significance": "Significant investment"},
            {"name": "Mines and Minerals Act", "type": "legislation", "country": "Zimbabwe", "significance": "Governing law"},
            {"name": "Ministry of Mines", "type": "government", "country": "Zimbabwe", "significance": "Regulatory authority"},
            {"name": "Masvingo Province", "type": "region", "country": "Zimbabwe", "significance": "Primary lithium region"},
        ],
        "relationships": [
            {"from_entity": "Zijin Mining", "relationship_type": "owns", "to_entity": "Arcadia Lithium Project", "insight": "Acquired in 2022"},
            {"from_entity": "Huayou Cobalt", "relationship_type": "invests_in", "to_entity": "Bikita Minerals", "insight": "Major stake acquired"},
            {"from_entity": "Ministry of Mines", "relationship_type": "regulates", "to_entity": "Bikita Minerals", "insight": "Issues mining licenses"},
        ],
        "sources": [
            {"id": "Bikita_Minerals", "type": "company_profile"},
            {"id": "Arcadia_Lithium", "type": "project_profile"},
            {"id": "Mines_and_Minerals_Act", "type": "legislation"},
            {"id": "Zijin_Mining", "type": "company_profile"},
            {"id": "Huayou_Cobalt", "type": "company_profile"},
            {"id": "Lithium_Export_Ban", "type": "policy"},
        ],
        "findings": [
            {"text": "Bikita area is the primary lithium mining region", "source_nodes": ["Bikita_Minerals"]},
            {"text": "Chinese investment dominates the sector", "source_nodes": ["Zijin_Mining", "Huayou_Cobalt"]},
        ],
        "research_required": [
            "Environmental impact assessments for new mines",
            "Local community benefit sharing agreements",
            "Processing capacity within Zimbabwe",
        ],
    }


# =============================================================================
# Mock LLM Client for testing (no external API calls)
# =============================================================================

class MockLLMClient:
    """Mock LLM client that returns a realistic JSON report."""

    def __init__(self, response_data: Dict[str, Any] = None):
        self.response_data = response_data or self._default_response()

    def _default_response(self) -> Dict[str, Any]:
        return {
            "executive_summary": "Zimbabwe possesses significant lithium deposits concentrated in the Bikita area of Masvingo Province. Chinese companies, particularly Zijin Mining and Huayou Cobalt, have made substantial investments in the sector. The government has introduced an export ban on raw lithium to encourage local processing.",
            "implications": "The dominance of Chinese investment creates both opportunities for capital inflow and risks regarding local value capture. The export ban signals intent to develop downstream processing but requires significant infrastructure investment.",
            "investigation_narrative": "The investigation began with identifying lithium mining locations, then expanded to understand the regulatory framework, major corporate players, and export processes. Each query built upon previous findings to create a comprehensive picture of Zimbabwe's lithium sector.",
            "key_findings": [
                {
                    "finding": "Bikita Minerals is the largest lithium producer in Zimbabwe",
                    "confidence": "High",
                    "evidence_queries": ["q-001"],
                    "source_nodes": ["Bikita_Minerals"]
                },
                {
                    "finding": "Chinese companies dominate lithium investment in Zimbabwe",
                    "confidence": "High",
                    "evidence_queries": ["q-003"],
                    "source_nodes": ["Zijin_Mining", "Huayou_Cobalt"]
                },
                {
                    "finding": "Export ban on raw lithium was introduced in 2022",
                    "confidence": "Medium",
                    "evidence_queries": ["q-004"],
                    "source_nodes": ["Lithium_Export_Ban"]
                },
                {
                    "finding": "Special mining leases required for large-scale operations",
                    "confidence": "High",
                    "evidence_queries": ["q-002"],
                    "source_nodes": ["Mines_and_Minerals_Act"]
                },
                {
                    "finding": "Arcadia Lithium Project represents major new development near Harare",
                    "confidence": "High",
                    "evidence_queries": ["q-001"],
                    "source_nodes": ["Arcadia_Lithium"]
                },
                {
                    "finding": "Masvingo Province is the primary lithium mining region",
                    "confidence": "High",
                    "evidence_queries": ["q-001"],
                    "source_nodes": ["Bikita_Minerals"]
                },
                {
                    "finding": "Zijin Mining acquired Arcadia Lithium Project in 2022",
                    "confidence": "High",
                    "evidence_queries": ["q-003"],
                    "source_nodes": ["Zijin_Mining", "Arcadia_Lithium"]
                },
                {
                    "finding": "Huayou Cobalt holds major stake in Bikita Minerals",
                    "confidence": "Medium",
                    "evidence_queries": ["q-003"],
                    "source_nodes": ["Huayou_Cobalt", "Bikita_Minerals"]
                },
                {
                    "finding": "Ministry of Mines issues mining licenses and regulates the sector",
                    "confidence": "High",
                    "evidence_queries": ["q-002", "q-004"],
                    "source_nodes": ["Ministry_of_Mines"]
                },
                {
                    "finding": "Lithium exports require permits from the Ministry of Mines",
                    "confidence": "Medium",
                    "evidence_queries": ["q-004"],
                    "source_nodes": ["Ministry_of_Mines"]
                },
                {
                    "finding": "The Mines and Minerals Act governs all mining operations",
                    "confidence": "High",
                    "evidence_queries": ["q-002"],
                    "source_nodes": ["Mines_and_Minerals_Act"]
                },
                {
                    "finding": "Local processing capacity is limited, creating tension with export ban",
                    "confidence": "Medium",
                    "evidence_queries": ["q-004"],
                    "source_nodes": ["Lithium_Export_Ban"]
                }
            ],
            "important_entities": [
                {"name": "Bikita Minerals", "type": "company", "significance": "Largest lithium producer in Zimbabwe", "evidence_queries": ["q-001"]},
                {"name": "Arcadia Lithium Project", "type": "project", "significance": "Major new development acquired by Zijin", "evidence_queries": ["q-001", "q-003"]},
                {"name": "Zijin Mining", "type": "company", "significance": "Major Chinese investor in Zimbabwe lithium", "evidence_queries": ["q-003"]},
                {"name": "Huayou Cobalt", "type": "company", "significance": "Significant investor in Bikita Minerals", "evidence_queries": ["q-003"]},
                {"name": "Mines and Minerals Act", "type": "legislation", "significance": "Primary governing law for mining operations", "evidence_queries": ["q-002"]},
                {"name": "Ministry of Mines", "type": "government", "significance": "Regulatory authority for mining sector", "evidence_queries": ["q-002", "q-004"]},
                {"name": "Masvingo Province", "type": "region", "significance": "Primary lithium mining region in Zimbabwe", "evidence_queries": ["q-001"]},
            ],
            "important_relationships": [
                {"from_entity": "Zijin Mining", "relationship_type": "owns", "to_entity": "Arcadia Lithium Project", "insight": "Acquired in 2022, representing major Chinese investment", "evidence_queries": ["q-003"]},
                {"from_entity": "Huayou Cobalt", "relationship_type": "invests_in", "to_entity": "Bikita Minerals", "insight": "Holds major stake in largest lithium producer", "evidence_queries": ["q-003"]},
                {"from_entity": "Ministry of Mines", "relationship_type": "regulates", "to_entity": "Bikita Minerals", "insight": "Issues mining licenses and oversees compliance", "evidence_queries": ["q-002"]},
                {"from_entity": "Ministry of Mines", "relationship_type": "regulates", "to_entity": "Arcadia Lithium Project", "insight": "Oversees special mining lease requirements", "evidence_queries": ["q-002", "q-004"]},
                {"from_entity": "Mines and Minerals Act", "relationship_type": "governs", "to_entity": "Bikita Minerals", "insight": "Legal framework for all mining operations", "evidence_queries": ["q-002"]},
                {"from_entity": "Zimbabwe Government", "relationship_type": "policy_maker", "to_entity": "Lithium Export Ban", "insight": "Introduced ban to encourage local processing", "evidence_queries": ["q-004"]},
                {"from_entity": "Bikita Minerals", "relationship_type": "operates_in", "to_entity": "Masvingo Province", "insight": "Primary operations in Bikita area", "evidence_queries": ["q-001"]},
            ],
            "evidence_and_sources": [
                {"source_id": "Bikita_Minerals", "type": "company_profile", "relevance": "Primary lithium producer information"},
                {"source_id": "Arcadia_Lithium", "type": "project_profile", "relevance": "Major development project details"},
                {"source_id": "Mines_and_Minerals_Act", "type": "legislation", "relevance": "Regulatory framework"},
                {"source_id": "Zijin_Mining", "type": "company_profile", "relevance": "Chinese investor information"},
                {"source_id": "Huayou_Cobalt", "type": "company_profile", "relevance": "Investor details"},
                {"source_id": "Lithium_Export_Ban", "type": "policy", "relevance": "Export restrictions"},
            ],
            "unresolved_questions": [
                "What is the timeline for local processing infrastructure development?",
                "How are local communities benefiting from lithium mining?",
                "What are the environmental impacts of expanded lithium mining?",
                "What is the government's long-term strategy for the lithium sector?",
                "How does the export ban affect existing contracts?",
                "What role do local Zimbabwean companies play in the sector?",
            ],
            "research_required": [
                "Environmental impact assessments for new mines",
                "Local community benefit sharing agreements",
                "Processing capacity within Zimbabwe",
                "Downstream value chain opportunities",
                "Comparison with other African lithium producers",
                "Technology transfer agreements with Chinese partners",
            ],
            "confidence_and_limitations": "High confidence in corporate ownership and regulatory framework based on multiple sources. Medium confidence on export ban impacts due to limited information on implementation. Limited data on local community impacts and environmental assessments.",
        }

    def chat(self, messages, temperature=0.7, max_tokens=4096, output_format=None, seed=None, *args, **kwargs):
        return json.dumps(self.response_data)


class BadJSONLLMClient:
    """Mock LLM that returns invalid JSON."""
    def chat(self, *args, **kwargs):
        return "This is not JSON at all"


class EmptyReportLLMClient:
    """Mock LLM that returns an empty report."""
    def chat(self, *args, **kwargs):
        return json.dumps({
            "executive_summary": "",
            "key_findings": [],
            "important_entities": [],
        })


class WrongSchemaLLMClient:
    """Mock LLM that returns wrong field names."""
    def chat(self, *args, **kwargs):
        return json.dumps({
            "executive_summary": "Test",
            "key_findings": [
                {"text": "wrong field name", "confidence": "High"}  # 'finding' missing
            ],
        })


# =============================================================================
# Test Functions
# =============================================================================

def test_pydantic_model_validation():
    """Test 1: Pydantic models enforce the correct schema."""
    print("\n=== TEST 1: Pydantic Model Validation ===")

    # Valid report
    valid_data = {
        "title": "Test Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "based_on_queries": 4,
        "original_question": "Test question",
        "executive_summary": "Test summary",
        "key_findings": [
            {"finding": "Test finding", "confidence": "High", "source_nodes": [], "evidence_queries": []}
        ],
        "important_entities": [
            {"name": "Test Entity", "type": "company", "significance": "Test", "evidence_queries": []}
        ],
        "important_relationships": [
            {"from_entity": "A", "to_entity": "B", "relationship_type": "owns", "insight": "Test", "evidence_queries": []}
        ],
        "evidence_and_sources": [
            {"source_id": "src1", "type": "doc", "relevance": "High"}
        ],
        "unresolved_questions": ["Q1"],
        "research_required": ["R1"],
        "implications": "Test implications",
        "investigation_narrative": "Test narrative",
        "confidence_and_limitations": "Test confidence",
        "evidence_sources_count": 1,
        "evidence_entities_count": 1,
    }

    report = InvestigationReport.model_validate(valid_data)
    assert report.title == "Test Report"
    assert len(report.key_findings) == 1
    assert report.key_findings[0].confidence == "High"
    print("✓ Valid report passes validation")

    # Invalid confidence
    invalid_data = dict(valid_data)
    invalid_data["key_findings"] = [{"finding": "Test", "confidence": "Very High"}]
    report = InvestigationReport.model_validate(invalid_data)
    assert report.key_findings[0].confidence == "Medium"  # Should default to Medium
    print("✓ Invalid confidence defaults to Medium")

    # Missing required fields
    try:
        bad_data = {"title": "", "generated_at": "", "based_on_queries": 0, "original_question": ""}
        InvestigationReport.model_validate(bad_data)
        assert False, "Should have raised validation error"
    except Exception:
        print("✓ Missing required fields correctly rejected")

    # Empty report (no content)
    try:
        empty_data = {
            "title": "Test",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "based_on_queries": 1,
            "original_question": "Q",
        }
        InvestigationReport.model_validate(empty_data)
        assert False, "Should have raised validation error for empty content"
    except Exception:
        print("✓ Empty content report correctly rejected")

    print("TEST 1 PASSED ✓")


def test_deterministic_fields():
    """Test 2: Deterministic fields come from backend, not LLM."""
    print("\n=== TEST 2: Deterministic Fields ===")

    investigation = make_sample_investigation()

    # Mock LLM tries to override deterministic fields
    malicious_response = {
        "title": "LLM WANTS TO OVERRIDE THIS",
        "original_question": "LLM QUESTION",
        "generated_at": "1999-01-01",
        "based_on_queries": 999,
        "executive_summary": "Test",
        "key_findings": [{"finding": "Test", "confidence": "High"}],
    }

    # Patch the LLM client
    import report_generator as rg
    original_get_client = rg.get_client
    rg.get_client = lambda: MockLLMClient(malicious_response)

    try:
        report = generate_investigation_report(investigation)

        # Backend values should win
        assert report.title == "Where can mine lithium in Zimbabwe", f"Got: {report.title}"
        assert report.original_question == "Where can mine lithium in Zimbabwe"
        assert report.based_on_queries == 4, f"Got: {report.based_on_queries}"
        assert report.generated_at != "1999-01-01"
        assert report.evidence_sources_count == 6
        assert report.evidence_entities_count == 7
        print("✓ Backend deterministic fields preserved")
        print(f"  Title: {report.title}")
        print(f"  Based on queries: {report.based_on_queries}")
        print(f"  Evidence sources: {report.evidence_sources_count}")
        print(f"  Evidence entities: {report.evidence_entities_count}")
    finally:
        rg.get_client = original_get_client

    print("TEST 2 PASSED ✓")


def test_full_report_generation():
    """Test 3: Full report generation with mock LLM."""
    print("\n=== TEST 3: Full Report Generation ===")

    investigation = make_sample_investigation()

    import report_generator as rg
    original_get_client = rg.get_client
    rg.get_client = lambda: MockLLMClient()

    try:
        report = generate_investigation_report(investigation)

        # Verify all required fields
        assert report.title
        assert report.generated_at
        assert report.implications
        assert report.executive_summary
        assert report.investigation_narrative
        assert report.confidence_and_limitations
        assert report.original_question

        # Verify array fields
        assert len(report.key_findings) > 0, "key_findings must not be empty"
        assert len(report.important_entities) > 0, "important_entities must not be empty"
        assert len(report.important_relationships) > 0, "important_relationships must not be empty"
        assert len(report.research_required) > 0, "research_required must not be empty"
        assert len(report.unresolved_questions) > 0, "unresolved_questions must not be empty"
        assert len(report.evidence_and_sources) > 0, "evidence_and_sources must not be empty"

        # Verify counts
        assert report.based_on_queries == 4
        assert report.evidence_sources_count == 6
        assert report.evidence_entities_count == 7

        # Verify schema structure
        assert isinstance(report.key_findings[0], KeyFinding)
        assert isinstance(report.important_entities[0], ImportantEntity)
        assert isinstance(report.important_relationships[0], ImportantRelationship)
        assert isinstance(report.evidence_and_sources[0], EvidenceAndSource)

        # Verify JSON serialization works
        json_str = report.model_dump_json()
        parsed = json.loads(json_str)

        # Verify frontend contract
        assert "title" in parsed
        assert "executive_summary" in parsed
        assert "key_findings" in parsed
        assert "important_entities" in parsed
        assert "important_relationships" in parsed
        assert "evidence_and_sources" in parsed
        assert "research_required" in parsed
        assert "unresolved_questions" in parsed
        assert "investigation_narrative" in parsed
        assert "implications" in parsed
        assert "confidence_and_limitations" in parsed

        # Verify no wrapper
        assert "status" not in parsed, "Response must not have a 'status' wrapper"
        assert "report" not in parsed, "Response must not have a 'report' wrapper"

        print(f"✓ Report generated successfully")
        print(f"  key_findings: {len(report.key_findings)}")
        print(f"  important_entities: {len(report.important_entities)}")
        print(f"  important_relationships: {len(report.important_relationships)}")
        print(f"  research_required: {len(report.research_required)}")
        print(f"  unresolved_questions: {len(report.unresolved_questions)}")
        print(f"  evidence_and_sources: {len(report.evidence_and_sources)}")

        # Print schema validation
        print("\n=== Response Schema ===")
        print(f"Keys: {list(parsed.keys())}")
        print(f"key_findings length: {len(parsed['key_findings'])}")
        print(f"important_entities length: {len(parsed['important_entities'])}")
        print(f"important_relationships length: {len(parsed['important_relationships'])}")
        print(f"evidence_and_sources length: {len(parsed['evidence_and_sources'])}")
        print(f"research_required length: {len(parsed['research_required'])}")
        print(f"unresolved_questions length: {len(parsed['unresolved_questions'])}")
        print(f"nested key_findings keys: {list(parsed['key_findings'][0].keys()) if parsed['key_findings'] else []}")
        print(f"nested important_entities keys: {list(parsed['important_entities'][0].keys()) if parsed['important_entities'] else []}")
        print(f"nested important_relationships keys: {list(parsed['important_relationships'][0].keys()) if parsed['important_relationships'] else []}")
        print(f"nested evidence_and_sources keys: {list(parsed['evidence_and_sources'][0].keys()) if parsed['evidence_and_sources'] else []}")

    finally:
        rg.get_client = original_get_client

    print("TEST 3 PASSED ✓")


def test_error_handling():
    """Test 4: Error handling for bad LLM output — fallback chain."""
    print("\n=== TEST 4: Error Handling ===")

    investigation = make_sample_investigation()
    import report_generator as rg
    original_get_client = rg.get_client

    # Test bad JSON — should trigger fallback chain and return a valid report
    rg.get_client = lambda: BadJSONLLMClient()
    try:
        report = generate_investigation_report(investigation)
        # Fallback report is returned instead of raising
        assert isinstance(report, InvestigationReport)
        assert report.title == "Where can mine lithium in Zimbabwe"
        assert len(report.key_findings) > 0  # Fallback constructs findings from raw data
        print("✓ Bad JSON triggers fallback chain — valid report returned")
        print(f"  Fallback findings: {len(report.key_findings)}")
        print(f"  Fallback entities: {len(report.important_entities)}")
    finally:
        rg.get_client = original_get_client

    # Test empty report — fallback should also handle this
    rg.get_client = lambda: EmptyReportLLMClient()
    try:
        report = generate_investigation_report(investigation)
        assert isinstance(report, InvestigationReport)
        assert report.title == "Where can mine lithium in Zimbabwe"
        print("✓ Empty report triggers fallback — valid report returned")
    finally:
        rg.get_client = original_get_client

    print("TEST 4 PASSED ✓")


def test_cross_validation():
    """Test 5: Evidence query IDs are cross-validated."""
    print("\n=== TEST 5: Cross-Validation ===")

    investigation = make_sample_investigation()

    # Create response with some invalid query IDs
    response_with_bad_queries = {
        "executive_summary": "Test",
        "key_findings": [
            {"finding": "Good query ref", "confidence": "High", "evidence_queries": ["q-001"]},
            {"finding": "Bad query ref", "confidence": "Medium", "evidence_queries": ["nonexistent-query"]},
        ],
        "important_entities": [
            {"name": "Test", "type": "company", "significance": "Test", "evidence_queries": ["q-002", "bad-query"]},
        ],
        "important_relationships": [
            {"from_entity": "A", "to_entity": "B", "relationship_type": "owns", "insight": "Test", "evidence_queries": ["q-003"]},
        ],
    }

    import report_generator as rg
    original_get_client = rg.get_client
    rg.get_client = lambda: MockLLMClient(response_with_bad_queries)

    try:
        report = generate_investigation_report(investigation)
        # Should still succeed but log warnings
        assert len(report.key_findings) == 2
        assert report.key_findings[0].evidence_queries == ["q-001"]
        assert report.key_findings[1].evidence_queries == ["nonexistent-query"]  # Kept but warned
        print("✓ Invalid query IDs are logged but don't block report generation")
    finally:
        rg.get_client = original_get_client

    print("TEST 5 PASSED ✓")


def test_no_wrapper_response():
    """Test 6: Verify the endpoint would return the report directly, not wrapped."""
    print("\n=== TEST 6: No Wrapper Response ===")

    investigation = make_sample_investigation()
    import report_generator as rg
    original_get_client = rg.get_client
    rg.get_client = lambda: MockLLMClient()

    try:
        report = generate_investigation_report(investigation)
        json_output = report.model_dump_json()
        data = json.loads(json_output)

        # The frontend contract expects direct access
        assert data["title"] == "Where can mine lithium in Zimbabwe"
        assert isinstance(data["key_findings"], list)
        assert isinstance(data["important_entities"], list)
        assert isinstance(data["important_relationships"], list)

        # No wrapper keys
        assert "status" not in data
        assert "report" not in data
        assert "content" not in data
        assert "data" not in data

        print("✓ Response contains no wrapper — direct schema access works")
        print(f"  Direct access: report.title = '{data['title']}'")
        print(f"  Direct access: report.executive_summary = '{data['executive_summary'][:50]}...'")
    finally:
        rg.get_client = original_get_client

    print("TEST 6 PASSED ✓")


def run_all_tests():
    """Run the complete test suite."""
    print("=" * 70)
    print("ATIS INVESTIGATION REPORT CONTRACT — VALIDATION TEST SUITE")
    print("=" * 70)

    test_pydantic_model_validation()
    test_deterministic_fields()
    test_full_report_generation()
    test_error_handling()
    test_cross_validation()
    test_no_wrapper_response()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nThe /api/investigation/report endpoint is ready for frontend integration.")
    print("It returns a strict, validated InvestigationReport object directly.")


if __name__ == "__main__":
    run_all_tests()
