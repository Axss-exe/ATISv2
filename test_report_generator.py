#!/usr/bin/env python3
"""Tests for report_generator.py — payload-based investigation report generation."""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch

# Ensure imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report_generator import (
    generate_investigation_report,
    _normalize_investigation,
    _build_synthesis_context,
    _parse_and_validate_report,
)


class MockLLMClient:
    """Mock LLM client that returns a fixed response."""
    def __init__(self, response: str):
        self.response = response
        self.last_messages: list = []
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None
        self.last_seed: int | None = None

    def chat(self, messages, temperature=None, max_tokens=None, seed=None):
        self.last_messages = messages
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        self.last_seed = seed
        return self.response


class TestReportGenerator(unittest.TestCase):
    """Comprehensive tests for the report generator module."""

    def setUp(self):
        self.sample_investigation = {
            "investigation_id": "inv-123",
            "title": "Test Investigation",
            "original_question": "Who controls the cobalt trade in DRC?",
            "perspective": {"country": "Zimbabwe", "country_code": "ZW"},
            "queries": [
                {
                    "sequence": 1,
                    "question": "Who controls the cobalt trade in DRC?",
                    "answer": "Major Chinese mining companies control 70% of DRC cobalt output.",
                    "findings": [
                        {"text": "Chinese firms own majority stakes in key mines.", "source_nodes": ["mine_a"]}
                    ],
                },
                {
                    "sequence": 2,
                    "question": "What are the implications for Zimbabwe?",
                    "answer": "Zimbabwe could benefit from regional processing partnerships.",
                    "findings": [
                        {"text": "Regional value chains are emerging.", "source_nodes": ["report_b"]}
                    ],
                },
            ],
            "entities": [
                {"name": "Chinese Mining Consortium", "type": "business", "country": "China", "summary": "Controls major DRC cobalt mines."},
                {"name": "DRC Government", "type": "government", "country": "DRC", "summary": "Regulates mining licenses."},
            ],
            "relationships": [
                {"from_entity": "Chinese Mining Consortium", "relationship_type": "controls", "to_entity": "DRC cobalt mines", "insight": "Majority ownership structure."},
            ],
            "sources": [
                {"node_id": "mine_a", "node_type": "article"},
                {"node_id": "report_b", "node_type": "report"},
            ],
            "findings": [
                {"text": "Chinese firms own majority stakes in key mines.", "source_nodes": ["mine_a"]},
                {"text": "Regional value chains are emerging.", "source_nodes": ["report_b"]},
            ],
        }

        self.valid_report_json = json.dumps({
            "title": "Cobalt Trade Control Investigation",
            "executive_summary": "Chinese mining companies dominate DRC cobalt production.",
            "original_question": "Who controls the cobalt trade in DRC?",
            "investigation_narrative": "The investigation began with ownership questions...",
            "key_findings": [
                {"finding": "Chinese firms control 70% of output", "confidence": "High", "evidence_queries": ["Q1"], "source_nodes": ["mine_a"]}
            ],
            "important_entities": [
                {"name": "Chinese Mining Consortium", "type": "business", "significance": "Major cobalt controller", "evidence_queries": ["Q1"]}
            ],
            "important_relationships": [
                {"from_entity": "Chinese Mining Consortium", "relationship_type": "controls", "to_entity": "DRC cobalt mines", "insight": "Majority ownership", "evidence_queries": ["Q1"]}
            ],
            "evidence_and_sources": [
                {"source_id": "mine_a", "type": "article", "relevance": "Primary evidence"}
            ],
            "unresolved_questions": ["What is the Zimbabwean policy response?"],
            "research_required": ["Analyze Zimbabwe-DRC trade agreements"],
            "implications": "Zimbabwe should pursue regional processing partnerships.",
            "confidence_and_limitations": "High confidence on ownership data; limited on policy implications.",
        })

    # -------------------------------------------------------------------------
    # 1. Valid Investigation payload reaches the report generator
    # -------------------------------------------------------------------------
    @patch("report_generator.get_client")
    def test_valid_investigation_payload_reaches_generator(self, mock_get_client):
        mock_get_client.return_value = MockLLMClient(self.valid_report_json)
        report = generate_investigation_report(self.sample_investigation)
        self.assertIsInstance(report, dict)
        self.assertIn("title", report)
        self.assertEqual(report["title"], "Cobalt Trade Control Investigation")
        self.assertIn("key_findings", report)
        self.assertEqual(len(report["key_findings"]), 1)

    # -------------------------------------------------------------------------
    # 2. Existing LLM client is invoked
    # -------------------------------------------------------------------------
    @patch("report_generator.get_client")
    def test_existing_llm_client_is_invoked(self, mock_get_client):
        mock_client = MockLLMClient(self.valid_report_json)
        mock_get_client.return_value = mock_client
        generate_investigation_report(self.sample_investigation)
        mock_get_client.assert_called_once()
        # Verify the client received the expected arguments
        self.assertEqual(mock_client.last_temperature, 0.0)
        self.assertEqual(mock_client.last_max_tokens, 8192)
        self.assertEqual(mock_client.last_seed, 42)
        self.assertEqual(len(mock_client.last_messages), 2)
        self.assertEqual(mock_client.last_messages[0]["role"], "system")
        self.assertEqual(mock_client.last_messages[1]["role"], "user")

    # -------------------------------------------------------------------------
    # 3. Report JSON is parsed correctly
    # -------------------------------------------------------------------------
    @patch("report_generator.get_client")
    def test_report_json_is_parsed(self, mock_get_client):
        mock_get_client.return_value = MockLLMClient(self.valid_report_json)
        report = generate_investigation_report(self.sample_investigation)
        self.assertIsInstance(report, dict)
        self.assertIn("executive_summary", report)
        self.assertIn("investigation_narrative", report)
        self.assertIn("important_entities", report)
        self.assertEqual(len(report["important_entities"]), 1)
        self.assertIn("generated_at", report)
        self.assertIn("based_on_queries", report)

    # -------------------------------------------------------------------------
    # 4. Invalid LLM JSON fails safely
    # -------------------------------------------------------------------------
    @patch("report_generator.get_client")
    def test_invalid_llm_json_fails_safely(self, mock_get_client):
        mock_get_client.return_value = MockLLMClient("this is not json {{{")
        with self.assertRaises(RuntimeError) as ctx:
            generate_investigation_report(self.sample_investigation)
        self.assertIn("invalid json", str(ctx.exception).lower())

    @patch("report_generator.get_client")
    def test_invalid_llm_json_with_markdown_fences_fails_safely(self, mock_get_client):
        mock_get_client.return_value = MockLLMClient("```json\nnot json at all\n```")
        with self.assertRaises(RuntimeError) as ctx:
            generate_investigation_report(self.sample_investigation)
        self.assertIn("invalid json", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 5. Missing investigation fields are handled
    # -------------------------------------------------------------------------
    def test_missing_queries_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            generate_investigation_report({"title": "Empty"})
        self.assertIn("at least one query", str(ctx.exception).lower())

    def test_empty_queries_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            generate_investigation_report({"title": "Empty", "queries": []})
        self.assertIn("at least one query", str(ctx.exception).lower())

    def test_missing_title_derived_from_question(self):
        normalized = _normalize_investigation({
            "original_question": "What is happening with lithium in Zimbabwe?",
            "queries": [{"question": "What is happening with lithium in Zimbabwe?"}],
        })
        self.assertEqual(normalized["title"], "What is happening with lithium in Zimbabwe?")

    # -------------------------------------------------------------------------
    # 6. LLM provider failure returns a useful error
    # -------------------------------------------------------------------------
    @patch("report_generator.get_client")
    def test_llm_provider_failure_returns_useful_error(self, mock_get_client):
        from llm_client import LLMRequestError
        mock_get_client.side_effect = LLMRequestError("provider down")
        with self.assertRaises(RuntimeError) as ctx:
            generate_investigation_report(self.sample_investigation)
        self.assertIn("llm provider error", str(ctx.exception).lower())

    @patch("report_generator.get_client")
    def test_llm_timeout_returns_useful_error(self, mock_get_client):
        mock_get_client.side_effect = TimeoutError("connection timed out")
        with self.assertRaises(RuntimeError) as ctx:
            generate_investigation_report(self.sample_investigation)
        self.assertIn("report generation failed", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 7. API key is never returned or exposed
    # -------------------------------------------------------------------------
    def test_api_key_never_in_source(self):
        import report_generator
        with open(report_generator.__file__, "r") as fh:
            source = fh.read()
        self.assertNotIn("api_key", source.lower())
        self.assertNotIn("apikey", source.lower())
        self.assertNotIn("sk-", source.lower())
        self.assertNotIn("mistral", source.lower())

    def test_api_key_never_in_report_output(self):
        with patch("report_generator.get_client") as mock_get_client:
            mock_get_client.return_value = MockLLMClient(self.valid_report_json)
            report = generate_investigation_report(self.sample_investigation)
            report_str = json.dumps(report)
            self.assertNotIn("api_key", report_str.lower())
            self.assertNotIn("apikey", report_str.lower())
            self.assertNotIn("sk-", report_str.lower())

    # -------------------------------------------------------------------------
    # 8. Existing ATIS query functionality is unaffected
    # -------------------------------------------------------------------------
    def test_report_generator_does_not_import_atis_query(self):
        import report_generator
        with open(report_generator.__file__, "r") as fh:
            source = fh.read()
        self.assertNotIn("ATIS_Query", source)
        self.assertNotIn("run_query_pipeline", source)
        self.assertNotIn("investigation_manager", source)
        self.assertNotIn("create_investigation", source)

    # -------------------------------------------------------------------------
    # Normalization tests
    # -------------------------------------------------------------------------
    def test_normalization_preserves_query_sequence(self):
        normalized = _normalize_investigation(self.sample_investigation)
        self.assertEqual(len(normalized["queries"]), 2)
        self.assertEqual(normalized["queries"][0]["sequence"], 1)
        self.assertEqual(normalized["queries"][1]["sequence"], 2)

    def test_normalization_extracts_entities_from_queries_when_sparse(self):
        sparse = {
            "original_question": "Test",
            "queries": [
                {
                    "question": "Q1",
                    "result": {
                        "key_entities": [
                            {"entity_name": "Entity A", "entity_type": "person", "country": "ZW"}
                        ]
                    }
                }
            ]
        }
        normalized = _normalize_investigation(sparse)
        self.assertEqual(len(normalized["entities"]), 1)
        self.assertEqual(normalized["entities"][0]["name"], "Entity A")

    def test_normalization_handles_various_field_names(self):
        """Test that various frontend field naming conventions are handled."""
        inv = {
            "id": "inv-456",
            "investigation_title": "Alternative Title",
            "root_question": "Root Q",
            "perspective_country": "South Africa",
            "perspective_country_code": "ZA",
            "query_sequence": [
                {"query": "Q1 text", "answer": "A1", "sequence": 1}
            ],
            "researchRequired": True,
        }
        normalized = _normalize_investigation(inv)
        self.assertEqual(normalized["investigation_id"], "inv-456")
        self.assertEqual(normalized["title"], "Alternative Title")
        self.assertEqual(normalized["original_question"], "Root Q")
        self.assertEqual(normalized["perspective_country"], "South Africa")
        self.assertEqual(normalized["perspective_country_code"], "ZA")
        self.assertEqual(len(normalized["queries"]), 1)
        self.assertEqual(normalized["queries"][0]["question"], "Q1 text")
        self.assertEqual(normalized["research_required"], ["Further research recommended"])

    def test_normalization_handles_aggregated_context(self):
        inv = {
            "original_question": "Test",
            "queries": [{"question": "Q1"}],
            "aggregated_context": {
                "entities": [{"name": "Agg Entity", "type": "org"}],
                "relationships": [{"from_entity": "A", "relationship_type": "owns", "to_entity": "B"}],
                "sources": [{"node_id": "src1", "node_type": "doc"}],
                "findings": [{"text": "Finding 1"}],
            }
        }
        normalized = _normalize_investigation(inv)
        self.assertEqual(len(normalized["entities"]), 1)
        self.assertEqual(len(normalized["relationships"]), 1)
        self.assertEqual(len(normalized["sources"]), 1)
        self.assertEqual(len(normalized["findings"]), 1)

    # -------------------------------------------------------------------------
    # Prompt construction tests
    # -------------------------------------------------------------------------
    def test_synthesis_context_contains_metadata(self):
        normalized = _normalize_investigation(self.sample_investigation)
        context = _build_synthesis_context(normalized)
        self.assertIn("ATIS INVESTIGATION KNOWLEDGE REPORT", context)
        self.assertIn("Test Investigation", context)
        self.assertIn("Who controls the cobalt trade in DRC?", context)
        self.assertIn("Zimbabwe", context)
        self.assertIn("Q1:", context)
        self.assertIn("Q2:", context)

    def test_synthesis_context_contains_entities_and_relationships(self):
        normalized = _normalize_investigation(self.sample_investigation)
        context = _build_synthesis_context(normalized)
        self.assertIn("Chinese Mining Consortium", context)
        self.assertIn("controls", context)
        self.assertIn("DRC cobalt mines", context)

    # -------------------------------------------------------------------------
    # Parsing tests
    # -------------------------------------------------------------------------
    def test_parse_strips_markdown_fences(self):
        raw = "```json\n" + self.valid_report_json + "\n```"
        normalized = _normalize_investigation(self.sample_investigation)
        report = _parse_and_validate_report(raw, normalized)
        self.assertEqual(report["title"], "Cobalt Trade Control Investigation")

    def test_parse_falls_back_to_empty_fields(self):
        raw = json.dumps({"title": "Minimal"})
        normalized = _normalize_investigation(self.sample_investigation)
        report = _parse_and_validate_report(raw, normalized)
        self.assertEqual(report["title"], "Minimal")
        self.assertEqual(report["key_findings"], [])
        self.assertEqual(report["unresolved_questions"], [])

    def test_parse_sets_generated_at(self):
        raw = json.dumps({"title": "Test"})
        normalized = _normalize_investigation(self.sample_investigation)
        report = _parse_and_validate_report(raw, normalized)
        self.assertIn("T", report["generated_at"])  # ISO format contains T

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------
    def test_single_query_investigation(self):
        single = {
            "original_question": "Single Q",
            "queries": [{"question": "Single Q", "answer": "Single A"}],
        }
        with patch("report_generator.get_client") as mock_get_client:
            mock_get_client.return_value = MockLLMClient(self.valid_report_json)
            report = generate_investigation_report(single)
            self.assertEqual(report["based_on_queries"], 1)

    def test_boolean_research_required(self):
        inv = {
            "original_question": "Test",
            "queries": [{"question": "Q1"}],
            "research_required": True,
        }
        normalized = _normalize_investigation(inv)
        self.assertEqual(normalized["research_required"], ["Further research recommended"])

    def test_empty_research_required(self):
        inv = {
            "original_question": "Test",
            "queries": [{"question": "Q1"}],
            "research_required": False,
        }
        normalized = _normalize_investigation(inv)
        self.assertEqual(normalized["research_required"], [])


if __name__ == "__main__":
    unittest.main()
