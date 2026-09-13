#!/usr/bin/env python3
"""
Regression test for investigation report JSON parsing failure.

This test validates the fix for the production issue where:
- LLM returned content_len=17809 with max_tokens=4096
- JSON parsing failed with: "Expecting ',' delimiter: line 65 column 200 (char 17622)"
- The error was: Invalid report request: Expecting ',' delimiter

The fix uses safe_json_loads from ATIS_Query which has layered JSON recovery.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from investigation_manager import (
    create_investigation,
    generate_investigation_report,
    _parse_report_response,
)


class TestInvestigationReportJSONParsing(unittest.TestCase):
    """Test JSON parsing robustness in investigation report generation."""

    def setUp(self):
        self.sample_investigation = {
            "investigation_id": "test-8c8894bc-86b5-4e72-bfea-7091d3ccf9f7",
            "title": "Test Investigation",
            "status": "active",
            "perspective": {"country": "Zimbabwe", "country_code": "ZW"},
            "root_question": "Where can mine lithium in Zimbabwe",
            "queries": [
                {
                    "query_id": "q-001",
                    "sequence": 1,
                    "question": "Where can mine lithium in Zimbabwe",
                    "parent_query_id": None,
                    "result": {
                        "executive_summary": "Lithium mining in Zimbabwe is concentrated...",
                        "key_entities": [],
                        "structured_intelligence": [],
                        "source_nodes": [],
                        "key_entities": [],
                    },
                    "created_at": "2024-01-01T00:00:00Z",
                }
            ],
            "aggregated_context": {
                "entities": [],
                "relationships": [],
                "findings": [],
                "risks": [],
                "opportunities": [],
                "sources": [],
            },
            "report": None,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        self.sample_aggregated = {
            "entities": [],
            "relationships": [],
            "findings": [],
            "risks": [],
            "opportunities": [],
            "sources": [],
        }

    def test_parse_valid_json_report(self):
        """Test that valid JSON is parsed correctly."""
        valid_json = json.dumps({
            "executive_summary": "Test summary",
            "investigation_objective": "Test objective",
            "key_findings": [{"finding": "Test finding", "source_nodes": []}],
            "key_entities": [],
            "key_relationships": [],
            "timeline": "Test timeline",
            "evidence_summary": "Test evidence",
            "risks": [],
            "opportunities": [],
            "contradictions": [],
            "knowledge_gaps": [],
            "unanswered_questions": [],
            "conclusion": "Test conclusion",
        })
        result = _parse_report_response(
            valid_json, self.sample_investigation, self.sample_aggregated
        )
        self.assertEqual(result["executive_summary"], "Test summary")
        self.assertEqual(result["investigation_objective"], "Test objective")
        self.assertEqual(len(result["key_findings"]), 1)
        self.assertIn("generated_at", result)
        self.assertEqual(result["based_on_queries"], 1)

    def test_parse_json_with_markdown_fences(self):
        """Test that JSON wrapped in markdown fences is parsed correctly."""
        json_with_fences = "```json\n" + json.dumps({
            "executive_summary": "Test summary",
            "investigation_objective": "Test objective",
        }) + "\n```"
        result = _parse_report_response(
            json_with_fences, self.sample_investigation, self.sample_aggregated
        )
        self.assertEqual(result["executive_summary"], "Test summary")

    def test_parse_json_with_trailing_comma(self):
        """Test that JSON with trailing comma is handled via safe_json_loads."""
        json_with_trailing_comma = json.dumps({
            "executive_summary": "Test",
            "key_findings": [{"finding": "Test"}],
        }) + ","
        result = _parse_report_response(
            json_with_trailing_comma, self.sample_investigation, self.sample_aggregated
        )
        self.assertEqual(result["executive_summary"], "Test")

    def test_parse_json_with_embedded_text(self):
        """Test parsing when LLM returns text before/after JSON."""
        embedded_json = "Some text before\n" + json.dumps({
            "executive_summary": "Test",
            "investigation_objective": "Test obj",
        }) + "\nSome text after"
        result = _parse_report_response(
            embedded_json, self.sample_investigation, self.sample_aggregated
        )
        self.assertEqual(result["executive_summary"], "Test")

    def test_parse_json_with_unterminated_string(self):
        """Test parsing when JSON has unterminated string - should use safe_json_loads recovery."""
        # This simulates the kind of malformed JSON that caused the production error
        malformed = json.dumps({
            "executive_summary": "Test summary",
            "investigation_objective": "Test objective",
            "key_findings": [
                {"finding": "Finding 1", "source_nodes": []},
                {"finding": "Finding 2", "source_nodes": []},
            ],
        }) + '"'  # Extra quote at the end
        result = _parse_report_response(
            malformed, self.sample_investigation, self.sample_aggregated
        )
        # Should still extract valid data
        self.assertEqual(result["executive_summary"], "Test summary")

    def test_parse_json_with_unbalanced_braces(self):
        """Test parsing when JSON has unbalanced braces."""
        unbalanced = json.dumps({
            "executive_summary": "Test",
            "key_findings": [
                {"finding": "Test", "source_nodes": []},
            ],
        }) + "}"  # Extra closing brace
        result = _parse_report_response(
            unbalanced, self.sample_investigation, self.sample_aggregated
        )
        # Should extract the valid JSON portion
        self.assertEqual(result["executive_summary"], "Test")

    def test_parse_json_with_newlines_in_strings(self):
        """Test parsing JSON with newlines in string values."""
        json_with_newlines = json.dumps({
            "executive_summary": "Line 1\nLine 2\nLine 3",
            "investigation_objective": "Objective",
        })
        result = _parse_report_response(
            json_with_newlines, self.sample_investigation, self.sample_aggregated
        )
        self.assertIn("Line 1", result["executive_summary"])
        self.assertIn("Line 2", result["executive_summary"])

    def test_parse_empty_response(self):
        """Test parsing empty response returns empty dict fields."""
        result = _parse_report_response(
            "", self.sample_investigation, self.sample_aggregated
        )
        self.assertEqual(result["executive_summary"], "")
        self.assertEqual(result["key_findings"], [])
        self.assertIn("generated_at", result)

    def test_parse_non_json_response(self):
        """Test parsing non-JSON text returns empty dict fields."""
        result = _parse_report_response(
            "This is not JSON at all", self.sample_investigation, self.sample_aggregated
        )
        self.assertEqual(result["executive_summary"], "")
        self.assertEqual(result["key_findings"], [])

    def test_generate_investigation_report_with_mocked_llm(self):
        """Test full report generation with mocked LLM returning valid JSON."""
        valid_report_json = json.dumps({
            "executive_summary": "Comprehensive report",
            "investigation_objective": "To find lithium mining locations",
            "key_findings": [
                {"finding": "Bikita has lithium", "confidence": "High", "source_nodes": ["Bikita"], "evidence_queries": ["Q1"]},
            ],
            "key_entities": [
                {"name": "Bikita Minerals", "type": "company", "significance": "Major producer"},
            ],
            "key_relationships": [],
            "timeline": "2024",
            "evidence_summary": "Strong evidence",
            "risks": [],
            "opportunities": [],
            "contradictions": [],
            "knowledge_gaps": [],
            "unanswered_questions": [],
            "conclusion": "Lithium is in Bikita",
        })

        with patch("investigation_manager.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = valid_report_json
            mock_get_client.return_value = mock_client

            with patch("investigation_manager.get_investigation") as mock_get_inv:
                with patch("investigation_manager._persist_investigation"):
                    mock_get_inv.return_value = self.sample_investigation
                    report = generate_investigation_report("test-123")

                    self.assertIn("executive_summary", report)
                    self.assertEqual(report["executive_summary"], "Comprehensive report")
                    self.assertEqual(len(report["key_findings"]), 1)

    def test_generate_investigation_report_with_malformed_json(self):
        """Test full report generation with mocked LLM returning malformed JSON."""
        # Simulate the production error: malformed JSON at position 17622
        malformed_json = json.dumps({
            "executive_summary": "A" * 1000,
            "investigation_objective": "B" * 1000,
            "key_findings": [
                {"finding": "C" * 100, "source_nodes": []} for _ in range(50)
            ],
        }) + '"'  # Extra quote causing parse error

        with patch("investigation_manager.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.return_value = malformed_json
            mock_get_client.return_value = mock_client

            with patch("investigation_manager.get_investigation") as mock_get_inv:
                with patch("investigation_manager._persist_investigation"):
                    mock_get_inv.return_value = self.sample_investigation
                    # Should not raise an exception
                    report = generate_investigation_report("test-123")

                    # Should return a report with some fields populated
                    self.assertIn("executive_summary", report)
                    self.assertIn("generated_at", report)


if __name__ == "__main__":
    unittest.main()
