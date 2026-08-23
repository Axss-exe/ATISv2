#!/usr/bin/env python3
"""
test_perspective.py

Integration tests for the ATIS Perspective-First architecture.
Validates that:
  1. Perspective-side nodes are retrieved from the vault
  2. Cross-border bridges are discovered via graph traversal
  3. Opportunities are validated deterministically against vault evidence
  4. RESEARCH_REQUIRED is assigned when perspective-side evidence is missing
  5. Opportunity geography is NOT defaulted to perspective country
  6. Cross-border status is NOT automatic — requires bridge evidence
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure we can import the modules under test
sys.path.insert(0, str(Path(__file__).parent))

from atis_context import (
    PerspectiveContext,
    validate_opportunity,
    CROSS_BORDER_PATHWAYS,
    DOMESTIC_PATHWAYS,
)


# =============================================================================
# Mock Vault Setup Helpers
# =============================================================================
def create_mock_vault(tmpdir: Path) -> Path:
    """Create a minimal mock vault with nodes from multiple countries."""
    vault = tmpdir / "vault"
    vault.mkdir()

    # Zimbabwe nodes (perspective country)
    zim_dir = vault / "Zimbabwe"
    zim_dir.mkdir()
    (zim_dir / "ZESA Holdings.md").write_text(
        "---\n"
        "country: Zimbabwe\n"
        "node_type: infrastructure_node\n"
        "sector: Energy\n"
        "---\n"
        "# ZESA Holdings\n"
        "National power utility of Zimbabwe. Operates Hwange Power Station.\n"
        "- [[Hwange Power Station]]\n"
        "- [[ZERA]]\n"
        "- [[Zambia REAP]]\n"
    )
    (zim_dir / "Hwange Power Station.md").write_text(
        "---\n"
        "country: Zimbabwe\n"
        "node_type: infrastructure_node\n"
        "sector: Energy\n"
        "---\n"
        "# Hwange Power Station\n"
        "Coal-fired power station in Hwange, Zimbabwe.\n"
        "- [[ZESA Holdings]]\n"
    )
    (zim_dir / "ZERA.md").write_text(
        "---\n"
        "country: Zimbabwe\n"
        "node_type: government_agency\n"
        "---\n"
        "# Zimbabwe Energy Regulatory Authority\n"
        "Regulates the energy sector in Zimbabwe.\n"
    )
    (zim_dir / "Bikita Minerals.md").write_text(
        "---\n"
        "country: Zimbabwe\n"
        "node_type: mining_refinery\n"
        "sector: Mining\n"
        "---\n"
        "# Bikita Minerals\n"
        "Lithium mining company in Zimbabwe.\n"
    )

    # Zambia nodes (source country)
    zam_dir = vault / "Zambia"
    zam_dir.mkdir()
    (zam_dir / "Zambia REAP.md").write_text(
        "---\n"
        "country: Zambia\n"
        "node_type: policy_framework\n"
        "sector: Energy\n"
        "---\n"
        "# Zambia Rural Electricity Access Project\n"
        "Rural electrification initiative in Zambia.\n"
        "- [[ZESCO]]\n"
        "- [[ZESA Holdings]]\n"
    )
    (zam_dir / "ZESCO.md").write_text(
        "---\n"
        "country: Zambia\n"
        "node_type: infrastructure_node\n"
        "sector: Energy\n"
        "---\n"
        "# ZESCO\n"
        "Zambia Electricity Supply Corporation.\n"
    )

    # South Africa node (third country)
    sa_dir = vault / "South Africa"
    sa_dir.mkdir()
    (sa_dir / "Eskom.md").write_text(
        "---\n"
        "country: South Africa\n"
        "node_type: infrastructure_node\n"
        "sector: Energy\n"
        "---\n"
        "# Eskom\n"
        "South African electricity public utility.\n"
    )

    return vault


# =============================================================================
# Unit Tests
# =============================================================================
class TestPerspectiveContext(unittest.TestCase):
    def test_default_perspective(self):
        ctx = PerspectiveContext()
        self.assertEqual(ctx.country, "Zimbabwe")
        self.assertEqual(ctx.country_code, "ZW")

    def test_from_values(self):
        ctx = PerspectiveContext.from_values("Zambia", "ZM")
        self.assertEqual(ctx.country, "Zambia")
        self.assertEqual(ctx.country_code, "ZM")

    def test_from_values_infer_code(self):
        ctx = PerspectiveContext.from_values("Zimbabwe")
        self.assertEqual(ctx.country_code, "ZW")

    def test_from_payload(self):
        payload = {"perspective_country": "South Africa", "perspective_country_code": "ZA"}
        ctx = PerspectiveContext.from_payload(payload)
        self.assertEqual(ctx.country, "South Africa")
        self.assertEqual(ctx.country_code, "ZA")

    def test_from_payload_nested(self):
        payload = {"perspective": {"country": "Botswana", "country_code": "BW"}}
        ctx = PerspectiveContext.from_payload(payload)
        self.assertEqual(ctx.country, "Botswana")
        self.assertEqual(ctx.country_code, "BW")


class TestValidateOpportunity(unittest.TestCase):
    def setUp(self):
        self.perspective = PerspectiveContext("Zimbabwe", "ZW")
        self.perspective_node_ids = {"ZESA Holdings", "Hwange Power Station", "ZERA", "Bikita Minerals"}
        self.cross_border_bridges = [
            {
                "from_node": "ZESA Holdings",
                "from_country": "Zimbabwe",
                "to_node": "Zambia REAP",
                "to_country": "Zambia",
                "relationship_type": "outbound_link",
            }
        ]

    def test_valid_cross_border_opportunity(self):
        """A properly evidenced cross-border opportunity should be VALID."""
        opp = {
            "opportunity_id": "OPP-001",
            "title": "Cross-border power supply",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "National grid operations",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=self.cross_border_bridges,
        )
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["cross_border"])
        self.assertTrue(result["perspective_actor_evidence"])
        self.assertTrue(result["perspective_capability_evidence"])
        self.assertTrue(result["pathway_evidence"])
        self.assertEqual(result["opportunity_country"], "Zambia")

    def test_missing_perspective_actor(self):
        """Missing perspective_actor should result in RESEARCH_REQUIRED."""
        opp = {
            "opportunity_id": "OPP-002",
            "title": "Local supply",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "",
            "perspective_capability": "Some capability",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=self.cross_border_bridges,
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")
        self.assertIn("Missing perspective_actor", result["validation_note"])

    def test_un evidenced_perspective_actor(self):
        """An invented perspective actor should result in RESEARCH_REQUIRED."""
        opp = {
            "opportunity_id": "OPP-003",
            "title": "Fake opportunity",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "Fake Zimbabwe Company",
            "perspective_capability": "Fake capability",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=self.cross_border_bridges,
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")
        self.assertIn("Perspective actor 'Fake Zimbabwe Company' is not evidenced", result["validation_note"])
        self.assertFalse(result["perspective_actor_evidence"])

    def test_cross_border_without_bridges(self):
        """Cross-border pathway without bridge evidence should be RESEARCH_REQUIRED."""
        opp = {
            "opportunity_id": "OPP-004",
            "title": "Unbridged cross-border",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=[],  # No bridges!
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")
        self.assertIn("Cross-border pathway lacks vault evidence", result["validation_note"])
        self.assertFalse(result["cross_border"])

    def test_domestic_pathway_for_foreign_event(self):
        """Domestic pathway for a foreign event without cross-border bridge should fail."""
        opp = {
            "opportunity_id": "OPP-005",
            "title": "Domestic for foreign event",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "domestic procurement",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=self.cross_border_bridges,
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")
        self.assertIn("Domestic pathway is invalid for a foreign-source event", result["validation_note"])

    def test_opportunity_country_not_defaulted(self):
        """If LLM does not set opportunity_country, it should NOT default to perspective."""
        opp = {
            "opportunity_id": "OPP-006",
            "title": "Missing geography",
            "source_country": "Zambia",
            "event_country": "Zambia",
            # No opportunity_country set!
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=self.cross_border_bridges,
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")
        self.assertIn("Missing opportunity_country", result["validation_note"])
        self.assertNotEqual(result["opportunity_country"], "Zimbabwe")

    def test_cross_border_false_without_bridge(self):
        """Cross-border should be False when no bridges exist, even with different countries."""
        opp = {
            "opportunity_id": "OPP-007",
            "title": "No bridge",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=[],  # No bridges
        )
        self.assertFalse(result["cross_border"])
        self.assertEqual(result["cross_border_countries"], [])

    def test_valid_domestic_opportunity(self):
        """A domestic opportunity for a domestic event should be VALID."""
        opp = {
            "opportunity_id": "OPP-008",
            "title": "Domestic Zimbabwe opportunity",
            "source_country": "Zimbabwe",
            "event_country": "Zimbabwe",
            "opportunity_country": "Zimbabwe",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "domestic procurement",
            "source_nodes": ["Hwange Power Station"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=[],  # No bridges needed for domestic
        )
        self.assertEqual(result["status"], "VALID")
        self.assertFalse(result["cross_border"])

    def test_perspective_actor_node_id_set(self):
        """When actor matches a node, perspective_actor_node_id should be set."""
        opp = {
            "opportunity_id": "OPP-009",
            "title": "Actor match",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, self.perspective,
            perspective_node_ids=self.perspective_node_ids,
            cross_border_bridges=self.cross_border_bridges,
        )
        self.assertEqual(result.get("perspective_actor_node_id"), "ZESA Holdings")

    def test_country_agnostic_perspective(self):
        """The validator should work with any perspective country, not just Zimbabwe."""
        sa_perspective = PerspectiveContext("South Africa", "ZA")
        sa_node_ids = {"Eskom"}
        sa_bridges = [
            {
                "from_node": "Eskom",
                "from_country": "South Africa",
                "to_node": "Zambia REAP",
                "to_country": "Zambia",
                "relationship_type": "outbound_link",
            }
        ]
        opp = {
            "opportunity_id": "OPP-010",
            "title": "SA perspective on Zambia",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "Eskom",
            "perspective_capability": "Power generation",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            opp, sa_perspective,
            perspective_node_ids=sa_node_ids,
            cross_border_bridges=sa_bridges,
        )
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["perspective_actor_evidence"])


class TestVaultRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.vault_path = create_mock_vault(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_perspective_node_retrieval(self):
        """Test that perspective-side nodes are correctly retrieved."""
        # Import here to avoid circular import issues
        from ATIS_News import ObsidianVaultManager

        mgr = ObsidianVaultManager(self.vault_path)
        perspective = PerspectiveContext("Zimbabwe", "ZW")

        nodes = mgr.get_perspective_nodes(perspective)
        node_ids = {n["node_id"] for n in nodes}

        self.assertIn("ZESA Holdings", node_ids)
        self.assertIn("Hwange Power Station", node_ids)
        self.assertIn("ZERA", node_ids)
        self.assertIn("Bikita Minerals", node_ids)
        self.assertNotIn("ZESCO", node_ids)  # Zambia node
        self.assertNotIn("Eskom", node_ids)  # SA node
        self.assertEqual(len(nodes), 4)

    def test_cross_border_bridge_discovery(self):
        """Test that cross-border bridges are discovered via link traversal."""
        from ATIS_News import ObsidianVaultManager

        mgr = ObsidianVaultManager(self.vault_path)
        perspective = PerspectiveContext("Zimbabwe", "ZW")

        bridges = mgr.get_cross_border_bridges(perspective, "Zambia")

        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0]["from_node"], "ZESA Holdings")
        self.assertEqual(bridges[0]["from_country"], "Zimbabwe")
        self.assertEqual(bridges[0]["to_node"], "Zambia REAP")
        self.assertEqual(bridges[0]["to_country"], "Zambia")

    def test_no_bridges_for_same_country(self):
        """No bridges should be found when source and perspective are the same."""
        from ATIS_News import ObsidianVaultManager

        mgr = ObsidianVaultManager(self.vault_path)
        perspective = PerspectiveContext("Zimbabwe", "ZW")

        bridges = mgr.get_cross_border_bridges(perspective, "Zimbabwe")
        self.assertEqual(len(bridges), 0)

    def test_perspective_nodes_from_query_engine(self):
        """Test perspective retrieval via the Query engine's vault manager."""
        from ATIS_Query import ObsidianVaultManager

        mgr = ObsidianVaultManager(self.vault_path)
        mgr.build_index()
        perspective = PerspectiveContext("Zimbabwe", "ZW")

        nodes = mgr.get_perspective_nodes(perspective)
        node_ids = {n.uid for n in nodes}

        self.assertIn("ZESA Holdings", node_ids)
        self.assertIn("Bikita Minerals", node_ids)
        self.assertNotIn("ZESCO", node_ids)

    def test_cross_border_bridges_from_query_engine(self):
        """Test bridge discovery via the Query engine's vault manager."""
        from ATIS_Query import ObsidianVaultManager

        mgr = ObsidianVaultManager(self.vault_path)
        mgr.build_index()
        perspective = PerspectiveContext("Zimbabwe", "ZW")

        bridges = mgr.get_cross_border_bridges(perspective, "Zambia")

        self.assertTrue(len(bridges) > 0)
        bridge_countries = set()
        for b in bridges:
            bridge_countries.add(b["from_country"])
            bridge_countries.add(b["to_country"])
        self.assertIn("Zimbabwe", bridge_countries)
        self.assertIn("Zambia", bridge_countries)


class TestExecuteValidation(unittest.TestCase):
    def setUp(self):
        self.perspective = PerspectiveContext("Zimbabwe", "ZW")

    def test_valid_opportunity_passes_execution_gate(self):
        """A VALID opportunity should pass execution validation."""
        from ATIS_Execute import validate_opportunity_for_execution

        opp = {
            "status": "VALID",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "perspective_actor_evidence": True,
            "perspective_capability_evidence": True,
            "pathway_evidence": True,
        }
        is_valid, msg = validate_opportunity_for_execution(opp)
        self.assertTrue(is_valid)
        self.assertIn("validated", msg.lower())

    def test_research_required_blocked(self):
        """RESEARCH_REQUIRED opportunities should be blocked from execution."""
        from ATIS_Execute import validate_opportunity_for_execution

        opp = {
            "status": "RESEARCH_REQUIRED",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "perspective_actor_evidence": False,
            "perspective_capability_evidence": False,
            "pathway_evidence": False,
        }
        is_valid, msg = validate_opportunity_for_execution(opp)
        self.assertFalse(is_valid)
        self.assertIn("RESEARCH_REQUIRED", msg)

    def test_missing_actor_blocked(self):
        """Missing perspective_actor should block execution."""
        from ATIS_Execute import validate_opportunity_for_execution

        opp = {
            "status": "VALID",
            "perspective_actor": "",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "perspective_actor_evidence": False,
            "perspective_capability_evidence": True,
            "pathway_evidence": True,
        }
        is_valid, msg = validate_opportunity_for_execution(opp)
        self.assertFalse(is_valid)
        self.assertIn("perspective_actor", msg)

    def test_missing_evidence_blocked(self):
        """Missing evidence flags should block execution even if status is VALID."""
        from ATIS_Execute import validate_opportunity_for_execution

        opp = {
            "status": "VALID",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "cross-border energy trade",
            "perspective_actor_evidence": False,
            "perspective_capability_evidence": True,
            "pathway_evidence": True,
        }
        is_valid, msg = validate_opportunity_for_execution(opp)
        self.assertFalse(is_valid)
        self.assertIn("not evidenced", msg)


# =============================================================================
# Integration Test: End-to-End Pipeline Validation
# =============================================================================
class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.vault_path = create_mock_vault(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_zimbabwe_perspective_zambia_event(self):
        """
        Zimbabwe perspective + Zambia event:
        - Should retrieve Zimbabwe nodes
        - Should discover ZESA Holdings -> Zambia REAP bridge
        - Should validate cross-border opportunity as VALID
        - Should reject domestic-only opportunities
        """
        from ATIS_News import ObsidianVaultManager
        from atis_context import validate_opportunity

        mgr = ObsidianVaultManager(self.vault_path)
        perspective = PerspectiveContext("Zimbabwe", "ZW")

        # Simulate entity extraction from a Zambia REAP article
        entities = [
            {"name": "Zambia REAP", "class": "[POLICY_FRAMEWORK]", "context": "Zambia launches Rural Electricity Access Project"},
            {"name": "ZESCO", "class": "[INFRASTRUCTURE_NODE]", "context": "ZESCO implements the project"},
        ]

        graph_context, perspective_nodes, cross_border_bridges = mgr.build_graph_context(entities, perspective)
        perspective_node_ids = {pn["node_id"] for pn in perspective_nodes}

        # Verify perspective nodes were retrieved
        self.assertTrue(len(perspective_nodes) > 0)
        self.assertIn("ZESA Holdings", perspective_node_ids)

        # Verify cross-border bridges were discovered
        self.assertTrue(len(cross_border_bridges) > 0)
        bridge = cross_border_bridges[0]
        self.assertEqual(bridge["from_node"], "ZESA Holdings")
        self.assertEqual(bridge["to_node"], "Zambia REAP")

        # Test VALID cross-border opportunity
        valid_opp = {
            "opportunity_id": "OPP-ZW-ZM-001",
            "title": "Cross-border power supply to Zambia REAP",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "National grid operations and cross-border power trade",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            valid_opp, perspective,
            perspective_node_ids=perspective_node_ids,
            cross_border_bridges=cross_border_bridges,
        )
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["cross_border"])
        self.assertEqual(result["opportunity_country"], "Zambia")

        # Test RESEARCH_REQUIRED: invented actor
        fake_opp = {
            "opportunity_id": "OPP-ZW-ZM-002",
            "title": "Fake Zimbabwe company opportunity",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "Fake Zimbabwe Power Corp",
            "perspective_capability": "Fake capability",
            "pathway": "cross-border energy trade",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            fake_opp, perspective,
            perspective_node_ids=perspective_node_ids,
            cross_border_bridges=cross_border_bridges,
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")

        # Test RESEARCH_REQUIRED: domestic pathway for foreign event
        domestic_opp = {
            "opportunity_id": "OPP-ZW-ZM-003",
            "title": "Domestic procurement for Zambia event",
            "source_country": "Zambia",
            "event_country": "Zambia",
            "opportunity_country": "Zambia",
            "perspective_actor": "ZESA Holdings",
            "perspective_capability": "Grid operations",
            "pathway": "domestic procurement",
            "source_nodes": ["Zambia REAP"],
        }
        result = validate_opportunity(
            domestic_opp, perspective,
            perspective_node_ids=perspective_node_ids,
            cross_border_bridges=cross_border_bridges,
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")

    def test_zambia_perspective_zimbabwe_event(self):
        """
        Zambia perspective + Zimbabwe event:
        - Should retrieve Zambia nodes
        - Should validate Zambia-relevant opportunities
        """
        from ATIS_News import ObsidianVaultManager
        from atis_context import validate_opportunity

        mgr = ObsidianVaultManager(self.vault_path)
        perspective = PerspectiveContext("Zambia", "ZM")

        entities = [
            {"name": "Hwange Power Station", "class": "[INFRASTRUCTURE_NODE]", "context": "Zimbabwe expands Hwange Power Station"},
        ]

        graph_context, perspective_nodes, cross_border_bridges = mgr.build_graph_context(entities, perspective)
        perspective_node_ids = {pn["node_id"] for pn in perspective_nodes}

        # Zambia perspective should retrieve Zambia nodes
        self.assertTrue(len(perspective_nodes) > 0)
        self.assertIn("ZESCO", perspective_node_ids)
        self.assertNotIn("ZESA Holdings", perspective_node_ids)

    def test_no_cross_border_without_bridge(self):
        """
        Even with different countries, cross-border should be False without bridge evidence.
        """
        from ATIS_News import ObsidianVaultManager
        from atis_context import validate_opportunity

        # Create a vault WITHOUT cross-border links
        tmpdir2 = Path(tempfile.mkdtemp())
        vault2 = tmpdir2 / "vault"
        vault2.mkdir()

        zim_dir = vault2 / "Zimbabwe"
        zim_dir.mkdir()
        (zim_dir / "ZESA Holdings.md").write_text(
            "---\ncountry: Zimbabwe\n---\n# ZESA Holdings\nNational power utility.\n"
        )

        zam_dir = vault2 / "Zambia"
        zam_dir.mkdir()
        (zam_dir / "Zambia REAP.md").write_text(
            "---\ncountry: Zambia\n---\n# Zambia REAP\nRural electrification.\n"
        )

        try:
            mgr = ObsidianVaultManager(vault2)
            perspective = PerspectiveContext("Zimbabwe", "ZW")

            entities = [
                {"name": "Zambia REAP", "class": "[POLICY_FRAMEWORK]", "context": "Zambia launches REAP"},
            ]

            graph_context, perspective_nodes, cross_border_bridges = mgr.build_graph_context(entities, perspective)
            perspective_node_ids = {pn["node_id"] for pn in perspective_nodes}

            # No bridges because no links between countries
            self.assertEqual(len(cross_border_bridges), 0)

            opp = {
                "opportunity_id": "OPP-NO-BRIDGE",
                "title": "Unbridged opportunity",
                "source_country": "Zambia",
                "event_country": "Zambia",
                "opportunity_country": "Zambia",
                "perspective_actor": "ZESA Holdings",
                "perspective_capability": "Grid operations",
                "pathway": "cross-border energy trade",
                "source_nodes": ["Zambia REAP"],
            }
            result = validate_opportunity(
                opp, perspective,
                perspective_node_ids=perspective_node_ids,
                cross_border_bridges=cross_border_bridges,
            )
            self.assertEqual(result["status"], "RESEARCH_REQUIRED")
            self.assertFalse(result["cross_border"])
        finally:
            import shutil
            shutil.rmtree(tmpdir2, ignore_errors=True)


# =============================================================================
# Run Tests
# =============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
