import unittest

from atis_context import PerspectiveContext, validate_opportunity
from ATIS_Execute import expand_keywords
from Main import _query_cache_key


class PerspectiveArchitectureTests(unittest.TestCase):
    def test_cache_isolated_by_country_code(self):
        question = "What opportunities exist from Zambia's energy investment?"
        self.assertNotEqual(
            _query_cache_key(question, PerspectiveContext.from_values("Zimbabwe", "ZW")),
            _query_cache_key(question, PerspectiveContext.from_values("Zambia", "ZM")),
        )

    def test_valid_cross_border_opportunity(self):
        context = PerspectiveContext.from_values("Zimbabwe", "ZW")
        result = validate_opportunity(
            {
                "source_country": "Zambia",
                "event_country": "Zambia",
                "opportunity_country": "Zambia",
                "perspective_actor": "Zimbabwe environmental consultancy",
                "perspective_capability": "Environmental consulting",
                "pathway": "Can supply verified Zambia demand",
                "source_nodes": ["Zambia event", "Zimbabwe capability"],
            },
            context,
            {"Zambia event", "Zimbabwe capability"},
        )
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["cross_border"])
        self.assertEqual(result["perspective_country_code"], "ZW")

    def test_missing_capability_requires_research(self):
        result = validate_opportunity(
            {"source_country": "Zambia", "source_nodes": ["event"]},
            PerspectiveContext(),
            {"event"},
        )
        self.assertEqual(result["status"], "RESEARCH_REQUIRED")
        self.assertEqual(result["perspective_country"], "Zimbabwe")

    def test_same_event_retains_different_perspectives(self):
        opportunity = {"source_country": "Zambia", "source_nodes": ["event"]}
        zimbabwe = validate_opportunity(opportunity, PerspectiveContext.from_values("Zimbabwe", "ZW"), {"event"})
        zambia = validate_opportunity(opportunity, PerspectiveContext.from_values("Zambia", "ZM"), {"event"})
        self.assertNotEqual(zimbabwe["perspective_country_code"], zambia["perspective_country_code"])
        self.assertEqual(zimbabwe["source_country"], zambia["source_country"])

    def test_perspective_payload_is_explicit(self):
        context = PerspectiveContext.from_payload({
            "perspective_country": "Zambia",
            "perspective_country_code": "ZM",
        })
        self.assertEqual(context.as_dict(), {"country": "Zambia", "country_code": "ZM"})

    def test_execute_seeds_include_perspective_assets(self):
        seeds = expand_keywords({
            "title": "Renewable energy demand",
            "perspective_country": "Zimbabwe",
            "perspective_actor": "Zimbabwe solar company",
            "perspective_capability": "Solar engineering",
            "source_country": "Zambia",
        })
        self.assertIn("Zimbabwe", seeds)
        self.assertIn("Zimbabwe solar company", seeds)
        self.assertIn("Zambia", seeds)


if __name__ == "__main__":
    unittest.main()