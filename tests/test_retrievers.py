import unittest
import socket
import os
from unittest.mock import patch

from scholarseek.arxiv_api import ArxivClient, _parse_entry
from scholarseek.models import Paper, QueryPlan
from scholarseek.openalex import OpenAlexClient
from scholarseek.retrievers import MultiSourceRetriever, _merge_papers, _paper_key, parse_sources
from scholarseek.semantic_scholar import _parse_paper


class RetrieverTest(unittest.TestCase):
    def test_arxiv_retries_socket_timeout_then_fails_cleanly(self):
        client = ArxivClient(timeout=1, pause=0, max_retries=1)
        plan = QueryPlan(original="query", search_queries=["query"])
        with patch("scholarseek.arxiv_api.ARXIV_GATE") as gate:
            with patch("scholarseek.arxiv_api.urlopen", side_effect=socket.timeout("timed out")) as mocked:
                with patch("scholarseek.arxiv_api.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        list(client._search_once("query", plan, 1))

        self.assertEqual(2, mocked.call_count)
        sleep.assert_called_once_with(3.0)
        gate.block.assert_called_once_with(60.0)

    def test_parse_sources_supports_aliases_and_all(self):
        self.assertEqual(parse_sources("all"), ["openalex", "semantic-scholar", "arxiv"])
        self.assertEqual(parse_sources("OpenAlex,s2,arxiv"), ["openalex", "semantic-scholar", "arxiv"])

    def test_paper_key_prefers_doi(self):
        paper = _paper("1", "A Test Paper", doi="https://doi.org/10.123/example")

        self.assertEqual(_paper_key(paper), "doi:10.123/example")

    def test_merge_papers_keeps_best_metadata(self):
        primary = _paper("1", "A Test Paper", abstract="", citations=1, source="OpenAlex")
        secondary = _paper(
            "2",
            "A Test Paper",
            abstract="Detailed abstract.",
            citations=12,
            source="Semantic Scholar",
        )

        merged = _merge_papers(primary, secondary)

        self.assertEqual(merged.abstract, "Detailed abstract.")
        self.assertEqual(merged.citation_count, 12)
        self.assertIn("OpenAlex", merged.source)
        self.assertIn("Semantic Scholar", merged.source)

    def test_multi_source_retriever_dedupes_by_title_when_doi_missing(self):
        class FakeRetriever(MultiSourceRetriever):
            def _client_for(self, source):
                class Client:
                    def search(self, plan, per_query=10):
                        if source == "openalex":
                            return [_paper("doi-paper", "Same Paper", citations=5, source="OpenAlex", doi="https://doi.org/10.1/test")]
                        return [_paper("arxiv-paper", "Same Paper", citations=0, source="arXiv")]

                return Client()

        retriever = FakeRetriever(sources=["openalex", "arxiv"])
        papers = retriever.search(plan=type("Plan", (), {"search_queries": ["q"]})())

        self.assertEqual(len(papers), 1)
        self.assertIn("OpenAlex", papers[0].source)
        self.assertIn("arXiv", papers[0].source)

    def test_openalex_parallel_search_merges_results(self):
        client = OpenAlexClient()
        plan = QueryPlan(original="query", search_queries=["query one", "query two"])

        def fake_search_once(query, _plan, _per_query):
            return [_paper(query, f"Paper for {query}", source="OpenAlex")]

        client._search_once = fake_search_once
        with patch.dict("os.environ", {"SCHOLARSEEK_OPENALEX_WORKERS": "2"}):
            papers = client.search(plan, per_query=1)

        self.assertEqual(
            sorted(paper.title for paper in papers),
            ["Paper for query one", "Paper for query two"],
        )

    def test_semantic_scholar_parse_paper(self):
        parsed = _parse_paper(
            {
                "paperId": "abc",
                "title": "Semantic Scholar Paper",
                "abstract": "abstract",
                "year": 2025,
                "venue": "ACL",
                "authors": [{"name": "Ada"}],
                "externalIds": {"DOI": "10.1000/test"},
                "citationCount": 7,
                "url": "https://example.com",
            }
        )

        self.assertEqual(parsed.source, "Semantic Scholar")
        self.assertEqual(parsed.doi, "https://doi.org/10.1000/test")
        self.assertEqual(parsed.authors, ["Ada"])

    def test_arxiv_parse_entry(self):
        import xml.etree.ElementTree as ET

        xml = """
        <entry xmlns="http://www.w3.org/2005/Atom">
          <id>http://arxiv.org/abs/2501.10120v1</id>
          <published>2025-01-17T00:00:00Z</published>
          <title>PaSa: An LLM Agent for Comprehensive Academic Paper Search</title>
          <summary>We introduce PaSa.</summary>
          <author><name>Yichen He</name></author>
        </entry>
        """
        parsed = _parse_entry(ET.fromstring(xml))

        self.assertEqual(parsed.id, "arxiv:2501.10120")
        self.assertEqual(parsed.year, 2025)
        self.assertEqual(parsed.source, "arXiv")


def _paper(identifier, title, abstract="abstract", citations=0, source="test", doi=None):
    return Paper(
        id=identifier,
        title=title,
        year=2025,
        venue="Test Venue",
        authors=[],
        abstract=abstract,
        doi=doi,
        url=None,
        citation_count=citations,
        source=source,
        raw={},
    )


if __name__ == "__main__":
    unittest.main()
