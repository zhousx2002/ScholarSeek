from __future__ import annotations


QUERY_REFINEMENT_SYSTEM = """You optimize natural-language research questions for academic APIs.
Return only JSON. Preserve the user's research intent and constraints, remove conversational filler,
and produce concise English queries suitable for OpenAlex, Semantic Scholar, and arXiv."""


QUERY_EVOLUTION_SYSTEM = """You are SPAR's Query Evolver for iterative scholarly retrieval.
Use the original query and relevant retrieved papers to generate new searches that uncover missing
methods, applications, comparisons, limitations, and critiques. Do not repeat searched queries.
Return only a JSON object with an `evolved_queries` array. Each item must contain `query`, `direction`,
and `reason`. Generate at most three concise English academic queries."""


JUDGEMENT_SYSTEM = """You are SPAR's Judgement Agent. Decide whether a candidate paper satisfies the
user's academic information need using its title and abstract. Return only JSON with `decision`
(`related`, `unrelated`, or `uncertain`), `score` (0 to 1), and `reason`. Judge topical and intent
alignment; citation count and recency must not rescue an off-topic paper."""


AUTHORITY_RERANK_SYSTEM = """You rerank already relevant academic papers. Relevance remains primary.
Use venue, citations, and publication year only as secondary authority and timeliness evidence.
Return only JSON with a `papers` array containing `index`, `score`, and `reason`."""


def query_evolution_payload(original_query, searched_queries, papers):
    evidence = []
    for paper in papers[:5]:
        evidence.append(
            {
                "title": paper.title,
                "abstract": (paper.abstract or "")[:1200],
                "year": paper.year,
                "venue": paper.venue,
            }
        )
    return {
        "original_query": original_query,
        "searched_queries": list(searched_queries),
        "relevant_papers": evidence,
        "directions": ["methodology", "applications", "limitations"],
    }


def judgement_payload(query, paper):
    return {
        "query": query,
        "paper": {
            "title": paper.title,
            "abstract": (paper.abstract or "")[:1600],
            "year": paper.year,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
        },
    }
