"""
Research Query Generator

Generates targeted search queries for each STORM perspective,
incorporating SEO keywords and GEO terms naturally.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.analysis import Section, generate_research_queries


class ResearchQuery(BaseModel):
    """Research query"""
    query: str
    perspective: str
    seo_keyword: bool = False
    geo_keyword: bool = False


# Content Type Modifiers
CONTENT_TYPE_MODIFIERS = {
    "blog-post": ["guide", "tutorial", "tips", "best practices"],
    "guide": ["how to", "step by step", "tutorial", "complete guide"],
    "comparison": ["vs", "comparison", "alternatives", "review"],
    "case-study": ["case study", "success story", "example", "results"],
    "local-guide": ["in", "near", "local", "best", "top rated"],
}


def generate_research_queries(
    section: Section,
    perspectives: List[str],
    seo_keywords: List[str],
    geo_keywords: List[str] = [],
) -> List[ResearchQuery]:
    """
    Generate targeted search queries for a STORM section.
    
    Args:
        section: Content section to generate queries for
        perspectives: List of perspective names
        seo_keywords: Primary and secondary SEO keywords
        geo_keywords: Local keywords for GEO targeting
    
    Returns:
        List of research queries
    """
    
    queries: List[ResearchQuery] = []
    
    # Base queries from section heading and subsections
    queries.append(ResearchQuery(
        query=section.heading,
        perspective="general",
        seo_keyword=False,
        geo_keyword=False,
    ))
    
    for subsection in section.subsections:
        queries.append(ResearchQuery(
            query=subsection.heading,
            perspective="general",
            seo_keyword=False,
            geo_keyword=False,
        ))
        for question in subsection.questions_to_answer:
            queries.append(ResearchQuery(
                query=question,
                perspective="general",
                seo_keyword=False,
                geo_keyword=False,
            ))
    
    # Add perspective-specific queries
    for perspective in perspectives:
        queries.append(ResearchQuery(
            query=f"{section.heading} {perspective} perspective",
            perspective=perspective,
            seo_keyword=False,
            geo_keyword=False,
        ))
    
    # Add SEO keyword queries
    if seo_keywords:
        primary_keyword = seo_keywords[0] if seo_keywords else ""
        
        queries.append(ResearchQuery(
            query=f"{primary_keyword} {section.heading}",
            perspective="seo",
            seo_keyword=True,
            geo_keyword=False,
        ))
        
        for kw in seo_keywords[1:]:
            queries.append(ResearchQuery(
                query=f"{kw} {section.heading}",
                perspective="seo",
                seo_keyword=True,
                geo_keyword=False,
            ))
    
    # Add GEO queries if enabled
    if section.include_location and geo_keywords:
        for geo_kw in geo_keywords:
            queries.append(ResearchQuery(
                query=f"{geo_kw} {section.heading}",
                perspective="geo",
                seo_keyword=False,
                geo_keyword=True,
            ))
    
    # Add content type modifiers
    content_type = section.__dict__.get("content_type", "blog-post")
    modifiers = CONTENT_TYPE_MODIFIERS.get(content_type, [])
    
    for modifier in modifiers:
        queries.append(ResearchQuery(
            query=f"{section.heading} {modifier}",
            perspective="general",
            seo_keyword=False,
            geo_keyword=False,
        ))
    
    return queries


def calculate_query_complexity(
    queries: List[ResearchQuery],
) -> float:
    """
    Calculate a complexity score for a set of queries.
    
    Args:
        queries: List of research queries
    
    Returns:
        Complexity score (0-100)
    """
    if not queries:
        return 0.0
    
    # Factors that increase complexity
    geo_queries = sum(1 for q in queries if q.geo_keyword)
    perspective_queries = sum(1 for q in queries if q.perspective != "general")
    seo_queries = sum(1 for q in queries if q.seo_keyword)
    content_type_queries = sum(1 for q in queries if q.perspective == "general" and q.query == q.query.split()[0])
    
    # Base complexity
    base_score = min(100, len(queries) * 5)
    
    # Add complexity factors
    complexity_score = base_score + (geo_queries * 2) + (perspective_queries * 3) + (seo_queries * 1) - (content_type_queries * 1)
    
    # Normalize to 0-100
    return min(100, complexity_score)


def prioritize_queries(
    queries: List[ResearchQuery],
    max_queries: int = 15,
) -> List[ResearchQuery]:
    """
    Prioritize research queries based on importance.
    
    Args:
        queries: List of research queries
        max_queries: Maximum number of queries to return
    
    Returns:
        Prioritized list of research queries
    """
    if not queries:
        return []
    
    # Calculate complexity scores
    query_scores = [(q, calculate_query_complexity([q])) for q in queries]
    
    # Sort by complexity (higher complexity = higher priority)
    sorted_queries = sorted(query_scores, key=lambda x: x[1], reverse=True)
    
    # Take top N queries
    prioritized_queries = [q[0] for q in sorted_queries[:max_queries]]
    
    # Add remaining queries with lower priority
    remaining_queries = [q[0] for q in sorted_queries[max_queries:]]
    
    return prioritized_queries + remaining_queries
