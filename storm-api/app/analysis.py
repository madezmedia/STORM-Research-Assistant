"""
STORM Analysis Engine

Multi-perspective topic breakdown using 7 key perspectives:
1. Beginner Perspective - Foundational concepts, common misconceptions
2. Business Owner Perspective - Practical benefits, ROI, costs, implementation time
3. Local Market Perspective - Local regulations, competitors, success stories, market uniqueness
4. Technical Perspective - How technology works, requirements, integrations
5. Competitive Perspective - Alternative solutions, comparison to traditional methods, pros/cons
6. Customer Perspective - Pain points, decision factors, user experience
7. Industry Expert Perspective - Best practices, trends, future outlook
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.llm import generate_json, get_recommended_model


class Section(BaseModel):
    """Content section"""
    heading: str
    perspective: str
    include_keyword: bool
    include_location: bool
    subsections: List["Subsection"]


class Subsection(BaseModel):
    """Content subsection"""
    heading: str
    questions_to_answer: List[str]
    examples_needed: int
    stats_needed: List[str]
    local_data_needed: bool


class StormOutline(BaseModel):
    """STORM analysis output"""
    title: str
    meta_description: str
    sections: List[Section]
    research_queries: List[str]
    data_needed: List[Dict[str, Any]]


class ResearchQuery(BaseModel):
    """Research query"""
    query: str
    perspective: str
    seo_keyword: bool
    geo_keyword: bool


# STORM Analysis Prompt Template
STORM_ANALYSIS_PROMPT = """
You are a STORM analysis engine for Mad EZ Media / SonicBrand AI.

TOPIC: "{topic}"
CONTENT TYPE: "{content_type}"

Your task is to break down this topic into a comprehensive outline using 7 distinct perspectives.

PERSPECTIVES TO ANALYZE:

1. BEGINNER PERSPECTIVE
   - What foundational concepts must someone understand first?
   - What are common misconceptions or myths?
   - What terminology needs to be defined?
   - What are the prerequisites?

2. BUSINESS OWNER PERSPECTIVE
   - What are the practical business benefits?
   - What is the ROI (return on investment)?
   - What are the costs involved (upfront, ongoing)?
   - How long does implementation typically take?
   - What are the risks or challenges?

3. LOCAL MARKET PERSPECTIVE
   - What are the local regulations or legal considerations?
   - Who are the local competitors in {location}?
   - What are the local success stories or case studies?
   - What makes this market unique locally?

4. TECHNICAL PERSPECTIVE
   - How does the technology or solution actually work?
   - What are the technical requirements?
   - What integrations are available?
   - What is the learning curve?

5. COMPETITIVE PERSPECTIVE
   - What are the alternative solutions or approaches?
   - How does this compare to traditional methods?
   - What are the pros and cons of each approach?
   - What are the key differentiators?

6. CUSTOMER PERSPECTIVE
   - What are the pain points or problems being solved?
   - What are the decision factors when choosing a solution?
   - What is the user experience like?
   - What are the common objections or concerns?

7. INDUSTRY EXPERT PERSPECTIVE
   - What are the industry best practices?
   - What are the current trends and future outlook?
   - What are the common pitfalls to avoid?
   - What innovations are on the horizon?

OUTPUT FORMAT:
Return a JSON object with:
{{
  "title": "SEO-optimized title including primary keyword",
  "meta_description": "150-160 character description",
  "sections": [
    {{
      "heading": "H2 section heading",
      "perspective": "which perspective this section represents",
      "include_keyword": true/false,
      "include_location": true/false,
      "subsections": [
        {{
          "heading": "H3 subsection heading",
          "questions_to_answer": ["specific question 1", "specific question 2"],
          "examples_needed": number,
          "stats_needed": ["statistic 1", "statistic 2"],
          "local_data_needed": true/false
        }}
      ]
    }}
  ],
  "research_queries": [
    "query 1 for web search",
    "query 2 for web search"
  ],
  "data_needed": [
    {{
      "type": "statistic",
      "query": "specific data to find",
      "source": "where to find this data"
    }}
  ]
}}

SEO REQUIREMENTS:
- Primary keyword: "{primary_keyword}"
- Include naturally in: Title (once), H2 headings (at least 1), first paragraph, conclusion
- Keyword density target: 1-2%
- Title length: 50-60 characters
- Meta description: 150-160 characters

GEO REQUIREMENTS:
{geo_requirements}

Return ONLY valid JSON.
"""


def generate_storm_analysis_prompt(
    topic: str,
    content_type: str,
    primary_keyword: str,
    secondary_keywords: List[str],
    geo_requirements: str = "",
    location: str = "",
) -> str:
    """Generate STORM analysis prompt"""

    geo_section = ""
    if geo_requirements:
        geo_section = f"""
GEO REQUIREMENTS:
- Mention "{geo_requirements}" in: Title, intro, 2-3 section headers
- Include local landmarks, businesses, or statistics
- Reference local radio stations or advertising options
"""

    return STORM_ANALYSIS_PROMPT.format(
        topic=topic,
        content_type=content_type,
        primary_keyword=primary_keyword,
        secondary_keywords=", ".join(f'"{kw}"' for kw in secondary_keywords),
        geo_requirements=geo_section,
        location=location or "your local area",
    )


def generate_research_queries(
    section: Section,
    perspectives: List[str],
    seo_keywords: List[str],
    geo_keywords: List[str] = [],
) -> List[ResearchQuery]:
    """Generate research queries for a section"""
    
    queries: List[ResearchQuery] = []
    
    # Base queries from section heading and subsections
    queries.append(ResearchQuery(
        query=section.heading,
        perspective="general",
        seo_keyword=section.include_keyword,
        geo_keyword=False,
    ))
    
    for subsection in section.subsections:
        queries.append(ResearchQuery(
            query=subsection.heading,
            perspective="general",
            seo_keyword=section.include_keyword,
            geo_keyword=False,
        ))
        for question in subsection.questions_to_answer:
            queries.append(ResearchQuery(
                query=question,
                perspective="general",
                seo_keyword=section.include_keyword,
                geo_keyword=False,
            ))
    
    # Add perspective-specific queries
    for perspective in perspectives:
        queries.append(ResearchQuery(
            query=f"{section.heading} {perspective} perspective",
            perspective=perspective,
            seo_keyword=section.include_keyword,
            geo_keyword=False,
        ))
    
    # Add SEO keyword queries
    if section.include_keyword:
        queries.append(ResearchQuery(
            query=f"{primary_keyword} {section.heading}",
            perspective="seo",
            seo_keyword=True,
            geo_keyword=False,
        ))
        for kw in secondary_keywords:
            queries.append(ResearchQuery(
                query=f"{kw} {section.heading}",
                perspective="seo",
                seo_keyword=True,
                geo_keyword=False,
            ))
    
    # Add GEO queries if enabled
    if section.include_location:
        for geo_kw in geo_keywords:
            queries.append(ResearchQuery(
                query=f"{geo_kw} {section.heading}",
                perspective="geo",
                seo_keyword=False,
                geo_keyword=True,
            ))
    
    return queries


def format_storm_outline(
    title: str,
    meta_description: str,
    sections: List[Section],
    research_queries: List[str],
    data_needed: List[Dict[str, Any]],
) -> StormOutline:
    """Format STORM outline as JSON"""

    return {
        "title": title,
        "meta_description": meta_description,
        "sections": [s.model_dump() for s in sections],
        "research_queries": research_queries,
        "data_needed": data_needed,
    }


def format_geo_requirements(geo_config: Optional[Dict[str, Any]]) -> str:
    """Format GEO requirements for prompt"""
    if not geo_config:
        return ""

    parts = []
    if geo_config.get("city"):
        parts.append(f"City: {geo_config['city']}")
    if geo_config.get("state"):
        parts.append(f"State: {geo_config['state']}")
    if geo_config.get("region"):
        parts.append(f"Region: {geo_config['region']}")
    if geo_config.get("local_keywords"):
        parts.append(f"Local keywords: {', '.join(geo_config['local_keywords'])}")

    return "\n".join(parts) if parts else ""


async def analyze_topic(
    topic: str,
    content_type: str,
    seo_config: Dict[str, Any],
    geo_config: Optional[Dict[str, Any]] = None,
    model: str = None,
) -> StormOutline:
    """
    Analyze a topic using STORM methodology with LLM.

    This function:
    1. Generates a STORM analysis prompt with SEO/GEO requirements
    2. Calls the LLM to break down the topic from 7 perspectives
    3. Parses the JSON response into a StormOutline

    Args:
        topic: The main topic to analyze
        content_type: Type of content (blog-post, guide, comparison, etc.)
        seo_config: SEO configuration dict with:
            - primary_keyword: Main keyword to target
            - secondary_keywords: List of supporting keywords
        geo_config: Optional GEO configuration dict with:
            - city: Target city
            - state: Target state
            - region: Target region
            - local_keywords: List of local keywords
        model: Optional LLM model override

    Returns:
        StormOutline with title, meta_description, sections, and research queries

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    # Extract SEO keywords
    primary_keyword = seo_config.get("primary_keyword", topic)
    secondary_keywords = seo_config.get("secondary_keywords", [])

    # Format GEO requirements
    geo_requirements = format_geo_requirements(geo_config)

    # Extract location from geo_config
    location = ""
    if geo_config:
        city = geo_config.get("city", "")
        state = geo_config.get("state", "")
        if city and state:
            location = f"{city}, {state}"
        elif city:
            location = city
        elif state:
            location = state

    # Generate the STORM analysis prompt
    prompt = generate_storm_analysis_prompt(
        topic=topic,
        content_type=content_type,
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords,
        geo_requirements=geo_requirements,
        location=location,
    )

    # System prompt for better JSON output
    system_prompt = """You are a STORM analysis engine that outputs valid JSON.
Always respond with a complete, valid JSON object matching the requested schema.
Do not include any text before or after the JSON."""

    # Call LLM to generate analysis
    outline_data = await generate_json(
        prompt=prompt,
        model=model or get_recommended_model("analysis"),
        system_prompt=system_prompt,
        temperature=0.4,  # Lower temperature for consistent structure
        max_tokens=4096,
    )

    # Parse sections into proper model structure
    sections = []
    for section_data in outline_data.get("sections", []):
        subsections = []
        for sub_data in section_data.get("subsections", []):
            subsections.append(Subsection(
                heading=sub_data.get("heading", ""),
                questions_to_answer=sub_data.get("questions_to_answer", []),
                examples_needed=sub_data.get("examples_needed", 0),
                stats_needed=sub_data.get("stats_needed", []),
                local_data_needed=sub_data.get("local_data_needed", False),
            ))
        sections.append(Section(
            heading=section_data.get("heading", ""),
            perspective=section_data.get("perspective", ""),
            include_keyword=section_data.get("include_keyword", False),
            include_location=section_data.get("include_location", False),
            subsections=subsections,
        ))

    return StormOutline(
        title=outline_data.get("title", f"Guide to {topic}"),
        meta_description=outline_data.get("meta_description", ""),
        sections=sections,
        research_queries=outline_data.get("research_queries", []),
        data_needed=outline_data.get("data_needed", []),
    )
