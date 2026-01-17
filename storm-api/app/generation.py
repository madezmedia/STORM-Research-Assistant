"""
Content Generation Engine

Section writers that generate content from research data using
specific perspectives and incorporating SEO/GEO requirements.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from app.analysis import StormOutline, Section
from app.llm import generate_text, get_recommended_model


class SectionWriter(BaseModel):
    """Section writer configuration"""
    section: Section
    research_data: Dict[str, Any]
    tone: str = "professional"
    word_count_target: int = 500
    include_local_data: bool = False
    seo_primary_keyword: Optional[str] = None
    seo_secondary_keywords: List[str] = []


class SectionWriterResponse(BaseModel):
    """Section writer response"""
    section_heading: str
    content: str
    word_count: int
    sources_used: List[str]


class ContentAssembler(BaseModel):
    """Content assembler configuration"""
    sections: List[SectionWriterResponse]
    introduction: Optional[str] = None
    conclusion: Optional[str] = None
    total_word_count: int = 0


class QualityChecker(BaseModel):
    """Quality checker configuration"""
    content: str
    check_grammar: bool = True
    check_spelling: bool = True
    check_readability: bool = True
    check_facts: bool = True
    check_duplicates: bool = True


class QualityScore(BaseModel):
    """Quality score response"""
    grammar_score: float
    spelling_score: float
    readability_score: float
    flesch_grade: str
    overall_score: float
    issues: List[str]


class SEOOptimizer(BaseModel):
    """SEO optimizer configuration"""
    content: str
    primary_keyword: str
    secondary_keywords: List[str]
    target_density: float = 1.5


class GEOEnhancer(BaseModel):
    """GEO enhancer configuration"""
    content: str
    location: str
    local_keywords: List[str]
    target_mentions: int = 3


# Section Writer Prompts
SECTION_WRITER_PROMPT = """
You are writing a section for an SEO-optimized blog post.

SECTION TO WRITE:
Heading: "{section.heading}"
Perspective: {section.perspective}

RESEARCH DATA AVAILABLE:
{research_data}

LOCAL DATA (use naturally):
- City: {location}
- Local businesses: {local_data.businesses[:3] if local_data.businesses else "N/A"}
- Local landmarks: {local_data.landmarks[:2] if local_data.landmarks else "N/A"}

SEO REQUIREMENTS:
{seo_requirements}

WRITING INSTRUCTIONS:
1. Write {word_count_target} words
2. Start with a compelling opening sentence
3. Answer all questions listed in subsections
4. Include specific examples and statistics with sources
5. Use active voice and action verbs
6. End with a transition to next section or clear takeaway
7. Format with proper markdown (H3 for subsections, bold for emphasis)

{local_requirements}

Write the complete section now:
"""


# Section Writers
async def write_section_from_perspective(
    section: Section,
    research_data: Dict[str, Any],
    config: SectionWriter,
    model: str = None,
) -> SectionWriterResponse:
    """
    Write a content section from a specific perspective using LLM.

    Args:
        section: The section definition from STORM outline
        research_data: Research data including web search results and local data
        config: Section writer configuration
        model: Optional LLM model override

    Returns:
        SectionWriterResponse with generated content
    """
    # Extract local data for GEO enhancement
    local_data = research_data.get("local_data", {})
    local_businesses = local_data.get("businesses", [])
    local_landmarks = local_data.get("landmarks", [])
    web_results = research_data.get("web_results", [])

    # Format local data section
    local_context = ""
    if config.include_local_data:
        city = research_data.get("city", "")
        if city:
            local_context += f"\nLOCAL CONTEXT:\n- City: {city}\n"
        if local_businesses:
            business_names = ", ".join(b.get('name', '') for b in local_businesses[:5] if b.get('name'))
            if business_names:
                local_context += f"- Notable local businesses: {business_names}\n"
        if local_landmarks:
            landmark_names = ", ".join(l.get('name', '') for l in local_landmarks[:3] if l.get('name'))
            if landmark_names:
                local_context += f"- Local landmarks: {landmark_names}\n"

    # Format research data
    research_context = ""
    if web_results:
        research_context = "\nRESEARCH DATA:\n"
        for i, result in enumerate(web_results[:5], 1):
            title = result.get('title', '')
            snippet = result.get('snippet', result.get('content', ''))[:300]
            url = result.get('url', '')
            if title and snippet:
                research_context += f"{i}. {title}\n   {snippet}...\n   Source: {url}\n\n"

    # Format subsection questions
    subsection_questions = ""
    for subsection in section.subsections:
        subsection_questions += f"\n### {subsection.heading}\n"
        if subsection.questions_to_answer:
            subsection_questions += "Questions to address:\n"
            for q in subsection.questions_to_answer:
                subsection_questions += f"- {q}\n"
        if subsection.stats_needed:
            subsection_questions += "Statistics to include:\n"
            for stat in subsection.stats_needed:
                subsection_questions += f"- {stat}\n"

    # Build the prompt
    prompt = f"""Write a detailed, engaging section for a blog post.

SECTION DETAILS:
- Heading: {section.heading}
- Perspective: {section.perspective}
- Target word count: {config.word_count_target} words

{research_context}
{local_context}

SUBSECTIONS TO COVER:
{subsection_questions}

SEO REQUIREMENTS:
- Primary keyword: "{config.seo_primary_keyword or 'the topic'}"
- Include keyword naturally 2-3 times
- Use in at least one subheading

WRITING GUIDELINES:
1. Write in a {config.tone} tone
2. Start with a compelling hook that draws readers in
3. Use short paragraphs (2-3 sentences max)
4. Include specific examples, statistics, or case studies from the research
5. Address each subsection question thoroughly
6. Use bullet points or numbered lists where appropriate
7. End with a clear takeaway or transition
8. Format with proper markdown (## for main heading, ### for subsections)
9. DO NOT include meta commentary or instructions - write the actual content

Write the complete section now (aim for {config.word_count_target} words):"""

    # System prompt for content generation
    system_prompt = """You are an expert content writer for Mad EZ Media.
Write engaging, SEO-optimized content that provides genuine value to readers.
Use a conversational yet professional tone. Include specific examples and data.
Format output in clean markdown without excessive formatting."""

    # Generate content using LLM
    content = await generate_text(
        prompt=prompt,
        model=model or get_recommended_model("content_generation"),
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=2048,
    )

    # Calculate word count and extract sources
    word_count = len(content.split())
    sources_used = [r.get('url', '') for r in web_results[:5] if r.get('url')]

    return SectionWriterResponse(
        section_heading=section.heading,
        content=content,
        word_count=word_count,
        sources_used=sources_used,
    )


async def write_all_sections(
    outline: StormOutline,
    research_data: Dict[str, Any],
    config: ContentAssembler,
    model: str = None,
) -> Dict[str, Any]:
    """
    Write all sections of the outline using parallel LLM calls.

    Args:
        outline: STORM outline with sections to write
        research_data: Research data for content generation
        config: Content assembler configuration
        model: Optional LLM model override

    Returns:
        Dict with introduction, sections, conclusion, and total word count
    """
    # Extract primary keyword from title or meta description
    primary_keyword = ""
    if outline.title:
        # Use first meaningful word(s) from title
        words = outline.title.split()
        primary_keyword = " ".join(words[:3]) if len(words) >= 3 else outline.title

    # Calculate word count per section
    num_sections = len(outline.sections)
    word_count_per_section = max(300, config.total_word_count // max(1, num_sections))

    # Create section writer configs
    async def write_single_section(section: Section) -> Dict[str, Any]:
        writer_config = SectionWriter(
            section=section,
            research_data=research_data,
            tone="professional",
            word_count_target=word_count_per_section,
            include_local_data=config.include_local_data if hasattr(config, 'include_local_data') else False,
            seo_primary_keyword=primary_keyword,
        )

        result = await write_section_from_perspective(
            section=section,
            research_data=research_data,
            config=writer_config,
            model=model,
        )

        return {
            "section_heading": result.section_heading,
            "content": result.content,
            "word_count": result.word_count,
            "sources_used": result.sources_used,
        }

    # Write all sections in parallel for faster generation
    section_tasks = [write_single_section(section) for section in outline.sections]
    section_results = await asyncio.gather(*section_tasks, return_exceptions=True)

    # Process results, handling any errors
    sections = []
    total_word_count = 0
    for i, result in enumerate(section_results):
        if isinstance(result, Exception):
            # Log error but continue with placeholder
            sections.append({
                "section_heading": outline.sections[i].heading,
                "content": f"## {outline.sections[i].heading}\n\nContent generation failed: {str(result)}",
                "word_count": 10,
                "sources_used": [],
            })
        else:
            sections.append(result)
            total_word_count += result.get("word_count", 0)

    # Generate introduction using LLM
    introduction = ""
    if config.introduction:
        intro_prompt = f"""Write a compelling introduction for an article titled "{outline.title}".

The article covers these main sections:
{chr(10).join('- ' + s.heading for s in outline.sections)}

Meta description: {outline.meta_description}

Write 2-3 paragraphs that:
1. Hook the reader with an interesting fact or question
2. Explain what they'll learn
3. Preview the value they'll get from reading

Keep it under 150 words. Write in markdown format."""

        try:
            introduction = await generate_text(
                prompt=intro_prompt,
                model=model or get_recommended_model("content_generation"),
                temperature=0.7,
                max_tokens=500,
            )
            total_word_count += len(introduction.split())
        except Exception:
            introduction = f"# {outline.title}\n\n{outline.meta_description}\n\n"

    # Generate conclusion using LLM
    conclusion = ""
    if config.conclusion:
        conclusion_prompt = f"""Write a strong conclusion for an article titled "{outline.title}".

The article covered:
{chr(10).join('- ' + s.heading for s in outline.sections)}

Write 2-3 paragraphs that:
1. Summarize the key takeaways
2. Provide a clear call-to-action
3. End with a memorable closing statement

Keep it under 150 words. Start with "## Conclusion" heading."""

        try:
            conclusion = await generate_text(
                prompt=conclusion_prompt,
                model=model or get_recommended_model("content_generation"),
                temperature=0.7,
                max_tokens=500,
            )
            total_word_count += len(conclusion.split())
        except Exception:
            conclusion = f"\n\n## Conclusion\n\nWe've covered the key aspects of {outline.title}. Use this information to make informed decisions.\n\n"

    return {
        "introduction": introduction,
        "sections": sections,
        "conclusion": conclusion,
        "total_word_count": total_word_count,
    }


# Quality Checker
async def check_grammar(content: str) -> float:
    """Check grammar score"""
    # TODO: Implement grammar checking
    return 95.0


async def check_spelling(content: str) -> float:
    """Check spelling score"""
    # TODO: Implement spelling checking
    return 98.0


async def check_readability(content: str) -> float:
    """Check readability score"""
    # TODO: Implement readability analysis
    word_count = len(content.split())
    avg_sentence_length = word_count / (content.count('.') + 1)
    
    # Flesch-Kincaid grade approximation
    if avg_sentence_length < 15:
        return 90.0
    elif avg_sentence_length < 20:
        return 85.0
    elif avg_sentence_length < 25:
        return 75.0
    else:
        return 65.0


async def check_facts(content: str, sources: List[str]) -> List[str]:
    """Verify facts against sources"""
    # TODO: Implement fact verification
    return []


async def check_duplicates(content: str) -> bool:
    """Check for duplicate content"""
    # TODO: Implement duplicate detection
    return False


async def run_quality_checks(
    content: str,
    sources: List[str],
    config: QualityChecker,
) -> QualityScore:
    """Run all quality checks"""
    
    scores = {}
    issues = []
    
    if config.check_grammar:
        scores["grammar"] = await check_grammar(content)
    
    if config.check_spelling:
        scores["spelling"] = await check_spelling(content)
    
    if config.check_readability:
        scores["readability"] = await check_readability(content)
    
    if config.check_facts:
        issues.extend(await check_facts(content, sources))
    
    if config.check_duplicates:
        has_duplicates = await check_duplicates(content)
        if has_duplicates:
            issues.append("Duplicate content detected")
    
    # Calculate overall score
    if scores:
        overall = sum(scores.values()) / len(scores)
    else:
        overall = 0.0
    
    # Flesch grade
    if scores.get("readability"):
        flesch = "A" if scores["readability"] >= 90 else "B" if scores["readability"] >= 80 else "C"
    else:
        flesch = "B"
    
    return QualityScore(
        grammar_score=scores.get("grammar", 0.0),
        spelling_score=scores.get("spelling", 0.0),
        readability_score=scores.get("readability", 0.0),
        flesch_grade=flesch,
        overall_score=overall,
        issues=issues,
    )


# SEO Optimizer
async def optimize_seo(
    content: str,
    config: SEOOptimizer,
) -> Dict[str, Any]:
    """Optimize content for SEO"""
    
    optimizations = []
    
    # Check keyword placement
    primary_keyword = config.primary_keyword
    title_has_keyword = primary_keyword.lower() in content[:100].lower()
    first_paragraph_has_keyword = primary_keyword.lower() in content[:500].lower()
    
    if not title_has_keyword:
        optimizations.append({
            "type": "keyword-placement",
            "priority": "critical",
            "message": f"Primary keyword '{primary_keyword}' not in title",
            "auto_fix": True,
        })
    
    if not first_paragraph_has_keyword:
        optimizations.append({
            "type": "keyword-placement",
            "priority": "critical",
            "message": f"Primary keyword '{primary_keyword}' not in first paragraph",
            "auto_fix": True,
        })
    
    # Check keyword density
    words = content.split()
    if len(words) > 0:
        density = (content.lower().count(primary_keyword.lower()) / len(words)) * 100
        if density < 0.01:
            optimizations.append({
                "type": "keyword-density",
                "priority": "high",
                "message": f"Keyword density is {density:.2f}%. Target: 1-2%",
                "fix": f"Add {max(1, int((0.015 - density) * len(words)))} more mentions of '{primary_keyword}'",
            })
        elif density > 0.03:
            optimizations.append({
                "type": "keyword-density",
                "priority": "medium",
                "message": f"Keyword density is {density:.2f}%. Target: 1-2%",
            })
    
    # Check secondary keywords
    for kw in config.secondary_keywords:
        if kw.lower() not in content.lower():
            optimizations.append({
                "type": "keyword-usage",
                "priority": "medium",
                "message": f"Secondary keyword '{kw}' not used",
            })
    
    return {
        "optimizations": optimizations,
        "keyword_density": density if len(words) > 0 else 0.0,
    }


# GEO Enhancer
async def enhance_geo(
    content: str,
    config: GEOEnhancer,
) -> Dict[str, Any]:
    """Enhance content with GEO targeting"""
    
    enhancements = []
    
    # Check location mentions
    location = config.location
    if location and location.lower() not in content.lower():
        mentions = content.lower().count(location.lower())
        
        if mentions < config.target_mentions:
            enhancements.append({
                "type": "geo-location",
                "priority": "high",
                "message": f"Only {mentions} mentions of '{location}'. Target: {config.target_mentions}",
                "suggestion": f"Add to an H2 heading: \"How {location} Businesses...\"",
            })
    
    # Check local keywords
    for kw in config.local_keywords:
        if kw.lower() not in content.lower():
            enhancements.append({
                "type": "geo-keywords",
                "priority": "medium",
                "message": f"Local keyword '{kw}' not used",
            })
    
    geo_score = min(100, (mentions / config.target_mentions) * 100)
    
    return {
        "enhancements": enhancements,
        "geo_score": geo_score,
        "location_mentions": mentions if location else 0,
    }
