"""
SEO Optimizer

Analyzes content for keyword density, placement, and provides
recommendations for optimization.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re


class KeywordAnalysis(BaseModel):
    """Keyword analysis result"""
    keyword: str
    count: int
    density: float
    positions: List[str]
    naturally_placed: bool


class HeadingAnalysis(BaseModel):
    """Heading analysis result"""
    heading: str
    level: str  # h1, h2, h3
    has_keyword: bool
    word_count: int


class SEOScore(BaseModel):
    """Overall SEO score"""
    overall: float
    grade: str  # A-F
    keyword_density: float
    readability: float
    heading_structure: Dict[str, Any]
    recommendations: List[Dict[str, str]]


class SEOOptimizer(BaseModel):
    """SEO optimizer configuration"""
    primary_keyword: str
    secondary_keywords: List[str]
    target_density: float = 1.5  # 1-2%


def analyze_keyword_density(
    content: str,
    keyword: str,
) -> float:
    """
    Calculate keyword density as percentage.
    """
    words = content.lower().split()
    if not words:
        return 0.0
    
    keyword_lower = keyword.lower()
    count = sum(1 for word in words if word.lower() == keyword_lower)
    density = (count / len(words)) * 100
    
    return round(density, 2)


def analyze_keyword_placement(
    content: str,
    keyword: str,
) -> Dict[str, Any]:
    """
    Analyze keyword placement in content.
    """
    content_lower = content.lower()
    keyword_lower = keyword.lower()
    
    positions: {
        "title": [],
        "first_paragraph": [],
        "h2_headings": [],
        "conclusion": [],
    }
    
    # Check title
    if content_lower.startswith(keyword_lower):
        positions["title"].append("Title")
    
    # Check first 500 characters
    first_para = content_lower[:500]
    if keyword_lower in first_para:
        positions["first_paragraph"].append("First paragraph")
    
    # Check H2 headings
    h2_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    for match in h2_pattern.finditer(content_lower):
        h2_heading = match.group().strip()
        if keyword_lower in h2_heading.lower():
            positions["h2_headings"].append(f"H2: {h2_heading}")
    
    # Check conclusion
    if content_lower.endswith(keyword_lower):
        positions["conclusion"].append("Conclusion")
    
    # Calculate placement score
    total_checks = sum(len(v) for v in positions.values())
    placement_score = total_checks / 8 if total_checks > 0 else 0
    
    return {
        "positions": positions,
        "score": placement_score,
        "naturally_placed": placement_score > 0,
    }


def analyze_heading_structure(
    content: str,
) -> Dict[str, Any]:
    """
    Analyze heading structure.
    """
    content_lower = content.lower()
    
    structure = {
        "h1": 0,
        "h2": 0,
        "h3": 0,
    }
    
    # Count headings
    h1_count = len(re.findall(r"^#\s+(.+)$", content_lower))
    h2_count = len(re.findall(r"^##\s+(.+)$", content_lower))
    h3_count = len(re.findall(r"^###\s+(.+)$", content_lower))
    
    structure["h1"] = h1_count
    structure["h2"] = h2_count
    structure["h3"] = h3_count
    
    # Check for missing levels
    missing = []
    if h1_count == 0:
        missing.append("H1 heading")
    if h2_count == 0:
        missing.append("H2 headings")
    if h3_count == 0:
        missing.append("H3 subsections")
    
    return {
        "structure": structure,
        "missing": missing,
        "score": 1.0 - (len(missing) * 0.2),
    }


def calculate_readability_score(
    content: str,
) -> float:
    """
    Calculate readability score using Flesch-Kincaid formula approximation.
    """
    words = content.split()
    if not words:
        return 0.0
    
    # Calculate average sentence length
    sentences = re.split(r'[.!?]+', content)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    # Approximate Flesch-Kincaid grade
    if avg_sentence_length < 15:
        return 90.0
    elif avg_sentence_length < 20:
        return 85.0
    elif avg_sentence_length < 25:
        return 75.0
    else:
        return 65.0


def analyze_seo(
    content: str,
    config: SEOOptimizer,
) -> SEOScore:
    """
    Analyze content for SEO.
    """
    
    # Keyword analysis
    keywords = [config.primary_keyword] + config.secondary_keywords
    
    keyword_analyses = []
    for kw in keywords:
        density = analyze_keyword_density(content, kw)
        placement = analyze_keyword_placement(content, kw)
        
        keyword_analyses.append(KeywordAnalysis(
            keyword=kw,
            count=placement["positions"]["score"],
            density=density,
            positions=placement["positions"],
            naturally_placed=placement["naturally_placed"],
        ))
    
    # Heading structure
    heading_analysis = analyze_heading_structure(content)
    
    # Readability
    readability = calculate_readability_score(content)
    
    # Calculate overall score
    weights = {
        "keyword_density": 0.3,
        "keyword_placement": 0.3,
        "heading_structure": 0.2,
        "readability": 0.2,
    }
    
    # Weighted scores
    total_score = 0.0
    for kw_analysis in keyword_analyses:
        total_score += kw_analysis.density * weights["keyword_density"]
        total_score += kw_analysis.positions["score"] * weights["keyword_placement"]
    
    total_score += heading_analysis["score"] * weights["heading_structure"]
    total_score += readability * weights["readability"]
    
    # Normalize to 0-100
    total_score = min(100, total_score)
    
    # Determine grade
    if total_score >= 90:
        grade = "A"
    elif total_score >= 80:
        grade = "B"
    elif total_score >= 70:
        grade = "C"
    else:
        grade = "D"
    
    # Generate recommendations
    recommendations = []
    
    # Check primary keyword issues
    primary_kw = config.primary_keyword.lower()
    primary_analysis = next((ka for ka in keyword_analyses if ka.keyword == primary_kw), None)
    
    if primary_analysis:
        if not primary_analysis.positions["title"]:
            recommendations.append({
                "type": "keyword-placement",
                "priority": "critical",
                "message": f"Primary keyword '{config.primary_keyword}' not in title",
                "auto_fix": True,
            })
        
        if not primary_analysis.positions["first_paragraph"]:
            recommendations.append({
                "type": "keyword-placement",
                "priority": "critical",
                "message": f"Primary keyword '{config.primary_keyword}' not in first paragraph",
                "auto_fix": True,
            })
    
    # Check density
    for kw_analysis in keyword_analyses:
        if kw_analysis.density < config.target_density * 0.8:
            recommendations.append({
                "type": "keyword-density",
                "priority": "high",
                "message": f"Keyword '{kw_analysis.keyword}' density is {kw_analysis.density:.2f}%. Target: {config.target_density:.2f}%",
                "fix": f"Add {max(1, int((config.target_density - kw_analysis.density) * len(content.split()) * 100))} more mentions of '{kw_analysis.keyword}'",
            })
        elif kw_analysis.density > config.target_density * 1.5:
            recommendations.append({
                "type": "keyword-density",
                "priority": "medium",
                "message": f"Keyword '{kw_analysis.keyword}' density is {kw_analysis.density:.2f}%. Target: {config.target_density:.2f}%",
            })
    
    # Check heading structure
    if heading_analysis["missing"]:
        recommendations.append({
            "type": "heading-structure",
            "priority": "high",
            "message": f"Missing: {', '.join(heading_analysis['missing'])}",
        })
    
    # Check readability
    if readability < 70:
        recommendations.append({
            "type": "readability",
            "priority": "medium",
            "message": f"Readability score is {readability:.1f}%. Consider shorter sentences",
            })
    
    return SEOScore(
        overall=round(total_score, 1),
        grade=grade,
        keyword_density=round(total_score / len(keyword_analyses) * 100, 2) if keyword_analyses else 0,
        readability=readability,
        heading_structure=heading_analysis["structure"],
        recommendations=recommendations,
    )
