"""
GEO Enhancer

Enhances content with local targeting including:
- Natural location mentions
- Local landmark references
- Local business citations
- Regional statistics integration
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class GEOEnhancer(BaseModel):
    """GEO enhancer configuration"""
    content: str
    location: str
    local_keywords: List[str]
    target_mentions: int = 3


class GEOEnhancement(BaseModel):
    """GEO enhancement result"""
    location_mentions: int
    local_keyword_usage: List[str]
    geo_score: float
    enhancements: List[str]


def enhance_content(
    content: str,
    config: GEOEnhancer,
) -> GEOEnhancement:
    """
    Enhance content with GEO targeting.
    """
    
    content_lower = content.lower()
    location_lower = config.location.lower()
    local_keywords_lower = [kw.lower() for kw in config.local_keywords]
    
    # Count location mentions
    location_mentions = content_lower.count(location_lower)
    
    # Count local keyword usage
    keyword_usage = {}
    for kw in local_keywords:
        count = content_lower.count(kw.lower())
        if count > 0:
            keyword_usage[kw] = count
    
    # Calculate GEO score
    # Target: 3 mentions, all keywords used
    target_mentions_score = min(100, (location_mentions / config.target_mentions) * 50)
    
    keyword_usage_score = 0
    for kw, count in keyword_usage.items():
        if count > 0:
            keyword_usage_score += 25
        else:
            keyword_usage_score += 0
    
    keyword_usage_score = min(100, keyword_usage_score)
    
    # Calculate overall GEO score
    geo_score = (target_mentions_score + keyword_usage_score) / 2
    
    # Generate enhancements
    enhancements = []
    
    # Check location mentions
    if location_mentions == 0:
        enhancements.append({
            "type": "geo-location",
            "priority": "high",
            "message": f"Only {location_mentions} mentions of '{config.location}'. Target: {config.target_mentions}",
            "suggestion": f"Add to an H2 heading: \"How {config.location} Businesses...\"",
        })
    elif location_mentions < config.target_mentions:
        enhancements.append({
            "type": "geo-location",
            "priority": "medium",
            "message": f"Only {location_mentions} mentions of '{config.location}'. Target: {config.target_mentions}",
            "suggestion": f"Add to an H2 heading: \"How {config.location} Businesses...\"",
        })
    
    # Check local keyword usage
    unused_keywords = [kw for kw, count in keyword_usage.items() if count == 0]
    if unused_keywords:
        enhancements.append({
            "type": "geo-keywords",
            "priority": "medium",
            "message": f"Some local keywords not used: {', '.join(unused_keywords)}",
        })
    
    # Suggest landmark integration
    if location_mentions > 0:
        enhancements.append({
            "type": "geo-landmarks",
            "priority": "medium",
            "message": f"Consider adding local landmarks to support '{config.location}' mentions",
        })
    
    # Suggest business citations
    if location_mentions > 0:
        enhancements.append({
            "type": "geo-businesses",
            "priority": "low",
            "message": f"Consider citing local {config.location} businesses for credibility",
        })
    
    return GEOEnhancement(
        location_mentions=location_mentions,
        local_keyword_usage=keyword_usage,
        geo_score=geo_score,
        enhancements=enhancements,
    )


def suggest_location_integrations(
    content: str,
    location: str,
) -> List[str]:
    """
    Suggest natural ways to integrate location mentions.
    """
    suggestions = []
    
    # Common patterns
    patterns = [
        f"located in {location}",
        f"serving the {location} area",
        f"based in {location}",
        f"{location}' businesses",
    ]
    
    for pattern in patterns:
        suggestions.append(pattern)
    
    return suggestions


def suggest_landmark_references(
    location: str,
) -> List[str]:
    """
    Suggest local landmarks to reference.
    """
    # TODO: Implement actual landmark database lookup
    # For now, return generic suggestions
    return [
        f"{location} Convention Center",
        f"{location} City Hall",
        f"{location} Museum",
        f"{location} Public Library",
    ]
