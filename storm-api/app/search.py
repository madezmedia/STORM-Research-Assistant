"""
Web Search and Local Data Collection

Handles web search via Google/Bing APIs and local data collection
from business directories, Chamber of Commerce data, and city statistics.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import httpx
import json


class SearchResult(BaseModel):
    """Search result"""
    url: str
    title: str
    snippet: str
    content: str
    source: str
    credibility_score: float = Field(default=0.0, ge=0.0, le=1.0)


class LocalBusiness(BaseModel):
    """Local business data"""
    name: str
    address: str
    phone: str
    website: str
    rating: Optional[float] = None
    reviews_count: int
    categories: List[str]


class LocalLandmark(BaseModel):
    """Local landmark data"""
    name: str
    type: str
    address: str
    description: str


class LocalMediaOutlet(BaseModel):
    """Local media outlet data"""
    name: str
    type: str  # radio, TV, newspaper
    url: str
    audience: Optional[int] = None  # estimated listenership


class CityStatistics(BaseModel):
    """City statistics"""
    city: str
    state: str
    population: int
    business_count: int
    median_income: Optional[int] = None


class ResearchDataResponse(BaseModel):
    """Research data response"""
    web_results: List[SearchResult]
    local_businesses: List[LocalBusiness]
    landmarks: List[LocalLandmark]
    media_outlets: List[LocalMediaOutlet]
    city_stats: Optional[CityStatistics]


class WebSearchConfig(BaseModel):
    """Web search configuration"""
    provider: str = Field(default="google", description="Search provider: google, bing")
    max_results: int = Field(default=10, ge=1, le=20)
    include_local_results: bool = Field(default=True)
    credibility_threshold: float = Field(default=0.5)


class LocalDataConfig(BaseModel):
    """Local data configuration"""
    enabled: bool = Field(default=False)
    city: str
    state: str


# Web Search Clients
class WebSearchClient:
    """Base class for web search clients"""
    
    async def search(self, query: str, config: WebSearchConfig) -> List[SearchResult]:
        raise NotImplementedError("Subclasses must implement search()")
    
    def calculate_credibility(self, result: SearchResult) -> float:
        """Calculate credibility score for a search result"""
        # Base score on source
        source_scores = {
            "google.com": 0.9,
            "wikipedia.org": 0.95,
            "scholar.google.com": 0.85,
            "news.google.com": 0.8,
            "gov": 0.9,
            "edu": 0.8,
            "org": 0.7,
        }
        
        base_score = source_scores.get(result.source, 0.7)
        
        # Adjust based on content quality
        if len(result.content) < 200:
            base_score *= 0.8
        elif len(result.content) < 500:
            base_score *= 0.9
        else:
            base_score *= 1.0
        
        # Adjust based on snippet quality
        if len(result.snippet) < 100:
            base_score *= 0.9
        elif len(result.snippet) < 200:
            base_score *= 0.95
        else:
            base_score *= 1.0
        
        return round(base_score, 2)


class GoogleSearchClient(WebSearchClient):
    """Google Custom Search API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, config: WebSearchConfig) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}?key={self.api_key}&cx={config.max_results}"
            params = {
                "q": query,
                "num": config.max_results,
            }
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = response.json()
                
                results: List[SearchResult] = []
                
                for item in data.get("items", []):
                    if "pagemap" in item:
                        continue
                    
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    
                    # Extract content
                    content = ""
                    if "htmlSnippet" in item:
                        content = item["htmlSnippet"]
                    elif "snippet" in item:
                        content = item.get("snippet", "")
                    
                    # Get source
                    source = "google.com"
                    if "link" in item:
                        try:
                            parsed = httpx.URL(link)
                            source = parsed.netloc
                        except:
                            pass
                    
                    # Calculate credibility
                    client = self()
                    results.append(SearchResult(
                        url=link,
                        title=title,
                        snippet=snippet,
                        content=content,
                        source=source,
                        credibility_score=client.calculate_credibility(SearchResult(
                            url=link,
                            title=title,
                            snippet=snippet,
                            content=content,
                            source=source,
                        ))
                    ))

                return results


class BingSearchClient(WebSearchClient):
    """Bing Search API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
    
    async def search(self, query: str, config: WebSearchConfig) -> List[SearchResult]:
        """Search using Bing Search API"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}?q={query}"
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []
                
                data = response.json()
                
                results: List[SearchResult] = []
                
                for item in data.get("webPages", {}).get("value", [])[:config.max_results]:
                    title = item.get("name", "")
                    snippet = item.get("snippet", "")
                    link = item.get("url", "")
                    
                    # Extract content
                    content = ""
                    if "description" in item:
                        content = item.get("description", "")
                    
                    # Get source
                    source = "bing.com"
                    
                    # Calculate credibility
                    client = self()
                    results.append(SearchResult(
                        url=link,
                        title=title,
                        snippet=snippet,
                        content=content,
                        source=source,
                        credibility_score=client.calculate_credibility(SearchResult(
                            url=link,
                            title=title,
                            snippet=snippet,
                            content=content,
                            source=source,
                        ))
                    ))

                return results


class LocalDataCollector:
    """Collects local business data, landmarks, and statistics"""
    
    def __init__(self):
        self.chamber_api_url = "https://api.chamberofcommerce.org/v2"
        self.census_api_url = "https://api.census.gov/data/2021/pep/pop"
    
    async def get_local_businesses(
        self,
        city: str,
        state: str,
        business_type: str = "restaurant",
        limit: int = 10,
    ) -> List[LocalBusiness]:
        """Get local businesses from Chamber of Commerce API"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.chamber_api_url}/search"
            params = {
                "api_key": "YOUR_CHAMBER_API_KEY",
                "q": f"{business_type} {city} {state}",
                "limit": limit,
            }
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = response.json()
                
                businesses: List[LocalBusiness] = []
                
                for item in data.get("results", {}).get("results", [])[:limit]:
                    name = item.get("name", "")
                    address = item.get("address", "")
                    phone = item.get("phone", "")
                    website = item.get("website", "")
                    
                    businesses.append(LocalBusiness(
                        name=name,
                        address=address,
                        phone=phone,
                        website=website,
                    ))
                
                return businesses
    
    async def get_landmarks(
        self,
        city: str,
        state: str,
        limit: int = 5,
    ) -> List[LocalLandmark]:
        """Get local landmarks from Google Places API"""
        # TODO: Implement Google Places API
        return []
    
    async def get_media_outlets(
        self,
        city: str,
        state: str,
        limit: int = 5,
    ) -> List[LocalMediaOutlet]:
        """Get local media outlets"""
        # TODO: Implement media outlet scraping
        return []
    
    async def get_city_statistics(
        self,
        city: str,
        state: str,
    ) -> Optional[CityStatistics]:
        """Get city statistics from Census API"""
        # TODO: Implement Census API integration
        return None


async def collect_local_data(
    config: LocalDataConfig,
) -> ResearchDataResponse:
    """Collect all local data based on configuration"""
    
    web_results: List[SearchResult] = []
    local_businesses: List[LocalBusiness] = []
    landmarks: List[LocalLandmark] = []
    media_outlets: List[LocalMediaOutlet] = []
    city_stats: Optional[CityStatistics] = None
    
    if config.enabled:
        collector = LocalDataCollector()
        
        # Collect local businesses
        local_businesses = await collector.get_local_businesses(
            city=config.city,
            state=config.state,
        )
        
        # Collect landmarks
        landmarks = await collector.get_landmarks(
            city=config.city,
            state=config.state,
        )
        
        # Collect media outlets
        media_outlets = await collector.get_media_outlets(
            city=config.city,
            state=config.state,
        )
        
        # Collect city statistics
        city_stats = await collector.get_city_statistics(
            city=config.city,
            state=config.state,
        )
    
    return ResearchDataResponse(
        web_results=web_results,
        local_businesses=local_businesses,
        landmarks=landmarks,
        media_outlets=media_outlets,
        city_stats=city_stats,
    )


def get_search_client(provider: str, api_key: str) -> WebSearchClient:
    """Get search client based on provider"""
    if provider == "google":
        return GoogleSearchClient(api_key)
    elif provider == "bing":
        return BingSearchClient(api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
