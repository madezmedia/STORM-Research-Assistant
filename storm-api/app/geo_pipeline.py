"""
GEO Pipeline Module

This module implements the Generative Engine Optimization (GEO) pipeline for
analyzing websites, extracting keywords, generating LLM monitoring prompts,
and tracking brand mentions across AI responses.

Components:
- NicheAnalyzer: KeyBERT + BERTopic for keyword/topic extraction
- GoogleKeywordResearch: Autocomplete + PAA questions
- PromptGenerator: Generate LLM monitoring prompts
- KeywordTracker: Track keyword mentions in LLM responses
- GEOPipeline: Main orchestrator
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from urllib.parse import urlparse, quote_plus
import httpx
from bs4 import BeautifulSoup


@dataclass
class NicheAnalysis:
    """Results from niche analysis"""
    keywords: list[tuple[str, float]] = field(default_factory=list)  # (keyword, score)
    topics: list[str] = field(default_factory=list)
    brand_mentions: list[str] = field(default_factory=list)
    content_summary: str = ""
    domain: str = ""
    pages_analyzed: int = 0


@dataclass
class KeywordResearch:
    """Results from keyword research"""
    autocomplete: list[str] = field(default_factory=list)
    paa_questions: list[str] = field(default_factory=list)
    related_searches: list[str] = field(default_factory=list)


@dataclass
class GEOPrompt:
    """A prompt for LLM monitoring"""
    prompt_text: str
    category: str  # "direct_mention", "comparison", "recommendation", "feature_inquiry"
    target_keywords: list[str] = field(default_factory=list)
    expected_mention: str = ""


@dataclass
class MentionAnalysis:
    """Analysis of keyword mentions in LLM response"""
    keyword: str
    count: int
    positions: list[int] = field(default_factory=list)
    context_snippets: list[str] = field(default_factory=list)


class NicheAnalyzer:
    """
    Analyzes website content to extract keywords and topics.

    Uses KeyBERT for keyword extraction and optionally BERTopic for topic modeling.
    Falls back to TF-IDF based extraction if KeyBERT is not available.
    """

    def __init__(self, use_keybert: bool = True, use_bertopic: bool = False):
        self.use_keybert = use_keybert
        self.use_bertopic = use_bertopic
        self.kw_model = None
        self.topic_model = None
        self._init_models()

    def _init_models(self):
        """Initialize NLP models"""
        if self.use_keybert:
            try:
                from keybert import KeyBERT
                self.kw_model = KeyBERT()
            except ImportError:
                print("KeyBERT not available, falling back to simple extraction")
                self.use_keybert = False

        if self.use_bertopic:
            try:
                from bertopic import BERTopic
                self.topic_model = BERTopic()
            except ImportError:
                print("BERTopic not available, skipping topic modeling")
                self.use_bertopic = False

    def extract_keywords(
        self,
        text: str,
        top_n: int = 20,
        keyphrase_ngram_range: tuple[int, int] = (1, 3)
    ) -> list[tuple[str, float]]:
        """
        Extract keywords from text.

        Args:
            text: Input text to analyze
            top_n: Number of keywords to extract
            keyphrase_ngram_range: Range of n-grams for keyphrases

        Returns:
            List of (keyword, score) tuples
        """
        if not text or len(text.strip()) < 50:
            return []

        if self.use_keybert and self.kw_model:
            try:
                keywords = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=keyphrase_ngram_range,
                    stop_words='english',
                    top_n=top_n,
                    use_mmr=True,  # Maximal Marginal Relevance for diversity
                    diversity=0.5
                )
                return keywords
            except Exception as e:
                print(f"KeyBERT extraction failed: {e}")

        # Fallback to simple word frequency
        return self._simple_keyword_extraction(text, top_n)

    def _simple_keyword_extraction(self, text: str, top_n: int) -> list[tuple[str, float]]:
        """Simple TF-IDF-like keyword extraction fallback"""
        # Common English stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'dare', 'ought', 'used', 'that', 'which', 'who', 'whom', 'whose',
            'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'i', 'me', 'my', 'not', 'no', 'nor', 'so', 'too', 'very', 'just',
            'also', 'now', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'than', 'then', 'what', 'any'
        }

        # Tokenize and count
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq: dict[str, int] = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and normalize
        total = sum(word_freq.values()) or 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [(word, count / total) for word, count in sorted_words[:top_n]]

    def extract_topics(self, documents: list[str], num_topics: int = 5) -> list[str]:
        """
        Extract topics from multiple documents.

        Args:
            documents: List of document texts
            num_topics: Number of topics to extract

        Returns:
            List of topic labels
        """
        if not documents:
            return []

        if self.use_bertopic and self.topic_model:
            try:
                topics, _ = self.topic_model.fit_transform(documents)
                topic_info = self.topic_model.get_topic_info()
                return topic_info['Name'].tolist()[:num_topics]
            except Exception as e:
                print(f"BERTopic extraction failed: {e}")

        # Fallback: extract common themes from keywords
        all_text = " ".join(documents)
        keywords = self.extract_keywords(all_text, top_n=num_topics)
        return [kw[0] for kw in keywords]


class GoogleKeywordResearch:
    """
    Performs keyword research using Google Autocomplete and PAA APIs.

    Note: This uses unofficial APIs and should be used responsibly.
    """

    def __init__(self):
        self.autocomplete_url = "https://suggestqueries.google.com/complete/search"
        self.paa_patterns = [
            "what is {}", "how to {}", "why is {}", "when to {}",
            "best {} for", "{} vs", "{} benefits", "{} examples",
            "how does {} work", "{} guide", "{} tips", "{} review"
        ]

    async def get_autocomplete(
        self,
        query: str,
        language: str = "en",
        country: str = "us"
    ) -> list[str]:
        """
        Get Google autocomplete suggestions.

        Args:
            query: Search query seed
            language: Language code
            country: Country code

        Returns:
            List of autocomplete suggestions
        """
        params = {
            "client": "firefox",
            "q": query,
            "hl": language,
            "gl": country
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.autocomplete_url, params=params, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1:
                        return data[1][:10]  # Return top 10 suggestions
        except Exception as e:
            print(f"Autocomplete request failed: {e}")

        return []

    async def get_paa_questions(self, topic: str) -> list[str]:
        """
        Generate PAA-style questions for a topic.

        This generates questions based on common patterns rather than
        scraping Google directly.

        Args:
            topic: Main topic to generate questions for

        Returns:
            List of question strings
        """
        questions = []
        for pattern in self.paa_patterns:
            question = pattern.format(topic)
            questions.append(question)

        # Also get autocomplete suggestions that are questions
        autocomplete = await self.get_autocomplete(f"{topic} how")
        questions.extend([s for s in autocomplete if "?" in s or s.startswith("how")])

        autocomplete = await self.get_autocomplete(f"{topic} what")
        questions.extend([s for s in autocomplete if "?" in s or s.startswith("what")])

        return list(set(questions))[:20]

    async def get_related_searches(self, topic: str) -> list[str]:
        """
        Get related search suggestions.

        Args:
            topic: Main topic

        Returns:
            List of related search terms
        """
        related = []

        # Get variations
        suffixes = ["", " tools", " software", " services", " companies", " alternatives"]
        for suffix in suffixes:
            suggestions = await self.get_autocomplete(f"{topic}{suffix}")
            related.extend(suggestions)

        return list(set(related))[:20]


class PromptGenerator:
    """
    Generates prompts for LLM monitoring to track brand visibility.

    Creates prompts across different categories:
    - Direct mention: "What is [brand]?"
    - Comparison: "[brand] vs [competitor]"
    - Recommendation: "Best [category] tools/services"
    - Feature inquiry: "Does [brand] offer [feature]?"
    """

    def __init__(self, brand_name: str, brand_info: Optional[dict[str, Any]] = None):
        self.brand_name = brand_name
        self.brand_info = brand_info or {}

    def generate_prompts(
        self,
        keywords: list[str],
        questions: list[str],
        competitors: Optional[list[str]] = None
    ) -> list[GEOPrompt]:
        """
        Generate comprehensive prompt set for brand monitoring.

        Args:
            keywords: Target keywords from niche analysis
            questions: PAA questions from keyword research
            competitors: Known competitor names

        Returns:
            List of GEOPrompt objects
        """
        prompts = []
        competitors = competitors or []

        # Direct mention prompts
        prompts.extend(self._generate_direct_mentions(keywords))

        # Comparison prompts
        prompts.extend(self._generate_comparisons(competitors, keywords))

        # Recommendation prompts
        prompts.extend(self._generate_recommendations(keywords))

        # Feature inquiry prompts
        prompts.extend(self._generate_feature_inquiries(keywords))

        # Question-based prompts
        prompts.extend(self._generate_from_questions(questions))

        return prompts

    def _generate_direct_mentions(self, keywords: list[str]) -> list[GEOPrompt]:
        """Generate prompts asking directly about the brand"""
        prompts = []

        templates = [
            f"What is {self.brand_name}?",
            f"Tell me about {self.brand_name}",
            f"What does {self.brand_name} do?",
            f"Who is {self.brand_name}?",
            f"What services does {self.brand_name} offer?",
        ]

        for template in templates:
            prompts.append(GEOPrompt(
                prompt_text=template,
                category="direct_mention",
                target_keywords=[self.brand_name.lower()],
                expected_mention=self.brand_name
            ))

        # Add keyword-specific direct mentions
        for kw in keywords[:5]:
            prompts.append(GEOPrompt(
                prompt_text=f"What is {self.brand_name}'s approach to {kw}?",
                category="direct_mention",
                target_keywords=[self.brand_name.lower(), kw.lower()],
                expected_mention=self.brand_name
            ))

        return prompts

    def _generate_comparisons(
        self,
        competitors: list[str],
        keywords: list[str]
    ) -> list[GEOPrompt]:
        """Generate comparison prompts"""
        prompts = []

        for competitor in competitors[:5]:
            prompts.append(GEOPrompt(
                prompt_text=f"Compare {self.brand_name} vs {competitor}",
                category="comparison",
                target_keywords=[self.brand_name.lower(), competitor.lower()],
                expected_mention=self.brand_name
            ))

            # Add keyword-specific comparisons
            for kw in keywords[:3]:
                prompts.append(GEOPrompt(
                    prompt_text=f"Which is better for {kw}: {self.brand_name} or {competitor}?",
                    category="comparison",
                    target_keywords=[self.brand_name.lower(), competitor.lower(), kw.lower()],
                    expected_mention=self.brand_name
                ))

        return prompts

    def _generate_recommendations(self, keywords: list[str]) -> list[GEOPrompt]:
        """Generate recommendation-seeking prompts"""
        prompts = []

        templates = [
            "What are the best {} tools?",
            "Recommend a {} service",
            "What {} company should I use?",
            "Top {} providers in 2024",
            "Best {} for small businesses",
        ]

        for kw in keywords[:5]:
            for template in templates:
                prompts.append(GEOPrompt(
                    prompt_text=template.format(kw),
                    category="recommendation",
                    target_keywords=[kw.lower(), self.brand_name.lower()],
                    expected_mention=self.brand_name
                ))

        return prompts

    def _generate_feature_inquiries(self, keywords: list[str]) -> list[GEOPrompt]:
        """Generate prompts asking about specific features"""
        prompts = []

        templates = [
            f"Does {self.brand_name} offer {{}}?",
            f"Can {self.brand_name} help with {{}}?",
            f"What {self.brand_name} features help with {{}}?",
        ]

        for kw in keywords[:5]:
            for template in templates:
                prompts.append(GEOPrompt(
                    prompt_text=template.format(kw),
                    category="feature_inquiry",
                    target_keywords=[self.brand_name.lower(), kw.lower()],
                    expected_mention=self.brand_name
                ))

        return prompts

    def _generate_from_questions(self, questions: list[str]) -> list[GEOPrompt]:
        """Generate prompts from PAA questions"""
        prompts = []

        for question in questions[:10]:
            # Add the brand name context
            prompt_text = f"{question} (considering options like {self.brand_name})"
            prompts.append(GEOPrompt(
                prompt_text=prompt_text,
                category="recommendation",
                target_keywords=[self.brand_name.lower()],
                expected_mention=self.brand_name
            ))

        return prompts

    def export_gego_format(self, prompts: list[GEOPrompt]) -> list[dict[str, Any]]:
        """
        Export prompts in GEGO-compatible JSON format.

        Args:
            prompts: List of GEOPrompt objects

        Returns:
            List of dicts in GEGO format
        """
        return [
            {
                "prompt": p.prompt_text,
                "category": p.category,
                "target_keywords": p.target_keywords,
                "expected_mention": p.expected_mention,
                "brand": self.brand_name
            }
            for p in prompts
        ]


class KeywordTracker:
    """
    Tracks keyword and brand mentions in LLM responses.

    Analyzes response text for:
    - Direct brand mentions
    - Keyword occurrences
    - Sentiment context
    - Position tracking
    """

    def __init__(self, brand_name: str, keywords: list[str]):
        self.brand_name = brand_name
        self.keywords = keywords

    def analyze_response(
        self,
        response_text: str,
        prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Analyze LLM response for brand and keyword mentions.

        Args:
            response_text: The LLM response text
            prompt: Original prompt (for context)

        Returns:
            Dictionary with analysis results
        """
        results = {
            "brand_mentioned": False,
            "brand_mentions": self._find_mentions(response_text, self.brand_name),
            "keyword_mentions": {},
            "total_mentions": 0,
            "sentiment_context": [],
            "position_score": 0.0  # Higher = mentioned earlier
        }

        # Check brand mentions
        brand_analysis = self._analyze_keyword(response_text, self.brand_name)
        results["brand_mentioned"] = brand_analysis.count > 0
        results["brand_mentions"] = {
            "count": brand_analysis.count,
            "positions": brand_analysis.positions,
            "contexts": brand_analysis.context_snippets
        }

        # Check keyword mentions
        for keyword in self.keywords:
            kw_analysis = self._analyze_keyword(response_text, keyword)
            if kw_analysis.count > 0:
                results["keyword_mentions"][keyword] = {
                    "count": kw_analysis.count,
                    "positions": kw_analysis.positions
                }

        # Calculate total mentions
        results["total_mentions"] = (
            results["brand_mentions"]["count"] +
            sum(m["count"] for m in results["keyword_mentions"].values())
        )

        # Calculate position score (0-1, higher = mentioned earlier)
        if results["brand_mentioned"]:
            first_pos = results["brand_mentions"]["positions"][0]
            text_len = len(response_text)
            results["position_score"] = 1.0 - (first_pos / text_len) if text_len > 0 else 0

        return results

    def _analyze_keyword(self, text: str, keyword: str) -> MentionAnalysis:
        """Analyze a single keyword in text"""
        positions = []
        contexts = []

        # Case-insensitive search
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for match in pattern.finditer(text):
            positions.append(match.start())

            # Extract context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            contexts.append(f"...{context}...")

        return MentionAnalysis(
            keyword=keyword,
            count=len(positions),
            positions=positions,
            context_snippets=contexts[:5]  # Limit to 5 snippets
        )

    def _find_mentions(self, text: str, term: str) -> list[dict[str, Any]]:
        """Find all mentions of a term with context"""
        mentions = []
        pattern = re.compile(re.escape(term), re.IGNORECASE)

        for match in pattern.finditer(text):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            mentions.append({
                "position": match.start(),
                "context": text[start:end]
            })

        return mentions


class WebsiteCrawler:
    """
    Crawls website content for analysis.

    Supports Firecrawl API for advanced crawling or falls back to simple HTTP fetch.
    """

    def __init__(self, firecrawl_api_key: Optional[str] = None):
        self.firecrawl_api_key = firecrawl_api_key
        self.use_firecrawl = firecrawl_api_key is not None

    async def crawl(
        self,
        url: str,
        max_pages: int = 10
    ) -> list[dict[str, Any]]:
        """
        Crawl a website and extract content.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl

        Returns:
            List of page data dictionaries
        """
        if self.use_firecrawl:
            return await self._firecrawl_crawl(url, max_pages)
        return await self._simple_crawl(url, max_pages)

    async def _firecrawl_crawl(
        self,
        url: str,
        max_pages: int
    ) -> list[dict[str, Any]]:
        """Crawl using Firecrawl API"""
        try:
            from firecrawl import FirecrawlApp

            app = FirecrawlApp(api_key=self.firecrawl_api_key)
            result = app.crawl_url(
                url,
                params={
                    "limit": max_pages,
                    "scrapeOptions": {
                        "formats": ["markdown", "html"]
                    }
                }
            )

            pages = []
            for page in result.get("data", []):
                pages.append({
                    "url": page.get("metadata", {}).get("sourceURL", url),
                    "title": page.get("metadata", {}).get("title", ""),
                    "content": page.get("markdown", ""),
                    "html": page.get("html", "")
                })

            return pages

        except Exception as e:
            print(f"Firecrawl failed: {e}, falling back to simple crawl")
            return await self._simple_crawl(url, max_pages)

    async def _simple_crawl(
        self,
        url: str,
        max_pages: int
    ) -> list[dict[str, Any]]:
        """Simple HTTP-based crawl"""
        pages = []
        visited = set()
        to_visit = [url]
        domain = urlparse(url).netloc

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            while to_visit and len(pages) < max_pages:
                current_url = to_visit.pop(0)
                if current_url in visited:
                    continue

                visited.add(current_url)

                try:
                    response = await client.get(current_url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Extract title
                        title = soup.title.string if soup.title else ""

                        # Extract main content
                        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                            script.decompose()

                        content = soup.get_text(separator=' ', strip=True)

                        pages.append({
                            "url": current_url,
                            "title": title,
                            "content": content,
                            "html": response.text
                        })

                        # Find more links on same domain
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if href.startswith('/'):
                                href = f"https://{domain}{href}"
                            if urlparse(href).netloc == domain and href not in visited:
                                to_visit.append(href)

                except Exception as e:
                    print(f"Failed to crawl {current_url}: {e}")

        return pages


class GEOPipeline:
    """
    Main orchestrator for the GEO (Generative Engine Optimization) pipeline.

    Coordinates:
    1. Website crawling and content extraction
    2. Keyword and topic extraction
    3. Google keyword research
    4. Prompt generation
    5. Response tracking
    """

    def __init__(
        self,
        brand_name: str,
        firecrawl_api_key: Optional[str] = None,
        use_keybert: bool = True,
        use_bertopic: bool = False
    ):
        self.brand_name = brand_name
        self.crawler = WebsiteCrawler(firecrawl_api_key)
        self.niche_analyzer = NicheAnalyzer(use_keybert=use_keybert, use_bertopic=use_bertopic)
        self.keyword_research = GoogleKeywordResearch()
        self.prompt_generator = PromptGenerator(brand_name)
        self.keyword_tracker: Optional[KeywordTracker] = None

    async def analyze_website(
        self,
        url: str,
        crawl_limit: int = 10
    ) -> NicheAnalysis:
        """
        Analyze a website to extract niche keywords and topics.

        Args:
            url: Website URL to analyze
            crawl_limit: Maximum pages to crawl

        Returns:
            NicheAnalysis with keywords, topics, and summary
        """
        # Crawl the website
        pages = await self.crawler.crawl(url, max_pages=crawl_limit)

        if not pages:
            return NicheAnalysis(domain=urlparse(url).netloc)

        # Combine all content
        all_content = " ".join(p["content"] for p in pages)

        # Extract keywords
        keywords = self.niche_analyzer.extract_keywords(all_content, top_n=30)

        # Extract topics
        documents = [p["content"] for p in pages if len(p["content"]) > 100]
        topics = self.niche_analyzer.extract_topics(documents, num_topics=10)

        # Find brand mentions
        brand_mentions = []
        for page in pages:
            if self.brand_name.lower() in page["content"].lower():
                brand_mentions.append(page["url"])

        # Create summary
        content_summary = all_content[:500] + "..." if len(all_content) > 500 else all_content

        return NicheAnalysis(
            keywords=keywords,
            topics=topics,
            brand_mentions=brand_mentions,
            content_summary=content_summary,
            domain=urlparse(url).netloc,
            pages_analyzed=len(pages)
        )

    async def research_keywords(
        self,
        seed_keywords: list[str]
    ) -> KeywordResearch:
        """
        Perform keyword research based on seed keywords.

        Args:
            seed_keywords: Initial keywords from niche analysis

        Returns:
            KeywordResearch with autocomplete, PAA, and related searches
        """
        all_autocomplete = []
        all_paa = []
        all_related = []

        # Research top seed keywords
        for keyword in seed_keywords[:5]:
            autocomplete = await self.keyword_research.get_autocomplete(keyword)
            all_autocomplete.extend(autocomplete)

            paa = await self.keyword_research.get_paa_questions(keyword)
            all_paa.extend(paa)

            related = await self.keyword_research.get_related_searches(keyword)
            all_related.extend(related)

        return KeywordResearch(
            autocomplete=list(set(all_autocomplete))[:30],
            paa_questions=list(set(all_paa))[:30],
            related_searches=list(set(all_related))[:30]
        )

    def generate_prompts(
        self,
        keywords: list[str],
        questions: list[str],
        competitors: Optional[list[str]] = None
    ) -> list[GEOPrompt]:
        """
        Generate monitoring prompts for LLM brand tracking.

        Args:
            keywords: Target keywords
            questions: PAA questions
            competitors: Known competitor names

        Returns:
            List of GEOPrompt objects
        """
        return self.prompt_generator.generate_prompts(
            keywords=keywords,
            questions=questions,
            competitors=competitors
        )

    def track_response(
        self,
        response_text: str,
        keywords: list[str],
        prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Track brand and keyword mentions in an LLM response.

        Args:
            response_text: LLM response text
            keywords: Keywords to track
            prompt: Original prompt (for context)

        Returns:
            Analysis results dictionary
        """
        tracker = KeywordTracker(self.brand_name, keywords)
        return tracker.analyze_response(response_text, prompt)

    def export_gego_prompts(self, prompts: list[GEOPrompt]) -> list[dict[str, Any]]:
        """
        Export prompts in GEGO-compatible format.

        Args:
            prompts: List of GEOPrompt objects

        Returns:
            JSON-serializable list of prompt dicts
        """
        return self.prompt_generator.export_gego_format(prompts)

    async def run_full_pipeline(
        self,
        url: str,
        crawl_limit: int = 10,
        competitors: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Run the complete GEO pipeline.

        Args:
            url: Website URL to analyze
            crawl_limit: Maximum pages to crawl
            competitors: Known competitor names

        Returns:
            Complete pipeline results
        """
        # Step 1: Analyze website
        niche = await self.analyze_website(url, crawl_limit)

        # Step 2: Research keywords
        seed_keywords = [kw[0] for kw in niche.keywords[:10]]
        research = await self.research_keywords(seed_keywords)

        # Step 3: Generate prompts
        prompts = self.generate_prompts(
            keywords=seed_keywords,
            questions=research.paa_questions,
            competitors=competitors
        )

        return {
            "analysis": {
                "domain": niche.domain,
                "pages_analyzed": niche.pages_analyzed,
                "keywords": niche.keywords,
                "topics": niche.topics,
                "brand_mentions": niche.brand_mentions,
                "content_summary": niche.content_summary
            },
            "research": {
                "autocomplete": research.autocomplete,
                "paa_questions": research.paa_questions,
                "related_searches": research.related_searches
            },
            "prompts": self.export_gego_prompts(prompts),
            "prompt_count": len(prompts)
        }
