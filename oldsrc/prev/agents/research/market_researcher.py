from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
from utils.nlp_utils import TextParser
from utils.data_validation import validate_market_data
import logging
from decimal import Decimal
import re

class MarketResearchAgent(BaseAgent):
    def __init__(self, tools_service: ToolsService, **kwargs):
        super().__init__(**kwargs)
        self.tools_service = tools_service
        self.research_tools = tools_service.get_research_tools()
        self.text_parser = TextParser()
        self.logger = logging.getLogger(__name__)
        self.state = {
            "memory": {},
            "last_update": None,
            "research_cache": {}
        }
    
    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market research tasks."""
        try:
            story_bible = task.get("story_bible", {})
            
            market_report = {
                "novel_id": story_bible.get("novel_id"),
                "research_type": "market",
                "genre_analysis": await self._analyze_genre(story_bible),
                "market_trends": await self._analyze_market_trends(story_bible),
                "competitor_analysis": await self._analyze_competitors(story_bible),
                "target_audience": await self._analyze_target_audience(story_bible),
                "recommendations": [],
                "timestamp": datetime.utcnow().isoformat(),
                "status": "complete"
            }
            
            # Generate recommendations based on all analyses
            market_report["recommendations"] = self._generate_recommendations(market_report)
            
            # Validate report data
            if not validate_market_data(market_report):
                raise ValueError("Market report validation failed")
            
            # Update state
            self.state["last_update"] = datetime.utcnow()
            self.state["memory"]["last_report"] = market_report
            
            return market_report
            
        except Exception as e:
            self.logger.error(f"Error in market research: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_genre(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the genre market and trends."""
        try:
            genre = story_bible.get("genre", "")
            
            analysis = {
                "genre": genre,
                "market_size": None,
                "growth_trend": None,
                "popular_themes": [],
                "genre_conventions": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get genre analysis from document retriever
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=f"genre analysis {genre}",
                filters={"category": "market_research", "topic": "genre_analysis"}
            )
            
            # Extract and process data
            analysis["market_size"] = self._extract_market_size(doc_results)
            analysis["growth_trend"] = self._extract_growth_trend(doc_results)
            analysis["popular_themes"] = self._extract_popular_themes(doc_results)
            analysis["genre_conventions"] = self._extract_genre_conventions(doc_results)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in genre analysis: {str(e)}")
            return {
                "genre": genre,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_market_trends(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market trends."""
        try:
            trends = {
                "current_trends": [],
                "emerging_trends": [],
                "declining_trends": [],
                "trend_confidence": {},
                "trend_impact": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get trend analysis from document retriever
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query="current book market trends",
                filters={"category": "market_research", "topic": "trends"}
            )
            
            # Process and categorize trends
            trends["current_trends"] = self._extract_current_trends(doc_results)
            trends["emerging_trends"] = self._extract_emerging_trends(doc_results)
            trends["declining_trends"] = self._extract_declining_trends(doc_results)
            
            # Calculate confidence and impact scores
            for trend in trends["current_trends"] + trends["emerging_trends"] + trends["declining_trends"]:
                trends["trend_confidence"][trend] = self._calculate_trend_confidence(trend, doc_results)
                trends["trend_impact"][trend] = self._calculate_trend_impact(trend, story_bible)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error in market trends analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_competitors(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competing works and authors."""
        try:
            genre = story_bible.get("genre", "")
            themes = story_bible.get("themes", [])
            
            competitor_analysis = {
                "direct_competitors": [],
                "indirect_competitors": [],
                "competitive_advantages": {},
                "market_gaps": [],
                "success_factors": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get competitor data from document retriever
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=f"successful {genre} books and authors",
                filters={"category": "market_research", "topic": "competitors"}
            )
            
            # Analyze competitors
            competitor_analysis["direct_competitors"] = self._identify_direct_competitors(
                doc_results,
                genre,
                themes
            )
            competitor_analysis["indirect_competitors"] = self._identify_indirect_competitors(
                doc_results,
                genre
            )
            
            # Analyze competitive factors
            for competitor in competitor_analysis["direct_competitors"]:
                competitor_analysis["competitive_advantages"][competitor["id"]] = (
                    self._analyze_competitive_advantages(competitor, story_bible)
                )
                competitor_analysis["success_factors"][competitor["id"]] = (
                    self._analyze_success_factors(competitor)
                )
            
            # Identify market gaps
            competitor_analysis["market_gaps"] = self._identify_market_gaps(
                competitor_analysis,
                story_bible
            )
            
            return competitor_analysis
            
        except Exception as e:
            self.logger.error(f"Error in competitor analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _analyze_target_audience(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze target audience demographics and preferences."""
        try:
            audience_analysis = {
                "primary_demographic": {},
                "secondary_demographics": [],
                "psychographic_profiles": [],
                "reading_preferences": {},
                "purchasing_behavior": {},
                "engagement_channels": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Get audience data from document retriever
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=f"book reader demographics and preferences",
                filters={"category": "market_research", "topic": "audience"}
            )
            
            # Analyze demographics
            audience_analysis["primary_demographic"] = self._identify_primary_demographic(
                doc_results,
                story_bible
            )
            audience_analysis["secondary_demographics"] = self._identify_secondary_demographics(
                doc_results,
                story_bible
            )
            
            # Analyze psychographic profiles
            audience_analysis["psychographic_profiles"] = self._analyze_psychographic_profiles(
                doc_results,
                story_bible
            )
            
            # Analyze preferences and behavior
            audience_analysis["reading_preferences"] = self._analyze_reading_preferences(
                doc_results,
                story_bible
            )
            audience_analysis["purchasing_behavior"] = self._analyze_purchasing_behavior(
                doc_results
            )
            audience_analysis["engagement_channels"] = self._identify_engagement_channels(
                doc_results
            )
            
            return audience_analysis
            
        except Exception as e:
            self.logger.error(f"Error in target audience analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _extract_market_size(self, doc_results: List[Dict[str, Any]]) -> Optional[Decimal]:
        """Extract market size from document results."""
        try:
            for doc in doc_results:
                content = doc.get("content", "")
                # Look for market size mentions with currency values
                matches = re.findall(r'\$?\d+\.?\d*\s*(?:billion|million|trillion)', content, re.IGNORECASE)
                if matches:
                    # Convert first match to decimal
                    value_str = re.sub(r'[^\d.]', '', matches[0])
                    multiplier = 1
                    if 'billion' in matches[0].lower():
                        multiplier = 1_000_000_000
                    elif 'million' in matches[0].lower():
                        multiplier = 1_000_000
                    elif 'trillion' in matches[0].lower():
                        multiplier = 1_000_000_000_000
                    return Decimal(value_str) * multiplier
            return None
        except Exception as e:
            self.logger.error(f"Error extracting market size: {str(e)}")
            return None

    def _extract_growth_trend(self, doc_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract growth trend information from document results."""
        try:
            trend_data = {
                "rate": None,
                "period": None,
                "direction": None,
                "confidence": 0.0
            }
            
            for doc in doc_results:
                content = doc.get("content", "")
                # Look for growth rate mentions
                rate_matches = re.findall(r'(-?\d+\.?\d*)%?\s*(?:growth|decline)', content, re.IGNORECASE)
                if rate_matches:
                    trend_data["rate"] = float(rate_matches[0])
                    trend_data["direction"] = "growth" if trend_data["rate"] > 0 else "decline"
                
                # Look for time period mentions
                period_matches = re.findall(r'(?:in|over|during|for)\s+(?:the\s+)?(\d+)\s+(?:year|month)s?', content, re.IGNORECASE)
                if period_matches:
                    trend_data["period"] = {
                        "value": int(period_matches[0]),
                        "unit": "years" if "year" in content else "months"
                    }
                
                # Calculate confidence based on source reliability
                trend_data["confidence"] = self._calculate_source_confidence(doc)
            
            return trend_data if trend_data["rate"] is not None else None
            
        except Exception as e:
            self.logger.error(f"Error extracting growth trend: {str(e)}")
            return None

    def _extract_popular_themes(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract popular themes from document results."""
        try:
            themes = []
            theme_mentions = {}
            
            for doc in doc_results:
                content = doc.get("content", "")
                sentences = sent_tokenize(content)
                
                for sentence in sentences:
                    # Look for theme mentions
                    if any(keyword in sentence.lower() for keyword in ["theme", "motif", "topic", "subject"]):
                        # Extract themes using NLP
                        extracted_themes = self.text_parser.extract_themes(sentence)
                        for theme in extracted_themes:
                            if theme not in theme_mentions:
                                theme_mentions[theme] = {
                                    "name": theme,
                                    "mentions": 0,
                                    "sentiment": 0.0,
                                    "confidence": 0.0
                                }
                            theme_mentions[theme]["mentions"] += 1
                            theme_mentions[theme]["sentiment"] += self.text_parser.analyze_sentiment(sentence)
            
            # Calculate average sentiment and confidence for each theme
            for theme in theme_mentions.values():
                theme["sentiment"] /= theme["mentions"]
                theme["confidence"] = self._calculate_theme_confidence(theme)
                themes.append(theme)
            
            # Sort by mentions
            themes.sort(key=lambda x: x["mentions"], reverse=True)
            return themes
            
        except Exception as e:
            self.logger.error(f"Error extracting popular themes: {str(e)}")
            return []

    def _extract_genre_conventions(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract genre conventions from document results."""
        try:
            conventions = []
            convention_mentions = {}
            
            for doc in doc_results:
                content = doc.get("content", "")
                sentences = sent_tokenize(content)
                
                for sentence in sentences:
                    # Look for convention mentions
                    if any(keyword in sentence.lower() for keyword in ["convention", "trope", "standard", "expectation"]):
                        # Extract conventions using NLP
                        extracted_conventions = self.text_parser.extract_conventions(sentence)
                        for convention in extracted_conventions:
                            if convention not in convention_mentions:
                                convention_mentions[convention] = {
                                    "name": convention,
                                    "mentions": 0,
                                    "importance": 0.0,
                                    "examples": []
                                }
                            convention_mentions[convention]["mentions"] += 1
                            # Extract examples if present
                            examples = self.text_parser.extract_examples(sentence)
                            if examples:
                                convention_mentions[convention]["examples"].extend(examples)
            
            # Calculate importance scores and prepare final list
            for convention in convention_mentions.values():
                convention["importance"] = self._calculate_convention_importance(convention)
                conventions.append(convention)
            
            # Sort by importance
            conventions.sort(key=lambda x: x["importance"], reverse=True)
            return conventions
            
        except Exception as e:
            self.logger.error(f"Error extracting genre conventions: {str(e)}")
            return []

    def _calculate_source_confidence(self, doc: Dict[str, Any]) -> float:
        """Calculate confidence score for a source."""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on source type
            source_type = doc.get("source_type", "")
            if source_type in ["academic", "industry_report"]:
                confidence += 0.3
            elif source_type in ["news", "blog"]:
                confidence += 0.1
            
            # Adjust based on date
            pub_date = doc.get("publication_date", "")
            if pub_date:
                pub_datetime = datetime.fromisoformat(pub_date)
                age_in_years = (datetime.utcnow() - pub_datetime).days / 365
                if age_in_years <= 1:
                    confidence += 0.2
                elif age_in_years <= 2:
                    confidence += 0.1
            
            # Adjust based on citations
            citations = doc.get("citations", 0)
            if citations > 100:
                confidence += 0.2
            elif citations > 50:
                confidence += 0.1
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating source confidence: {str(e)}")
            return 0.5

    def _calculate_theme_confidence(self, theme: Dict[str, Any]) -> float:
        """Calculate confidence score for a theme."""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on number of mentions
            if theme["mentions"] >= 10:
                confidence += 0.3
            elif theme["mentions"] >= 5:
                confidence += 0.2
            elif theme["mentions"] >= 3:
                confidence += 0.1
            
            # Adjust based on sentiment consistency
            sentiment_std = theme.get("sentiment_std", 0.5)
            if sentiment_std < 0.2:
                confidence += 0.2  # Consistent sentiment across mentions
            elif sentiment_std < 0.4:
                confidence += 0.1
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating theme confidence: {str(e)}")
            return 0.5

    def _calculate_convention_importance(self, convention: Dict[str, Any]) -> float:
        """Calculate importance score for a genre convention."""
        try:
            importance = 0.5  # Base importance
            
            # Adjust based on number of mentions
            if convention["mentions"] >= 15:
                importance += 0.3
            elif convention["mentions"] >= 8:
                importance += 0.2
            elif convention["mentions"] >= 4:
                importance += 0.1
            
            # Adjust based on number of examples
            if len(convention["examples"]) >= 5:
                importance += 0.2
            elif len(convention["examples"]) >= 3:
                importance += 0.1
            
            return min(importance, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating convention importance: {str(e)}")
            return 0.5

    def _extract_current_trends(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract current market trends from document results."""
        try:
            trends = []
            trend_mentions = {}
            
            for doc in doc_results:
                content = doc.get("content", "")
                sentences = sent_tokenize(content)
                
                for sentence in sentences:
                    # Look for current trend mentions
                    if any(keyword in sentence.lower() for keyword in ["current trend", "popular now", "trending"]):
                        extracted_trends = self.text_parser.extract_trends(sentence)
                        for trend in extracted_trends:
                            if trend not in trend_mentions:
                                trend_mentions[trend] = {
                                    "name": trend,
                                    "mentions": 0,
                                    "evidence": [],
                                    "market_impact": None
                                }
                            trend_mentions[trend]["mentions"] += 1
                            trend_mentions[trend]["evidence"].append(sentence)
            
            # Process and analyze each trend
            for trend, data in trend_mentions.items():
                trend_data = {
                    "name": trend,
                    "frequency": data["mentions"],
                    "evidence": data["evidence"],
                    "market_impact": self._analyze_trend_impact(trend, data["evidence"]),
                    "confidence_score": self._calculate_trend_confidence(trend, data)
                }
                trends.append(trend_data)
            
            # Sort by frequency and confidence
            trends.sort(key=lambda x: (x["frequency"], x["confidence_score"]), reverse=True)
            return trends
            
        except Exception as e:
            self.logger.error(f"Error extracting current trends: {str(e)}")
            return []

    def _extract_emerging_trends(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract emerging market trends from document results."""
        try:
            trends = []
            trend_mentions = {}
            
            for doc in doc_results:
                content = doc.get("content", "")
                sentences = sent_tokenize(content)
                
                for sentence in sentences:
                    # Look for emerging trend mentions
                    if any(keyword in sentence.lower() for keyword in ["emerging trend", "growing trend", "rising trend"]):
                        extracted_trends = self.text_parser.extract_trends(sentence)
                        for trend in extracted_trends:
                            if trend not in trend_mentions:
                                trend_mentions[trend] = {
                                    "name": trend,
                                    "mentions": 0,
                                    "evidence": [],
                                    "growth_indicators": []
                                }
                            trend_mentions[trend]["mentions"] += 1
                            trend_mentions[trend]["evidence"].append(sentence)
                            # Extract growth indicators
                            growth_indicators = self.text_parser.extract_growth_indicators(sentence)
                            if growth_indicators:
                                trend_mentions[trend]["growth_indicators"].extend(growth_indicators)
            
            # Process and analyze each trend
            for trend, data in trend_mentions.items():
                trend_data = {
                    "name": trend,
                    "frequency": data["mentions"],
                    "evidence": data["evidence"],
                    "growth_indicators": data["growth_indicators"],
                    "potential_impact": self._analyze_trend_potential(trend, data),
                    "confidence_score": self._calculate_trend_confidence(trend, data)
                }
                trends.append(trend_data)
            
            # Sort by potential impact and confidence
            trends.sort(key=lambda x: (x["potential_impact"], x["confidence_score"]), reverse=True)
            return trends
            
        except Exception as e:
            self.logger.error(f"Error extracting emerging trends: {str(e)}")
            return []

    def _extract_declining_trends(self, doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract declining market trends from document results."""
        try:
            trends = []
            trend_mentions = {}
            
            for doc in doc_results:
                content = doc.get("content", "")
                sentences = sent_tokenize(content)
                
                for sentence in sentences:
                    # Look for declining trend mentions
                    if any(keyword in sentence.lower() for keyword in ["declining trend", "fading trend", "decreasing trend"]):
                        extracted_trends = self.text_parser.extract_trends(sentence)
                        for trend in extracted_trends:
                            if trend not in trend_mentions:
                                trend_mentions[trend] = {
                                    "name": trend,
                                    "mentions": 0,
                                    "evidence": [],
                                    "decline_indicators": []
                                }
                            trend_mentions[trend]["mentions"] += 1
                            trend_mentions[trend]["evidence"].append(sentence)
                            # Extract decline indicators
                            decline_indicators = self.text_parser.extract_decline_indicators(sentence)
                            if decline_indicators:
                                trend_mentions[trend]["decline_indicators"].extend(decline_indicators)
            
            # Process and analyze each trend
            for trend, data in trend_mentions.items():
                trend_data = {
                    "name": trend,
                    "frequency": data["mentions"],
                    "evidence": data["evidence"],
                    "decline_indicators": data["decline_indicators"],
                    "market_impact": self._analyze_decline_impact(trend, data),
                    "confidence_score": self._calculate_trend_confidence(trend, data)
                }
                trends.append(trend_data)
            
            # Sort by market impact and confidence
            trends.sort(key=lambda x: (x["market_impact"], x["confidence_score"]), reverse=True)
            return trends
            
        except Exception as e:
            self.logger.error(f"Error extracting declining trends: {str(e)}")
            return []

    def _calculate_trend_confidence(self, trend: str, data: Dict[str, Any]) -> float:
        """Calculate confidence score for a trend."""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on number of mentions
            if data["mentions"] >= 8:
                confidence += 0.3
            elif data["mentions"] >= 4:
                confidence += 0.2
            elif data["mentions"] >= 2:
                confidence += 0.1
            
            # Adjust based on evidence quality
            evidence_quality = self._assess_evidence_quality(data["evidence"])
            confidence += evidence_quality * 0.2
            
            # Adjust based on source diversity
            source_diversity = self._calculate_source_diversity(data["evidence"])
            confidence += source_diversity * 0.2
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend confidence: {str(e)}")
            return 0.5

    def _analyze_trend_impact(self, trend: str, evidence: List[str]) -> float:
        """Analyze the market impact of a trend."""
        try:
            impact = 0.5  # Base impact
            
            # Analyze sentiment in evidence
            sentiment_scores = [
                self.text_parser.analyze_sentiment(text)
                for text in evidence
            ]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Adjust impact based on sentiment
            impact += avg_sentiment * 0.3
            
            # Look for impact indicators
            impact_indicators = self._extract_impact_indicators(evidence)
            impact += len(impact_indicators) * 0.1
            
            return min(max(impact, 0.0), 1.0)  # Keep between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend impact: {str(e)}")
            return 0.5

    def _analyze_trend_potential(self, trend: str, data: Dict[str, Any]) -> float:
        """Analyze the potential impact of an emerging trend."""
        try:
            potential = 0.5  # Base potential
            
            # Analyze growth indicators
            if data["growth_indicators"]:
                growth_score = sum(
                    self._assess_growth_indicator(indicator)
                    for indicator in data["growth_indicators"]
                ) / len(data["growth_indicators"])
                potential += growth_score * 0.3
            
            # Analyze evidence strength
            evidence_strength = self._assess_evidence_quality(data["evidence"])
            potential += evidence_strength * 0.2
            
            # Analyze market readiness
            market_readiness = self._assess_market_readiness(trend, data["evidence"])
            potential += market_readiness * 0.2
            
            return min(max(potential, 0.0), 1.0)  # Keep between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend potential: {str(e)}")
            return 0.5

    def _analyze_decline_impact(self, trend: str, data: Dict[str, Any]) -> float:
        """Analyze the market impact of a declining trend."""
        try:
            impact = 0.5  # Base impact
            
            # Analyze decline indicators
            if data["decline_indicators"]:
                decline_score = sum(
                    self._assess_decline_indicator(indicator)
                    for indicator in data["decline_indicators"]
                ) / len(data["decline_indicators"])
                impact += decline_score * 0.3
            
            # Analyze evidence strength
            evidence_strength = self._assess_evidence_quality(data["evidence"])
            impact += evidence_strength * 0.2
            
            # Analyze market persistence
            market_persistence = self._assess_market_persistence(trend, data["evidence"])
            impact += market_persistence * 0.2
            
            return min(max(impact, 0.0), 1.0)  # Keep between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error analyzing decline impact: {str(e)}")
            return 0.5

    def _assess_evidence_quality(self, evidence: List[str]) -> float:
        """Assess the quality of evidence."""
        try:
            if not evidence:
                return 0.0
                
            quality_scores = []
            for text in evidence:
                score = 0.5  # Base score
                
                # Check for specific details
                if re.search(r'\d+(?:\.\d+)?%', text):  # Contains percentages
                    score += 0.2
                if re.search(r'\$\d+(?:\.\d+)?', text):  # Contains monetary values
                    score += 0.2
                if re.search(r'\b\d{4}\b', text):  # Contains years
                    score += 0.1
                
                # Check for comparative language
                if re.search(r'\b(?:increase|decrease|growth|decline)\b', text, re.IGNORECASE):
                    score += 0.1
                
                # Check for citations or references
                if re.search(r'\b(?:according to|cited|reported by)\b', text, re.IGNORECASE):
                    score += 0.2
                
                quality_scores.append(min(score, 1.0))
            
            return sum(quality_scores) / len(quality_scores)
            
        except Exception as e:
            self.logger.error(f"Error assessing evidence quality: {str(e)}")
            return 0.5

    def _calculate_source_diversity(self, evidence: List[str]) -> float:
        """Calculate the diversity of sources in evidence."""
        try:
            if not evidence:
                return 0.0
                
            # Extract source mentions
            sources = set()
            for text in evidence:
                # Look for source attributions
                matches = re.findall(
                    r'\b(?:according to|cited by|reported by|says|said)\s+([^,\.]+)',
                    text,
                    re.IGNORECASE
                )
                sources.update(matches)
            
            # Calculate diversity score
            num_sources = len(sources)
            if num_sources >= 5:
                return 1.0
            elif num_sources >= 3:
                return 0.7
            elif num_sources >= 2:
                return 0.5
            elif num_sources >= 1:
                return 0.3
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating source diversity: {str(e)}")
            return 0.0

    def _extract_impact_indicators(self, evidence: List[str]) -> List[str]:
        """Extract impact indicators from evidence."""
        try:
            indicators = []
            
            for text in evidence:
                # Look for impact-related phrases
                matches = re.findall(
                    r'(?:significant|major|substantial|notable)\s+impact\s+on\s+([^,\.]+)',
                    text,
                    re.IGNORECASE
                )
                indicators.extend(matches)
                
                # Look for market effect phrases
                matches = re.findall(
                                        r'(?:affects|influences|changes)\s+(?:the\s+)?market\s+by\s+([^,\.]+)',
                    text,
                    re.IGNORECASE
                )
                indicators.extend(matches)
                
                # Look for quantified effects
                matches = re.findall(
                    r'(?:increases|decreases|changes)\s+(?:by\s+)?(\d+(?:\.\d+)?%)',
                    text,
                    re.IGNORECASE
                )
                indicators.extend([f"quantified effect: {m}" for m in matches])
            
            return list(set(indicators))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error extracting impact indicators: {str(e)}")
            return []

    def _assess_growth_indicator(self, indicator: str) -> float:
        """Assess the strength of a growth indicator."""
        try:
            strength = 0.5  # Base strength
            
            # Check for quantified growth
            if "quantified effect:" in indicator:
                percentage_match = re.search(r'(\d+(?:\.\d+)?)', indicator)
                if percentage_match:
                    percentage = float(percentage_match.group(1))
                    if percentage >= 50:
                        strength = 1.0
                    elif percentage >= 25:
                        strength = 0.8
                    elif percentage >= 10:
                        strength = 0.6
            else:
                # Assess qualitative indicators
                strong_terms = ["significant", "substantial", "rapid", "explosive"]
                moderate_terms = ["steady", "consistent", "moderate", "growing"]
                weak_terms = ["slight", "minimal", "potential", "possible"]
                
                if any(term in indicator.lower() for term in strong_terms):
                    strength += 0.3
                elif any(term in indicator.lower() for term in moderate_terms):
                    strength += 0.2
                elif any(term in indicator.lower() for term in weak_terms):
                    strength += 0.1
            
            return min(strength, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error assessing growth indicator: {str(e)}")
            return 0.5

    def _assess_market_readiness(self, trend: str, evidence: List[str]) -> float:
        """Assess market readiness for an emerging trend."""
        try:
            readiness = 0.5  # Base readiness
            
            # Look for readiness indicators in evidence
            positive_indicators = 0
            negative_indicators = 0
            
            for text in evidence:
                # Check for positive indicators
                if re.search(r'\b(?:ready|prepared|anticipating|eager|demand)\b', text, re.IGNORECASE):
                    positive_indicators += 1
                
                # Check for negative indicators
                if re.search(r'\b(?:hesitant|uncertain|skeptical|resistant)\b', text, re.IGNORECASE):
                    negative_indicators += 1
                
                # Check for infrastructure/support mentions
                if re.search(r'\b(?:infrastructure|support|framework|platform)\b', text, re.IGNORECASE):
                    positive_indicators += 1
                
                # Check for barrier mentions
                if re.search(r'\b(?:barrier|obstacle|challenge|limitation)\b', text, re.IGNORECASE):
                    negative_indicators += 1
            
            # Calculate readiness score
            total_indicators = positive_indicators + negative_indicators
            if total_indicators > 0:
                readiness = 0.5 + (0.5 * (positive_indicators - negative_indicators) / total_indicators)
            
            return min(max(readiness, 0.0), 1.0)  # Keep between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error assessing market readiness: {str(e)}")
            return 0.5

    def _assess_decline_indicator(self, indicator: str) -> float:
        """Assess the strength of a decline indicator."""
        try:
            strength = 0.5  # Base strength
            
            # Check for quantified decline
            if "quantified effect:" in indicator:
                percentage_match = re.search(r'(\d+(?:\.\d+)?)', indicator)
                if percentage_match:
                    percentage = float(percentage_match.group(1))
                    if percentage >= 50:
                        strength = 1.0
                    elif percentage >= 25:
                        strength = 0.8
                    elif percentage >= 10:
                        strength = 0.6
            else:
                # Assess qualitative indicators
                strong_terms = ["dramatic", "sharp", "significant", "substantial"]
                moderate_terms = ["steady", "consistent", "moderate", "declining"]
                weak_terms = ["slight", "minimal", "gradual", "slow"]
                
                if any(term in indicator.lower() for term in strong_terms):
                    strength += 0.3
                elif any(term in indicator.lower() for term in moderate_terms):
                    strength += 0.2
                elif any(term in indicator.lower() for term in weak_terms):
                    strength += 0.1
            
            return min(strength, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error assessing decline indicator: {str(e)}")
            return 0.5

    def _assess_market_persistence(self, trend: str, evidence: List[str]) -> float:
        """Assess market persistence of a declining trend."""
        try:
            persistence = 0.5  # Base persistence
            
            # Look for persistence indicators in evidence
            strong_persistence = 0
            weak_persistence = 0
            
            for text in evidence:
                # Check for strong persistence indicators
                if re.search(r'\b(?:enduring|persistent|resilient|stable)\b', text, re.IGNORECASE):
                    strong_persistence += 1
                
                # Check for weak persistence indicators
                if re.search(r'\b(?:temporary|transitional|short-term|passing)\b', text, re.IGNORECASE):
                    weak_persistence += 1
                
                # Check for established base mentions
                if re.search(r'\b(?:established|loyal|committed)\b', text, re.IGNORECASE):
                    strong_persistence += 1
                
                # Check for replacement mentions
                if re.search(r'\b(?:replaced|superseded|outdated)\b', text, re.IGNORECASE):
                    weak_persistence += 1
            
            # Calculate persistence score
            total_indicators = strong_persistence + weak_persistence
            if total_indicators > 0:
                persistence = 0.5 + (0.5 * (strong_persistence - weak_persistence) / total_indicators)
            
            return min(max(persistence, 0.0), 1.0)  # Keep between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error assessing market persistence: {str(e)}")
            return 0.5

    def _generate_recommendations(self, market_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on market research findings."""
        try:
            recommendations = []
            
            # Genre recommendations
            if "genre_analysis" in market_report:
                genre_recs = self._generate_genre_recommendations(market_report["genre_analysis"])
                recommendations.extend(genre_recs)
            
            # Trend-based recommendations
            if "market_trends" in market_report:
                trend_recs = self._generate_trend_recommendations(market_report["market_trends"])
                recommendations.extend(trend_recs)
            
            # Competition-based recommendations
            if "competitor_analysis" in market_report:
                competitor_recs = self._generate_competitor_recommendations(
                    market_report["competitor_analysis"]
                )
                recommendations.extend(competitor_recs)
            
            # Target audience recommendations
            if "target_audience" in market_report:
                audience_recs = self._generate_audience_recommendations(
                    market_report["target_audience"]
                )
                recommendations.extend(audience_recs)
            
            # Sort by priority and confidence
            recommendations.sort(
                key=lambda x: (x.get("priority", 0), x.get("confidence", 0)),
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    async def cleanup(self) -> None:
        """Cleanup after market research completion."""
        try:
            # Clear temporary state
            self.state["memory"] = {}
            
            # Archive old cache entries
            current_time = datetime.utcnow()
            self.state["research_cache"] = {
                topic: data 
                for topic, data in self.state.get("research_cache", {}).items()
                if (current_time - datetime.fromisoformat(data["timestamp"])).days < 30
            }
            
            # Log cleanup
            self.logger.info("Market research agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")