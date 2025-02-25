from typing import Dict, Any, List
from datetime import datetime
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
import logging
import json
from nltk.tokenize import sent_tokenize
from utils.nlp_utils import TextParser

class ConsumerInsightsAgent(BaseAgent):
    def __init__(self, tools_service: ToolsService, **kwargs):
        super().__init__(**kwargs)
        self.tools_service = tools_service
        self.research_tools = tools_service.get_research_tools()
        self.text_parser = TextParser()
        self.logger = logging.getLogger(__name__)
        self.system_prompts = self._load_system_prompts()
        self.state = {
            "memory": {},
            "last_update": None,
            "insights_cache": {}
        }

    def _load_system_prompts(self) -> Dict[str, Any]:
        """Load system prompts from configuration."""
        try:
            with open("config/agent_parameters.json", "r") as file:
                return json.load(file)
        except Exception as e:
            self.logger.error(f"Error loading system prompts: {str(e)}")
            return {}

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market research and contextual research to generate consumer insights."""
        try:
            story_bible = task.get("story_bible", {})
            market_research = task.get("market_research", {})
            contextual_research = task.get("contextual_research", {})

            insights_report = {
                "novel_id": story_bible.get("novel_id"),
                "research_type": "consumer_insights",
                "reader_personas": await self._generate_reader_personas(market_research),
                "engagement_analysis": await self._analyze_engagement_factors(
                    market_research, 
                    contextual_research
                ),
                "content_preferences": await self._analyze_content_preferences(
                    market_research,
                    story_bible
                ),
                "market_opportunities": await self._identify_market_opportunities(
                    market_research,
                    contextual_research
                ),
                "risk_factors": await self._analyze_risk_factors(
                    market_research,
                    contextual_research
                ),
                "recommendations": [],
                "timestamp": datetime.utcnow().isoformat()
            }

            insights_report["recommendations"] = self._generate_recommendations(insights_report)
            
            # Validate report
            if not self._validate_report(insights_report):
                raise ValueError("Consumer insights report validation failed")
            
            # Update state
            self.state["last_update"] = datetime.utcnow()
            self.state["memory"]["last_report"] = insights_report
            
            return insights_report
            
        except Exception as e:
            self.logger.error(f"Error generating consumer insights: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _generate_reader_personas(self, market_research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed reader personas based on market research."""
        try:
            personas = []
            target_audience = market_research.get("target_audience", {})
            
            # Generate primary personas
            for demographic in target_audience.get("demographics", []):
                persona = {
                    "name": f"Persona_{len(personas) + 1}",
                    "demographic_profile": demographic,
                    "reading_preferences": self._extract_reading_preferences(
                        demographic,
                        market_research
                    ),
                    "pain_points": self._identify_pain_points(demographic, market_research),
                    "motivations": self._identify_motivations(demographic, market_research),
                    "buying_behavior": self._analyze_buying_behavior(demographic, market_research),
                    "engagement_channels": self._identify_engagement_channels(demographic),
                    "confidence_score": 0.0
                }
                
                # Calculate confidence score
                persona["confidence_score"] = self._calculate_persona_confidence(persona)
                personas.append(persona)

            return personas
            
        except Exception as e:
            self.logger.error(f"Error generating reader personas: {str(e)}")
            return []

    def _extract_reading_preferences(self, demographic: Dict[str, Any], market_research: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reading preferences for a demographic group."""
        try:
            preferences = {
                "preferred_genres": [],
                "format_preferences": [],
                "reading_frequency": None,
                "price_sensitivity": None,
                "content_preferences": {
                    "length": None,
                    "complexity": None,
                    "themes": [],
                    "style": None
                },
                "discovery_channels": []
            }
            
            # Extract from market research data
            audience_data = market_research.get("audience_analysis", {})
            demographic_key = demographic.get("id", "")
            
            if demographic_key in audience_data:
                demo_data = audience_data[demographic_key]
                
                # Process genre preferences
                preferences["preferred_genres"] = demo_data.get("genres", [])
                
                # Process format preferences
                preferences["format_preferences"] = demo_data.get("formats", [])
                
                # Process reading frequency
                preferences["reading_frequency"] = demo_data.get("reading_frequency")
                
                # Process price sensitivity
                preferences["price_sensitivity"] = demo_data.get("price_sensitivity")
                
                # Process content preferences
                content_data = demo_data.get("content_preferences", {})
                preferences["content_preferences"].update(content_data)
                
                # Process discovery channels
                preferences["discovery_channels"] = demo_data.get("discovery_channels", [])
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error extracting reading preferences: {str(e)}")
            return {}

    def _identify_pain_points(self, demographic: Dict[str, Any], market_research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify pain points for a demographic group."""
        try:
            pain_points = []
            audience_data = market_research.get("audience_analysis", {})
            
            # Extract explicit pain points
            if "pain_points" in audience_data:
                for point in audience_data["pain_points"]:
                    if self._matches_demographic(point, demographic):
                        pain_points.append({
                            "description": point["description"],
                            "severity": point.get("severity", "medium"),
                            "frequency": point.get("frequency", "medium"),
                            "impact": point.get("impact", "medium"),
                            "potential_solutions": point.get("solutions", [])
                        })
            
            # Analyze feedback data for implicit pain points
            feedback_data = market_research.get("feedback_analysis", {})
            if feedback_data:
                implicit_points = self._analyze_feedback_for_pain_points(
                    feedback_data,
                    demographic
                )
                pain_points.extend(implicit_points)
            
            return pain_points
            
        except Exception as e:
            self.logger.error(f"Error identifying pain points: {str(e)}")
            return []

    def _identify_motivations(self, demographic: Dict[str, Any], market_research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify motivations for a demographic group."""
        try:
            motivations = []
            audience_data = market_research.get("audience_analysis", {})
            
            # Extract explicit motivations
            if "motivations" in audience_data:
                for motivation in audience_data["motivations"]:
                    if self._matches_demographic(motivation, demographic):
                        motivations.append({
                            "type": motivation["type"],
                            "description": motivation["description"],
                            "strength": motivation.get("strength", "medium"),
                            "triggers": motivation.get("triggers", []),
                            "related_needs": motivation.get("related_needs", [])
                        })
            
            # Analyze behavior data for implicit motivations
            behavior_data = market_research.get("behavior_analysis", {})
            if behavior_data:
                implicit_motivations = self._analyze_behavior_for_motivations(
                    behavior_data,
                    demographic
                )
                motivations.extend(implicit_motivations)
            
            return motivations
            
        except Exception as e:
            self.logger.error(f"Error identifying motivations: {str(e)}")
            return []

    def _analyze_buying_behavior(self, demographic: Dict[str, Any], market_research: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze buying behavior for a demographic group."""
        try:
            behavior = {
                "purchase_frequency": None,
                "price_range": {
                    "min": None,
                    "max": None,
                    "optimal": None
                },
                "decision_factors": [],
                "purchase_channels": [],
                "seasonal_patterns": {},
                "impulse_factors": []
            }
            
            # Extract from market research data
            sales_data = market_research.get("sales_analysis", {})
            demographic_key = demographic.get("id", "")
            
            if demographic_key in sales_data:
                demo_sales = sales_data[demographic_key]
                
                # Process purchase frequency
                behavior["purchase_frequency"] = demo_sales.get("frequency")
                
                # Process price range
                price_data = demo_sales.get("price_data", {})
                behavior["price_range"].update(price_data)
                
                # Process decision factors
                behavior["decision_factors"] = demo_sales.get("decision_factors", [])
                
                # Process purchase channels
                behavior["purchase_channels"] = demo_sales.get("channels", [])
                
                # Process seasonal patterns
                behavior["seasonal_patterns"] = demo_sales.get("seasonal_patterns", {})
                
                # Process impulse factors
                behavior["impulse_factors"] = demo_sales.get("impulse_factors", [])
            
            return behavior
            
        except Exception as e:
            self.logger.error(f"Error analyzing buying behavior: {str(e)}")
            return {}

    def _identify_engagement_channels(self, demographic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify preferred engagement channels for a demographic group."""
        try:
            channels = []
            
            # Define common channels
            common_channels = [
                "social_media",
                "email",
                "websites",
                "mobile_apps",
                "physical_stores",
                "online_communities",
                "events"
            ]
            
            # Score each channel based on demographic data
            for channel in common_channels:
                score = self._calculate_channel_score(channel, demographic)
                if score > 0.5:  # Only include channels with good fit
                    channels.append({
                        "name": channel,
                        "score": score,
                        "recommended_content": self._suggest_channel_content(channel, demographic),
                        "best_practices": self._get_channel_best_practices(channel, demographic)
                    })
            
            # Sort by score
            channels.sort(key=lambda x: x["score"], reverse=True)
            return channels
            
        except Exception as e:
            self.logger.error(f"Error identifying engagement channels: {str(e)}")
            return []

    async def _analyze_engagement_factors(
        self,
        market_research: Dict[str, Any],
        contextual_research: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze factors that drive reader engagement."""
        try:
            engagement_analysis = {
                "emotional_drivers": self._identify_emotional_drivers(market_research),
                "narrative_preferences": self._analyze_narrative_preferences(market_research),
                "content_engagement_patterns": self._analyze_engagement_patterns(market_research),
                "cultural_relevance": self._analyze_cultural_relevance(
                    contextual_research,
                    market_research
                ),
                "engagement_metrics": {},
                "recommendations": []
            }
            
            # Calculate engagement metrics
            engagement_analysis["engagement_metrics"] = self._calculate_engagement_metrics(
                market_research
            )
            
            # Generate engagement recommendations
            engagement_analysis["recommendations"] = self._generate_engagement_recommendations(
                engagement_analysis
            )
            
            return engagement_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing engagement factors: {str(e)}")
            return {}

    def _identify_emotional_drivers(self, market_research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify emotional drivers of reader engagement."""
        try:
            drivers = []
            
            # Analyze reader feedback
            feedback_data = market_research.get("reader_feedback", {})
            sentiment_analysis = self._analyze_feedback_sentiment(feedback_data)
            
            # Extract emotional triggers
            for emotion, data in sentiment_analysis.items():
                if data["frequency"] > 0.1:  # Only include significant emotions
                    drivers.append({
                        "emotion": emotion,
                        "strength": data["strength"],
                        "triggers": data["triggers"],
                        "context": data["context"],
                        "impact": self._calculate_emotional_impact(data)
                    })
            
            # Sort by impact
            drivers.sort(key=lambda x: x["impact"], reverse=True)
            return drivers
            
        except Exception as e:
            self.logger.error(f"Error identifying emotional drivers: {str(e)}")
            return []

    def _analyze_narrative_preferences(self, market_research: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze narrative preferences from market research."""
        try:
            preferences = {
                "story_structure": {
                    "preferred_patterns": [],
                    "pacing_preferences": {},
                    "complexity_level": None
                },
                "character_elements": {
                    "preferred_archetypes": [],
                    "relationship_dynamics": [],
                    "development_patterns": []
                },
                "setting_preferences": {
                    "popular_settings": [],
                    "world_building_elements": [],
                    "atmosphere_preferences": []
                },
                "thematic_elements": {
                    "resonant_themes": [],
                    "preferred_conflicts": [],
                    "resolution_patterns": []
                }
            }
            
            # Extract from reader feedback and sales data
            feedback_data = market_research.get("reader_feedback", {})
            sales_data = market_research.get("sales_analysis", {})
            
            # Analyze story structure preferences
            preferences["story_structure"] = self._analyze_structure_preferences(
                feedback_data,
                sales_data
            )
            
            # Analyze character preferences
            preferences["character_elements"] = self._analyze_character_preferences(
                feedback_data,
                sales_data
            )
            
            # Analyze setting preferences
            preferences["setting_preferences"] = self._analyze_setting_preferences(
                feedback_data,
                sales_data
            )
            
            # Analyze thematic preferences
            preferences["thematic_elements"] = self._analyze_thematic_preferences(
                feedback_data,
                sales_data
            )
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error analyzing narrative preferences: {str(e)}")
            return {}

    def _analyze_engagement_patterns(self, market_research: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in reader engagement."""
        try:
            patterns = {
                "reading_sessions": {
                    "average_duration": None,
                    "frequency": None,
                    "peak_times": [],
                    "completion_rates": None
                },
                                "interaction_patterns": {
                    "highlights": [],
                    "annotations": [],
                    "shares": [],
                    "reviews": []
                },
                "retention_metrics": {
                    "series_continuation": None,
                    "author_loyalty": None,
                    "genre_loyalty": None
                },
                "social_engagement": {
                    "discussion_topics": [],
                    "sharing_behavior": {},
                    "community_participation": {}
                }
            }
            
            # Analyze reading session data
            session_data = market_research.get("reading_sessions", {})
            if session_data:
                patterns["reading_sessions"] = self._analyze_session_metrics(session_data)
            
            # Analyze interaction data
            interaction_data = market_research.get("interactions", {})
            if interaction_data:
                patterns["interaction_patterns"] = self._analyze_interactions(interaction_data)
            
            # Analyze retention data
            retention_data = market_research.get("retention", {})
            if retention_data:
                patterns["retention_metrics"] = self._analyze_retention(retention_data)
            
            # Analyze social engagement
            social_data = market_research.get("social_engagement", {})
            if social_data:
                patterns["social_engagement"] = self._analyze_social_metrics(social_data)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing engagement patterns: {str(e)}")
            return {}

    def _analyze_session_metrics(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reading session metrics."""
        try:
            metrics = {
                "average_duration": None,
                "frequency": None,
                "peak_times": [],
                "completion_rates": None
            }
            
            # Calculate average duration
            if "durations" in session_data:
                durations = session_data["durations"]
                if durations:
                    metrics["average_duration"] = sum(durations) / len(durations)
            
            # Calculate reading frequency
            if "timestamps" in session_data:
                metrics["frequency"] = self._calculate_reading_frequency(
                    session_data["timestamps"]
                )
            
            # Identify peak reading times
            if "hourly_distribution" in session_data:
                metrics["peak_times"] = self._identify_peak_times(
                    session_data["hourly_distribution"]
                )
            
            # Calculate completion rates
            if "completions" in session_data and "starts" in session_data:
                metrics["completion_rates"] = (
                    session_data["completions"] / session_data["starts"]
                    if session_data["starts"] > 0 else 0
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing session metrics: {str(e)}")
            return {}

    def _analyze_interactions(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reader interaction patterns."""
        try:
            patterns = {
                "highlights": [],
                "annotations": [],
                "shares": [],
                "reviews": []
            }
            
            # Analyze highlights
            if "highlights" in interaction_data:
                patterns["highlights"] = self._analyze_highlight_patterns(
                    interaction_data["highlights"]
                )
            
            # Analyze annotations
            if "annotations" in interaction_data:
                patterns["annotations"] = self._analyze_annotation_patterns(
                    interaction_data["annotations"]
                )
            
            # Analyze shares
            if "shares" in interaction_data:
                patterns["shares"] = self._analyze_sharing_patterns(
                    interaction_data["shares"]
                )
            
            # Analyze reviews
            if "reviews" in interaction_data:
                patterns["reviews"] = self._analyze_review_patterns(
                    interaction_data["reviews"]
                )
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing interactions: {str(e)}")
            return {}

    def _analyze_retention(self, retention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reader retention metrics."""
        try:
            metrics = {
                "series_continuation": None,
                "author_loyalty": None,
                "genre_loyalty": None
            }
            
            # Calculate series continuation rate
            if "series_data" in retention_data:
                metrics["series_continuation"] = self._calculate_series_retention(
                    retention_data["series_data"]
                )
            
            # Calculate author loyalty score
            if "author_data" in retention_data:
                metrics["author_loyalty"] = self._calculate_author_loyalty(
                    retention_data["author_data"]
                )
            
            # Calculate genre loyalty score
            if "genre_data" in retention_data:
                metrics["genre_loyalty"] = self._calculate_genre_loyalty(
                    retention_data["genre_data"]
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing retention: {str(e)}")
            return {}

    def _analyze_social_metrics(self, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social engagement metrics."""
        try:
            metrics = {
                "discussion_topics": [],
                "sharing_behavior": {},
                "community_participation": {}
            }
            
            # Analyze discussion topics
            if "discussions" in social_data:
                metrics["discussion_topics"] = self._analyze_discussion_topics(
                    social_data["discussions"]
                )
            
            # Analyze sharing behavior
            if "sharing" in social_data:
                metrics["sharing_behavior"] = self._analyze_sharing_behavior(
                    social_data["sharing"]
                )
            
            # Analyze community participation
            if "community" in social_data:
                metrics["community_participation"] = self._analyze_community_participation(
                    social_data["community"]
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing social metrics: {str(e)}")
            return {}

    def _calculate_reading_frequency(self, timestamps: List[str]) -> Dict[str, Any]:
        """Calculate reading frequency from session timestamps."""
        try:
            frequency = {
                "daily_average": 0,
                "weekly_pattern": {},
                "monthly_pattern": {},
                "trend": None
            }
            
            if not timestamps:
                return frequency
            
            # Convert timestamps to datetime objects
            dates = [datetime.fromisoformat(ts) for ts in timestamps]
            dates.sort()
            
            # Calculate daily average
            total_days = (dates[-1] - dates[0]).days
            if total_days > 0:
                frequency["daily_average"] = len(dates) / total_days
            
            # Analyze weekly pattern
            week_days = [date.strftime("%A") for date in dates]
            frequency["weekly_pattern"] = {
                day: week_days.count(day) / len(week_days)
                for day in set(week_days)
            }
            
            # Analyze monthly pattern
            months = [date.strftime("%B") for date in dates]
            frequency["monthly_pattern"] = {
                month: months.count(month) / len(months)
                for month in set(months)
            }
            
            # Analyze trend
            frequency["trend"] = self._analyze_frequency_trend(dates)
            
            return frequency
            
        except Exception as e:
            self.logger.error(f"Error calculating reading frequency: {str(e)}")
            return {}

    def _identify_peak_times(self, hourly_distribution: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify peak reading times from hourly distribution."""
        try:
            peaks = []
            total_readings = sum(hourly_distribution.values())
            
            if total_readings == 0:
                return peaks
            
            # Calculate hourly percentages
            hourly_percentages = {
                hour: count / total_readings
                for hour, count in hourly_distribution.items()
            }
            
            # Identify peaks (hours with above average activity)
            average = 1/24  # Expected average for uniform distribution
            threshold = average * 1.5  # 50% above average
            
            for hour, percentage in hourly_percentages.items():
                if percentage > threshold:
                    peaks.append({
                        "hour": hour,
                        "percentage": percentage,
                        "relative_strength": percentage / average
                    })
            
            # Sort by percentage
            peaks.sort(key=lambda x: x["percentage"], reverse=True)
            return peaks
            
        except Exception as e:
            self.logger.error(f"Error identifying peak times: {str(e)}")
            return []

    def _analyze_cultural_relevance(
        self,
        contextual_research: Dict[str, Any],
        market_research: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze cultural relevance of content."""
        try:
            relevance = {
                "cultural_alignment": [],
                "social_trends": [],
                "demographic_fit": {},
                "zeitgeist_alignment": None,
                "recommendations": []
            }
            
            # Analyze cultural alignment
            if "cultural_factors" in contextual_research:
                relevance["cultural_alignment"] = self._analyze_cultural_alignment(
                    contextual_research["cultural_factors"],
                    market_research
                )
            
            # Analyze social trends
            if "social_trends" in contextual_research:
                relevance["social_trends"] = self._analyze_social_trends(
                    contextual_research["social_trends"]
                )
            
            # Analyze demographic fit
            if "demographics" in market_research:
                relevance["demographic_fit"] = self._analyze_demographic_fit(
                    contextual_research,
                    market_research["demographics"]
                )
            
            # Calculate zeitgeist alignment
            relevance["zeitgeist_alignment"] = self._calculate_zeitgeist_alignment(
                relevance["cultural_alignment"],
                relevance["social_trends"]
            )
            
            # Generate recommendations
            relevance["recommendations"] = self._generate_cultural_recommendations(
                relevance
            )
            
            return relevance
            
        except Exception as e:
            self.logger.error(f"Error analyzing cultural relevance: {str(e)}")
            return {}

    def _validate_report(self, report: Dict[str, Any]) -> bool:
        """Validate the consumer insights report."""
        try:
            # Check required sections
            required_sections = [
                "reader_personas",
                "engagement_analysis",
                "content_preferences",
                "market_opportunities",
                "risk_factors",
                "recommendations"
            ]
            
            if not all(section in report for section in required_sections):
                return False
            
            # Validate reader personas
            if not report["reader_personas"]:
                return False
            
            # Validate engagement analysis
            if not self._validate_engagement_analysis(report["engagement_analysis"]):
                return False
            
            # Validate content preferences
            if not self._validate_content_preferences(report["content_preferences"]):
                return False
            
            # Validate recommendations
            if not report["recommendations"]:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating report: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Cleanup after consumer insights analysis completion."""
        try:
            # Clear temporary state
            self.state["memory"] = {}
            
            # Archive old cache entries
            current_time = datetime.utcnow()
            self.state["insights_cache"] = {
                topic: data 
                for topic, data in self.state["insights_cache"].items()
                if (current_time - datetime.fromisoformat(data["timestamp"])).days < 30
            }
            
            # Log cleanup
            self.logger.info("Consumer insights agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")