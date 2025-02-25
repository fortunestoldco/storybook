from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
import json
import logging
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class ContextualResearchAgent(BaseAgent):
    def __init__(self, tools_service: ToolsService, **kwargs):
        super().__init__(**kwargs)
        self.tools_service = tools_service
        self.research_tools = tools_service.get_research_tools()
        self.logger = logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.system_prompts = self._load_system_prompts()
        self.state = {
            "memory": {},
            "last_update": None,
            "research_cache": {}
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
        """Handle contextual research tasks."""
        try:
            story_bible = task.get("story_bible", {})
            
            research_report = {
                "novel_id": story_bible.get("novel_id"),
                "research_type": "contextual",
                "historical_context": await self._research_historical_context(story_bible),
                "cultural_elements": await self._research_cultural_elements(story_bible),
                "scientific_facts": await self._research_scientific_elements(story_bible),
                "geographical_details": await self._research_geographical_elements(story_bible),
                "sociological_factors": await self._research_sociological_factors(story_bible),
                "technological_aspects": await self._research_technological_aspects(story_bible),
                "recommendations": [],
                "confidence_scores": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Generate recommendations
            research_report["recommendations"] = self._generate_recommendations(research_report)
            
            # Calculate confidence scores
            research_report["confidence_scores"] = self._calculate_confidence_scores(research_report)
            
            # Validate report
            if not self._validate_report(research_report):
                raise ValueError("Contextual research report validation failed")
            
            # Update state
            self.state["last_update"] = datetime.utcnow()
            self.state["memory"]["last_report"] = research_report
            
            return research_report
            
        except Exception as e:
            self.logger.error(f"Error in contextual research: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _research_historical_context(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Research historical context relevant to the story."""
        try:
            historical_context = {
                "time_period": {
                    "start": None,
                    "end": None,
                    "key_events": []
                },
                "social_conditions": [],
                "political_landscape": [],
                "economic_factors": [],
                "technological_state": [],
                "cultural_norms": [],
                "sources": [],
                "confidence_score": 0.0
            }
            
            # Extract time period from story bible
            time_period = story_bible.get("time_period", {})
            if time_period:
                historical_context["time_period"].update(time_period)
            
            # Research historical elements
            query = self._construct_historical_query(story_bible)
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=query,
                filters={"category": "historical", "time_period": time_period}
            )
            
            # Process results
            for doc in doc_results:
                self._process_historical_document(doc, historical_context)
            
            # Calculate confidence score
            historical_context["confidence_score"] = self._calculate_historical_confidence(
                historical_context
            )
            
            return historical_context
            
        except Exception as e:
            self.logger.error(f"Error researching historical context: {str(e)}")
            return {}

    async def _research_cultural_elements(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Research cultural elements relevant to the story."""
        try:
            cultural_elements = {
                "customs_traditions": [],
                "beliefs_values": [],
                "social_structures": [],
                "art_literature": [],
                "language_communication": [],
                "religious_practices": [],
                "sources": [],
                "confidence_score": 0.0
            }
            
            # Research cultural elements
            query = self._construct_cultural_query(story_bible)
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=query,
                filters={"category": "cultural"}
            )
            
            # Process results
            for doc in doc_results:
                self._process_cultural_document(doc, cultural_elements)
            
            # Calculate confidence score
            cultural_elements["confidence_score"] = self._calculate_cultural_confidence(
                cultural_elements
            )
            
            return cultural_elements
            
        except Exception as e:
            self.logger.error(f"Error researching cultural elements: {str(e)}")
            return {}

    async def _research_scientific_elements(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Research scientific elements relevant to the story."""
        try:
            scientific_elements = {
                "scientific_principles": [],
                "technological_concepts": [],
                "natural_phenomena": [],
                "medical_aspects": [],
                "environmental_factors": [],
                "sources": [],
                "confidence_score": 0.0
            }
            
            # Research scientific elements
            query = self._construct_scientific_query(story_bible)
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=query,
                filters={"category": "scientific"}
            )
            
            # Process results
            for doc in doc_results:
                self._process_scientific_document(doc, scientific_elements)
            
            # Calculate confidence score
            scientific_elements["confidence_score"] = self._calculate_scientific_confidence(
                scientific_elements
            )
            
            return scientific_elements
            
        except Exception as e:
            self.logger.error(f"Error researching scientific elements: {str(e)}")
            return {}

    async def _research_geographical_elements(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Research geographical elements relevant to the story."""
        try:
            geographical_elements = {
                "locations": [],
                "climate_weather": [],
                "terrain_features": [],
                "flora_fauna": [],
                "human_geography": [],
                "sources": [],
                "confidence_score": 0.0
            }
            
            # Research geographical elements
            query = self._construct_geographical_query(story_bible)
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=query,
                filters={"category": "geographical"}
            )
            
            # Process results
            for doc in doc_results:
                self._process_geographical_document(doc, geographical_elements)
            
            # Calculate confidence score
            geographical_elements["confidence_score"] = self._calculate_geographical_confidence(
                geographical_elements
            )
            
            return geographical_elements
            
        except Exception as e:
            self.logger.error(f"Error researching geographical elements: {str(e)}")
            return {}

    async def _research_sociological_factors(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Research sociological factors relevant to the story."""
        try:
            sociological_factors = {
                "social_structures": [],
                "class_systems": [],
                "gender_roles": [],
                "family_dynamics": [],
                "power_relationships": [],
                "education_systems": [],
                "sources": [],
                "confidence_score": 0.0
            }
            
            # Research sociological factors
            query = self._construct_sociological_query(story_bible)
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=query,
                filters={"category": "sociological"}
            )
            
            # Process results
            for doc in doc_results:
                self._process_sociological_document(doc, sociological_factors)
            
            # Calculate confidence score
            sociological_factors["confidence_score"] = self._calculate_sociological_confidence(
                sociological_factors
            )
            
            return sociological_factors
            
        except Exception as e:
            self.logger.error(f"Error researching sociological factors: {str(e)}")
            return {}

    async def _research_technological_aspects(self, story_bible: Dict[str, Any]) -> Dict[str, Any]:
        """Research technological aspects relevant to the story."""
        try:
            technological_aspects = {
                "available_technologies": [],
                "technological_impact": [],
                "innovation_patterns": [],
                "technological_limitations": [],
                "future_projections": [],
                "sources": [],
                "confidence_score": 0.0
            }
            
            # Research technological aspects
            query = self._construct_technological_query(story_bible)
            doc_results = await self.tools_service.get_document_retriever()._arun(
                query=query,
                filters={"category": "technological"}
            )
            
            # Process results
            for doc in doc_results:
                self._process_technological_document(doc, technological_aspects)
            
            # Calculate confidence score
            technological_aspects["confidence_score"] = self._calculate_technological_confidence(
                technological_aspects
            )
            
            return technological_aspects
            
        except Exception as e:
            self.logger.error(f"Error researching technological aspects: {str(e)}")
            return {}

    def _construct_historical_query(self, story_bible: Dict[str, Any]) -> str:
        """Construct a query for historical research."""
        try:
            query_elements = []
            
            # Add time period
            time_period = story_bible.get("time_period", {})
            if time_period:
                if "start" in time_period:
                    query_elements.append(f"from {time_period['start']}")
                if "end" in time_period:
                    query_elements.append(f"to {time_period['end']}")
            
            # Add location
            location = story_bible.get("location", "")
            if location:
                query_elements.append(f"in {location}")
            
            # Add themes
            themes = story_bible.get("themes", [])
            if themes:
                query_elements.append(f"related to {', '.join(themes)}")
            
            # Construct final query
            base_query = "historical context"
            if query_elements:
                base_query += " " + " ".join(query_elements)
            
            return base_query
            
        except Exception as e:
            self.logger.error(f"Error constructing historical query: {str(e)}")
            return "historical context"

    def _process_historical_document(self, doc: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Process a historical document and update context."""
        try:
            content = doc.get("content", "")
            sentences = sent_tokenize(content)
            
            for sentence in sentences:
                # Extract key events
                if self._contains_event_indicators(sentence):
                    event = self._extract_historical_event(sentence)
                    if event:
                        context["time_period"]["key_events"].append(event)
                
                # Extract social conditions
                if self._contains_social_indicators(sentence):
                    condition = self._extract_social_condition(sentence)
                    if condition:
                        context["social_conditions"].append(condition)
                
                # Extract political landscape
                if self._contains_political_indicators(sentence):
                    political_element = self._extract_political_element(sentence)
                    if political_element:
                        context["political_landscape"].append(political_element)
                
                # Extract economic factors
                if self._contains_economic_indicators(sentence):
                    economic_factor = self._extract_economic_factor(sentence)
                    if economic_factor:
                        context["economic_factors"].append(economic_factor)
            
            # Add source
            if doc.get("source"):
                context["sources"].append(doc["source"])
                
        except Exception as e:
            self.logger.error(f"Error processing historical document: {str(e)}")

    def _contains_event_indicators(self, text: str) -> bool:
        """Check if text contains historical event indicators."""
        indicators = [
            r"\bin\s+\d{4}\b",
            r"\bduring\s+the\b",
            r"\bwhen\s+the\b",
            r"\boccurred\b",
            r"\btook\s+place\b"
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in indicators)

    def _extract_historical_event(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract historical event details from text."""
        try:
            # Extract date
            date_match = re.search(r"\b(\d{4})\b", text)
            date = date_match.group(1) if date_match else None
            
            # Extract event description
            event_description = text
            
            # Extract location if present
            location_match = re.search(r"in\s+([^,.]+)", text)
            location = location_match.group(1) if location_match else None
            
            return {
                "date": date,
                "description": event_description,
                "location": location,
                "confidence": self._calculate_event_confidence(text)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting historical event: {str(e)}")
            return None

    def _calculate_event_confidence(self, text: str) -> float:
        """Calculate confidence score for a historical event."""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on date presence
            if re.search(r"\b\d{4}\b", text):
                confidence += 0.2
            
            # Adjust based on location presence
            if re.search(r"in\s+([^,.]+)", text):
                confidence += 0.1
            
            # Adjust based on source indicators
            if re.search(r"\b(?:according to|cited in|documented in)\b", text, re.IGNORECASE):
                confidence += 0.2
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating event confidence: {str(e)}")
            return 0.5

    def _validate_report(self, report: Dict[str, Any]) -> bool:
        """Validate the contextual research report."""
        try:
            # Check required sections
            required_sections = [
                "historical_context",
                "cultural_elements",
                "scientific_facts",
                "geographical_details",
                "sociological_factors",
                "technological_aspects",
                "recommendations",
                "confidence_scores"
            ]
            
            if not all(section in report for section in required_sections):
                return False

            # Validate historical context
            if not self._validate_historical_context(report["historical_context"]):
                return False

            # Validate cultural elements
            if not self._validate_cultural_elements(report["cultural_elements"]):
                return False

            # Validate scientific facts
            if not self._validate_scientific_facts(report["scientific_facts"]):
                return False

            # Validate geographical details
            if not self._validate_geographical_details(report["geographical_details"]):
                return False

            # Validate sociological factors
            if not self._validate_sociological_factors(report["sociological_factors"]):
                return False

            # Validate technological aspects
            if not self._validate_technological_aspects(report["technological_aspects"]):
                return False

            # Validate recommendations
            if not report["recommendations"]:
                return False

            # Validate confidence scores
            if not all(0.0 <= score <= 1.0 for score in report["confidence_scores"].values()):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating report: {str(e)}")
            return False

    def _validate_historical_context(self, context: Dict[str, Any]) -> bool:
        """Validate historical context section."""
        try:
            required_fields = [
                "time_period",
                "social_conditions",
                "political_landscape",
                "economic_factors",
                "sources",
                "confidence_score"
            ]

            if not all(field in context for field in required_fields):
                return False

            # Validate time period
            time_period = context["time_period"]
            if not isinstance(time_period, dict) or "key_events" not in time_period:
                return False

            # Validate confidence score
            if not 0.0 <= context["confidence_score"] <= 1.0:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating historical context: {str(e)}")
            return False

    def _generate_recommendations(self, research_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on research findings."""
        try:
            recommendations = []

            # Historical recommendations
            if "historical_context" in research_report:
                historical_recs = self._generate_historical_recommendations(
                    research_report["historical_context"]
                )
                recommendations.extend(historical_recs)

            # Cultural recommendations
            if "cultural_elements" in research_report:
                cultural_recs = self._generate_cultural_recommendations(
                    research_report["cultural_elements"]
                )
                recommendations.extend(cultural_recs)

            # Scientific recommendations
            if "scientific_facts" in research_report:
                scientific_recs = self._generate_scientific_recommendations(
                    research_report["scientific_facts"]
                )
                recommendations.extend(scientific_recs)

            # Geographical recommendations
            if "geographical_details" in research_report:
                geographical_recs = self._generate_geographical_recommendations(
                    research_report["geographical_details"]
                )
                recommendations.extend(geographical_recs)

            # Sort by priority and confidence
            recommendations.sort(
                key=lambda x: (x.get("priority", 0), x.get("confidence", 0)),
                reverse=True
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def _calculate_confidence_scores(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for each section of the report."""
        try:
            confidence_scores = {}

            # Calculate historical confidence
            if "historical_context" in report:
                confidence_scores["historical"] = report["historical_context"]["confidence_score"]

            # Calculate cultural confidence
            if "cultural_elements" in report:
                confidence_scores["cultural"] = report["cultural_elements"]["confidence_score"]

            # Calculate scientific confidence
            if "scientific_facts" in report:
                confidence_scores["scientific"] = report["scientific_facts"]["confidence_score"]

            # Calculate geographical confidence
            if "geographical_details" in report:
                confidence_scores["geographical"] = report["geographical_details"]["confidence_score"]

            # Calculate sociological confidence
            if "sociological_factors" in report:
                confidence_scores["sociological"] = report["sociological_factors"]["confidence_score"]

            # Calculate technological confidence
            if "technological_aspects" in report:
                confidence_scores["technological"] = report["technological_aspects"]["confidence_score"]

            # Calculate overall confidence
            if confidence_scores:
                confidence_scores["overall"] = sum(confidence_scores.values()) / len(confidence_scores)

            return confidence_scores

        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {str(e)}")
            return {}

    async def cleanup(self) -> None:
        """Cleanup after contextual research completion."""
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
            self.logger.info("Contextual research agent cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")