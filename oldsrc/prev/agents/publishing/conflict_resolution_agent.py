from datetime import datetime
import logging
from agents.base_agent import BaseAgent
from services.tools_service import ToolsService
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Conflict(BaseModel):
    """Model for a story conflict."""
    conflict_id: str = Field(description="Unique identifier for the conflict")
    type: str = Field(description="Type of conflict (internal, interpersonal, external, etc.)")
    participants: List[str] = Field(description="Characters or entities involved in the conflict")
    stakes: Dict[str, Any] = Field(description="What's at risk for each participant")
    motivations: Dict[str, List[str]] = Field(description="Motivations for each participant")
    escalation_points: List[Dict[str, Any]] = Field(description="Key points where conflict intensifies")
    resolution_options: List[Dict[str, Any]] = Field(description="Possible ways to resolve the conflict")
    status: str = Field(description="Current status of the conflict")
    impact: Dict[str, Any] = Field(description="Impact on story and characters")

class ConflictAnalysis(BaseModel):
    """Model for conflict analysis results."""
    tension_curve: List[Dict[str, Any]] = Field(description="Analysis of tension progression")
    balance_assessment: Dict[str, Any] = Field(description="Assessment of power dynamics")
    resolution_feasibility: Dict[str, Any] = Field(description="Evaluation of possible resolutions")
    plot_impact: Dict[str, Any] = Field(description="Impact on overall plot")
    character_development: Dict[str, Any] = Field(description="Impact on character arcs")
    recommendations: List[Dict[str, Any]] = Field(description="Suggested improvements")

class ConflictResolutionAgent(BaseAgent):
    """Agent responsible for managing and resolving story conflicts."""
    
    def __init__(self, tools_service: ToolsService):
        super().__init__(tools_service)
        self.logger = logging.getLogger(__name__)
        self.tools_service = tools_service
        
        try:
            self.llm = Ollama(
                base_url=tools_service.config.get('ollama_base_url'),
                model="mixtral"
            )
        except Exception as e:
            self.logger.error(f"Error initializing Ollama: {str(e)}")
            raise

        self.state = {
            "active_conflicts": {},
            "resolved_conflicts": {},
            "conflict_relationships": {},
            "last_analysis": None
        }

    async def create_conflict(
        self,
        conflict_type: str,
        participants: List[str],
        stakes: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> Conflict:
        """Create a new story conflict."""
        try:
            # Validate conflict type
            valid_types = [
                'internal', 'interpersonal', 'external',
                'situational', 'supernatural', 'societal'
            ]
            if conflict_type not in valid_types:
                raise ValueError(f"Invalid conflict type. Must be one of: {valid_types}")

            # Generate unique conflict ID
            conflict_id = f"CONF_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Analyze participants and stakes
            motivations = await self._analyze_participant_motivations(
                participants,
                stakes,
                story_bible
            )

            # Create conflict
            conflict = Conflict(
                conflict_id=conflict_id,
                type=conflict_type,
                participants=participants,
                stakes=stakes,
                motivations=motivations,
                escalation_points=[],
                resolution_options=[],
                status="active",
                impact={"plot": [], "characters": {}}
            )

            # Generate possible resolution options
            resolution_options = await self._generate_resolution_options(
                conflict,
                story_bible
            )
            conflict.resolution_options = resolution_options

            # Add to state
            self.state["active_conflicts"][conflict_id] = conflict

            # Update conflict relationships
            await self._update_conflict_relationships(conflict)

            return conflict

        except Exception as e:
            self.logger.error(f"Error creating conflict: {str(e)}")
            raise

    async def add_escalation_point(
        self,
        conflict_id: str,
        escalation_details: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> Conflict:
        """Add an escalation point to an existing conflict."""
        try:
            if conflict_id not in self.state["active_conflicts"]:
                raise ValueError(f"Conflict {conflict_id} not found or already resolved")

            conflict = self.state["active_conflicts"][conflict_id]

            # Validate escalation details
            required_fields = ['trigger', 'consequences', 'tension_level']
            for field in required_fields:
                if field not in escalation_details:
                    raise ValueError(f"Missing required field: {field}")

            # Analyze tension progression
            current_tension = await self._analyze_tension_level(
                conflict.escalation_points,
                escalation_details
            )

            # Add escalation point
            escalation_point = {
                "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
                "trigger": escalation_details["trigger"],
                "consequences": escalation_details["consequences"],
                "tension_level": current_tension,
                "impact": await self._analyze_escalation_impact(
                    conflict,
                    escalation_details,
                    story_bible
                )
            }

            conflict.escalation_points.append(escalation_point)

            # Update resolution options based on new escalation
            conflict.resolution_options = await self._generate_resolution_options(
                conflict,
                story_bible
            )

            return conflict

        except Exception as e:
            self.logger.error(f"Error adding escalation point: {str(e)}")
            raise

    async def analyze_conflict(
        self,
        conflict_id: str,
        story_bible: Dict[str, Any]
    ) -> ConflictAnalysis:
        """Analyze a conflict's structure and effectiveness."""
        try:
            if conflict_id not in self.state["active_conflicts"]:
                raise ValueError(f"Conflict {conflict_id} not found or already resolved")

            conflict = self.state["active_conflicts"][conflict_id]

            # Analyze tension progression
            tension_curve = await self._analyze_tension_curve(conflict)

            # Assess power balance
            balance_assessment = await self._assess_power_balance(
                conflict,
                story_bible
            )

            # Evaluate resolution feasibility
            resolution_feasibility = await self._evaluate_resolutions(
                conflict,
                story_bible
            )

            # Analyze plot impact
            plot_impact = await self._analyze_plot_impact(
                conflict,
                story_bible
            )

            # Analyze character development
            character_development = await self._analyze_character_impact(
                conflict,
                story_bible
            )

            # Generate recommendations
            recommendations = await self._generate_conflict_recommendations(
                conflict,
                tension_curve,
                balance_assessment,
                resolution_feasibility,
                plot_impact,
                character_development
            )

            analysis = ConflictAnalysis(
                tension_curve=tension_curve,
                balance_assessment=balance_assessment,
                resolution_feasibility=resolution_feasibility,
                plot_impact=plot_impact,
                character_development=character_development,
                recommendations=recommendations
            )

            # Update state
            self.state["last_analysis"] = {
                "conflict_id": conflict_id,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing conflict: {str(e)}")
            raise

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution_details: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> Conflict:
        """Resolve an active conflict."""
        try:
            if conflict_id not in self.state["active_conflicts"]:
                raise ValueError(f"Conflict {conflict_id} not found or already resolved")

            conflict = self.state["active_conflicts"][conflict_id]

            # Validate resolution
            if not await self._validate_resolution(
                conflict,
                resolution_details,
                story_bible
            ):
                raise ValueError("Invalid resolution: Does not satisfy conflict parameters")

            # Apply resolution
            conflict.status = "resolved"
            conflict.impact["resolution"] = {
                "method": resolution_details["method"],
                "consequences": resolution_details["consequences"],
                "timestamp": datetime.utcnow().isoformat()
            }

            # Update character arcs
            await self._update_character_arcs(
                conflict,
                resolution_details,
                story_bible
            )

            # Move to resolved conflicts
            self.state["resolved_conflicts"][conflict_id] = conflict
            del self.state["active_conflicts"][conflict_id]

            # Update related conflicts
            await self._update_related_conflicts(
                conflict,
                resolution_details
            )

            return conflict

        except Exception as e:
            self.logger.error(f"Error resolving conflict: {str(e)}")
            raise

    async def _analyze_participant_motivations(
        self,
        participants: List[str],
        stakes: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze and determine motivations for conflict participants."""
        try:
            motivations = {}
            for participant in participants:
                # Get character data from story bible
                character_data = story_bible.get("characters", {}).get(participant, {})

                # Generate motivations based on character traits and stakes
                prompt = f"""
                Analyze motivations for character {participant} in this conflict based on:
                Character traits: {character_data.get('traits', [])}
                Goals: {character_data.get('goals', [])}
                Stakes: {stakes.get(participant, {})}

                Return a list of primary motivations driving their involvement.
                """

                response = await self.llm.agenerate([prompt])
                parsed_motivations = self._parse_motivation_response(
                    response.generations[0].text
                )
                motivations[participant] = parsed_motivations

            return motivations

        except Exception as e:
            self.logger.error(f"Error analyzing participant motivations: {str(e)}")
            raise

    async def _generate_resolution_options(
        self,
        conflict: Conflict,
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate possible resolution options for the conflict."""
        try:
            prompt = f"""
            Generate potential resolution options for this conflict:
            Type: {conflict.type}
            Participants: {conflict.participants}
            Stakes: {conflict.stakes}
            Motivations: {conflict.motivations}
            Current escalation points: {conflict.escalation_points}

            Consider:
            - Character arcs and development
            - Plot implications
            - Theme alignment
            - Dramatic satisfaction
            - Logical consistency

            Return a list of resolution options with their consequences and requirements.
            """

            response = await self.llm.agenerate([prompt])
            return self._parse_resolution_options(response.generations[0].text)

        except Exception as e:
            self.logger.error(f"Error generating resolution options: {str(e)}")
            raise

    def _parse_motivation_response(self, response: str) -> List[str]:
        """Parse LLM response into a list of motivations."""
        try:
            # Clean and split response
            lines = response.strip().split('\n')
            motivations = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith(('#', '-', '*')):
                    motivations.append(line)

            return motivations

        except Exception as e:
            self.logger.error(f"Error parsing motivation response: {str(e)}")
            return []

    def _parse_resolution_options(
        self,
        response: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured resolution options."""
        try:
            # Clean up the response to ensure valid JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
                
            return json.loads(cleaned_response)

        except Exception as e:
            self.logger.error(f"Error parsing resolution options: {str(e)}")
            return []

    async def _analyze_tension_level(
        self,
        previous_points: List[Dict[str, Any]],
        new_point: Dict[str, Any]
    ) -> float:
        """Analyze and calculate tension level for new escalation point."""
        try:
            # Get previous tension level
            previous_tension = 0.0
            if previous_points:
                previous_tension = previous_points[-1]["tension_level"]

            # Calculate tension change based on escalation details
            tension_change = self._calculate_tension_change(new_point)

            # Ensure tension stays within 0-1 range
            new_tension = max(0.0, min(1.0, previous_tension + tension_change))

            return new_tension

        except Exception as e:
            self.logger.error(f"Error analyzing tension level: {str(e)}")
            return 0.0

    def _calculate_tension_change(
        self,
        escalation_point: Dict[str, Any]
    ) -> float:
        """Calculate tension change based on escalation details."""
        try:
            # Base tension change from provided level
            base_change = float(escalation_point["tension_level"])

            # Modify based on consequences
            consequences = escalation_point["consequences"]
            if isinstance(consequences, list):
                # More consequences = more tension
                base_change *= (1 + (len(consequences) * 0.1))

            return max(-0.5, min(0.5, base_change))

        except Exception as e:
            self.logger.error(f"Error calculating tension change: {str(e)}")
            return 0.0

    async def _analyze_escalation_impact(
        self,
        conflict: Conflict,
        escalation_details: Dict[str, Any],
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of an escalation point."""
        try:
            prompt = f"""
            Analyze the impact of this escalation:
            Conflict type: {conflict.type}
            Participants: {conflict.participants}
            Escalation trigger: {escalation_details['trigger']}
            Consequences: {escalation_details['consequences']}

            Consider impact on:
            1. Each participant's position
            2. Power dynamics
            3. Stakes and motivations
            4. Plot progression
            5. Theme development

            Return a structured analysis of the impacts.
            """

            response = await self.llm.agenerate([prompt])
            return self._parse_impact_analysis(response.generations[0].text)

        except Exception as e:
            self.logger.error(f"Error analyzing escalation impact: {str(e)}")
            raise

    def _parse_impact_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured impact analysis."""
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:-3]
            return json.loads(cleaned_response)

        except Exception as e:
            self.logger.error(f"Error parsing impact analysis: {str(e)}")
            return {}

    async def _analyze_tension_curve(
        self,
        conflict: Conflict
    ) -> List[Dict[str, Any]]:
        """Analyze the tension progression throughout the conflict."""
        try:
            tension_points = []
            current_tension = 0.0

            # Analyze initial tension
            initial_tension = {
                "point": "initial",
                "level": self._calculate_initial_tension(conflict),
                "timestamp": (
                    conflict.escalation_points[0]["timestamp"]
                    if conflict.escalation_points else None
                )
            }
            tension_points.append(initial_tension)
            current_tension = initial_tension["level"]

            # Analyze each escalation point
            for point in conflict.escalation_points:
                tension_change = self._calculate_tension_change(point)
                current_tension = max(0.0, min(1.0, current_tension + tension_change))

                tension_points.append({
                    "point": point["trigger"],
                    "level": current_tension,
                    "timestamp": point["timestamp"],
                    "factors": self._analyze_tension_factors(point)
                })

            return tension_points

        except Exception as e:
            self.logger.error(f"Error analyzing tension curve: {str(e)}")
            return []

    def _calculate_initial_tension(self, conflict: Conflict) -> float:
        """Calculate the initial tension level of a conflict."""
        try:
            initial_tension = 0.2  # Base tension level

            # Adjust based on conflict type
            type_modifiers = {
                'internal': 0.1,
                'interpersonal': 0.2,
                'external': 0.3,
                'situational': 0.25,
                'supernatural': 0.35,
                'societal': 0.3
            }
            initial_tension += type_modifiers.get(conflict.type, 0.0)

            # Adjust based on stakes
            if conflict.stakes:
                stakes_severity = sum(
                    self._evaluate_stakes_severity(stake)
                    for stake in conflict.stakes.values()
                ) / len(conflict.stakes)
                initial_tension += stakes_severity * 0.2

            # Adjust based on number of participants
            initial_tension += min(0.1, len(conflict.participants) * 0.02)

            return min(1.0, initial_tension)

        except Exception as e:
            self.logger.error(f"Error calculating initial tension: {str(e)}")
            return 0.2

    def _evaluate_stakes_severity(self, stake: Any) -> float:
        """Evaluate the severity of stakes on a 0-1 scale."""
        try:
            if isinstance(stake, dict):
                severity_indicators = {
                    'life': 1.0,
                    'death': 1.0,
                    'survival': 0.9,
                    'freedom': 0.8,
                    'love': 0.7,
                    'family': 0.7,
                    'power': 0.6,
                    'wealth': 0.5,
                    'reputation': 0.4,
                    'comfort': 0.3
                }

                max_severity = 0.0
                description = str(stake).lower()

                for indicator, value in severity_indicators.items():
                    if indicator in description:
                        max_severity = max(max_severity, value)

                return max_severity

            return 0.5  # Default medium severity

        except Exception as e:
            self.logger.error(f"Error evaluating stakes severity: {str(e)}")
            return 0.5

    def _analyze_tension_factors(
        self,
        escalation_point: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze factors contributing to tension at an escalation point."""
        try:
            factors = {
                "immediate_impact": [],
                "relationship_changes": [],
                "power_shifts": [],
                "stake_changes": []
            }

            consequences = escalation_point.get("consequences", [])
            if isinstance(consequences, list):
                factors["immediate_impact"] = [
                    {
                        "description": c,
                        "severity": self._evaluate_consequence_severity(c)
                    }
                    for c in consequences
                ]

            if "relationship_changes" in escalation_point:
                factors["relationship_changes"] = escalation_point["relationship_changes"]

            if "power_shifts" in escalation_point:
                factors["power_shifts"] = escalation_point["power_shifts"]

            if "stake_changes" in escalation_point:
                factors["stake_changes"] = escalation_point["stake_changes"]

            return factors

        except Exception as e:
            self.logger.error(f"Error analyzing tension factors: {str(e)}")
            return {}

    def _evaluate_consequence_severity(self, consequence: str) -> float:
        """Evaluate the severity of a consequence on a 0-1 scale."""
        try:
            severity_keywords = {
                'catastrophic': 1.0,
                'devastating': 0.9,
                'major': 0.8,
                'significant': 0.7,
                'moderate': 0.5,
                'minor': 0.3,
                'slight': 0.2,
                'minimal': 0.1
            }

            consequence_lower = consequence.lower()
            max_severity = 0.0

            for keyword, weight in severity_keywords.items():
                if keyword in consequence_lower:
                    max_severity = max(max_severity, weight)

            return max_severity if max_severity > 0 else 0.5

        except Exception as e:
            self.logger.error(f"Error evaluating consequence severity: {str(e)}")
            return 0.5

    async def _assess_power_balance(
        self,
        conflict: Conflict,
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the power dynamics between conflict participants."""
        try:
            assessment = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_balance": None,
                "participant_power": {},
                "power_shifts": [],
                "recommendations": []
            }

            for participant in conflict.participants:
                power_level = await self._analyze_participant_power(
                    participant,
                    conflict,
                    story_bible
                )
                assessment["participant_power"][participant] = power_level

            power_levels = list(assessment["participant_power"].values())
            if power_levels:
                max_power = max(power_levels)
                min_power = min(power_levels)
                power_range = max_power - min_power

                if max_power > 0:
                    balance_score = 1.0 - (power_range / max_power)
                else:
                    balance_score = 0.0

                if power_range > 0.3:
                    dominant_participant = max(
                        assessment["participant_power"].items(),
                        key=lambda x: x[1]
                    )[0]
                else:
                    dominant_participant = None

                assessment["overall_balance"] = {
                    "balance_score": balance_score,
                    "dominant_participant": dominant_participant
                }
            else:
                assessment["overall_balance"] = {
                    "balance_score": 0.0,
                    "dominant_participant": None
                }

            assessment["power_shifts"] = await self._analyze_power_shifts(
                conflict,
                assessment["participant_power"]
            )

            assessment["recommendations"] = await self._generate_power_balance_recommendations(
                conflict,
                assessment
            )

            return assessment

        except Exception as e:
            self.logger.error(f"Error assessing power balance: {str(e)}")
            raise

    async def _analyze_participant_power(
        self,
        participant: str,
        conflict: Conflict,
        story_bible: Dict[str, Any]
    ) -> float:
        """Analyze the power level of a conflict participant."""
        try:
            character_data = story_bible.get("characters", {}).get(participant, {})
            base_power = self._calculate_base_power(character_data)
            power_modifiers = self._calculate_conflict_power_modifiers(
                participant,
                conflict
            )
            context_modifiers = await self._analyze_context_power_modifiers(
                participant,
                conflict,
                story_bible
            )

            total_power = base_power
            for modifier in power_modifiers + context_modifiers:
                if modifier["type"] == "multiply":
                    total_power *= modifier["value"]
                else:
                    total_power += modifier["value"]

            return max(0.0, min(1.0, total_power))

        except Exception as e:
            self.logger.error(f"Error analyzing participant power: {str(e)}")
            return 0.5

    def _calculate_base_power(self, character_data: Dict[str, Any]) -> float:
        """Calculate base power level from character attributes."""
        try:
            base_power = 0.5

            abilities = character_data.get("abilities", {})
            if abilities:
                numeric_abilities = [
                    float(v) for v in abilities.values()
                    if isinstance(v, (int, float))
                ]
                if numeric_abilities:
                    ability_score = sum(numeric_abilities) / len(numeric_abilities)
                    base_power += ability_score * 0.2

            resources = character_data.get("resources", {})
            if resources:
                numeric_resources = [
                    float(v) for v in resources.values()
                    if isinstance(v, (int, float))
                ]
                if numeric_resources:
                    resource_score = sum(numeric_resources) / len(numeric_resources)
                    base_power += resource_score * 0.15

            influence = character_data.get("influence", {})
            if influence:
                numeric_influence = [
                    float(v) for v in influence.values()
                    if isinstance(v, (int, float))
                ]
                if numeric_influence:
                    influence_score = sum(numeric_influence) / len(numeric_influence)
                    base_power += influence_score * 0.15

            return max(0.0, min(1.0, base_power))

        except Exception as e:
            self.logger.error(f"Error calculating base power: {str(e)}")
            return 0.5

    def _calculate_conflict_power_modifiers(
        self,
        participant: str,
        conflict: Conflict
    ) -> List[Dict[str, Any]]:
        """Calculate power modifiers specific to the conflict."""
        try:
            modifiers = []

            participant_stakes = conflict.stakes.get(participant, {})
            stakes_severity = self._evaluate_stakes_severity(participant_stakes)
            modifiers.append({
                "type": "multiply",
                "value": 1.0 + (stakes_severity * 0.2),
                "reason": "Stakes motivation"
            })

            for point in conflict.escalation_points:
                shift_dict = point.get("impact", {}).get("power_shifts", {})
                if participant in shift_dict:
                    power_shift = float(shift_dict[participant])
                    modifiers.append({
                        "type": "add",
                        "value": power_shift,
                        "reason": f"Escalation: {point['trigger']}"
                    })

            return modifiers

        except Exception as e:
            self.logger.error(f"Error calculating conflict power modifiers: {str(e)}")
            return []

    async def _analyze_context_power_modifiers(
        self,
        participant: str,
        conflict: Conflict,
        story_bible: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze power modifiers based on current story context."""
        try:
            modifiers = []

            if "current_location" in story_bible:
                location_data = story_bible.get("locations", {}).get(
                    story_bible["current_location"],
                    {}
                )
                if participant in location_data.get("advantaged_characters", []):
                    modifiers.append({
                        "type": "multiply",
                        "value": 1.2,
                        "reason": "Location advantage"
                    })

            allies = story_bible.get("relationships", {}).get(participant, {}).get("allies", [])
            if allies:
                ally_powers = []
                for ally in allies:
                    ally_char_data = story_bible.get("characters", {}).get(ally, {})
                    ally_powers.append(self._calculate_base_power(ally_char_data))
                if ally_powers:
                    total_ally_power = sum(ally_powers) / len(ally_powers)
                    modifiers.append({
                        "type": "add",
                        "value": total_ally_power * 0.1,
                        "reason": "Alliance support"
                    })

            character_arc = story_bible.get("character_arcs", {}).get(participant, {})
            if character_arc.get("current_stage") == "climax":
                modifiers.append({
                    "type": "multiply",
                    "value": 1.3,
                    "reason": "Character arc climax"
                })

            return modifiers

        except Exception as e:
            self.logger.error(f"Error analyzing context power modifiers: {str(e)}")
            return []

    async def _analyze_power_shifts(
        self,
        conflict: Conflict,
        current_power: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Analyze how power dynamics have shifted throughout the conflict."""
        try:
            shifts = []
            previous_power = {
                participant: 0.5 for participant in conflict.participants
            }

            for point in conflict.escalation_points:
                point_shifts = {}
                for participant in conflict.participants:
                    before = previous_power.get(participant, 0.5)
                    shift_dict = point.get("impact", {}).get("power_shifts", {})
                    if participant in shift_dict:
                        after = max(0.0, min(1.0, before + float(shift_dict[participant])))
                    else:
                        after = before

                    delta = after - before
                    if abs(delta) > 0.1:
                        point_shifts[participant] = delta

                    previous_power[participant] = after

                if point_shifts:
                    shifts.append({
                        "timestamp": point.get("timestamp"),
                        "trigger": point.get("trigger"),
                        "shifts": point_shifts
                    })

            return shifts

        except Exception as e:
            self.logger.error(f"Error analyzing power shifts: {str(e)}")
            return []

    async def _generate_power_balance_recommendations(
        self,
        conflict: Conflict,
        assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for improving power balance."""
        try:
            recommendations = []
            balance_score = assessment["overall_balance"]["balance_score"]

            if balance_score < 0.3:
                recommendations.append({
                    "priority": "high",
                    "type": "balance_adjustment",
                    "description": "Severe power imbalance detected",
                    "suggestions": [
                        {
                            "action": "empower_weaker",
                            "details": "Introduce opportunities or resources for weaker participants",
                            "target_participants": [
                                p for p, val in assessment["participant_power"].items() if val < 0.3
                            ]
                        },
                        {
                            "action": "challenge_stronger",
                            "details": "Create obstacles or limitations for dominant participants",
                            "target_participants": [
                                p for p, val in assessment["participant_power"].items() if val > 0.7
                            ]
                        }
                    ]
                })
            elif balance_score < 0.6:
                # Moderate imbalance (expand or adjust logic as needed)
                recommendations.append({
                    "priority": "moderate",
                    "type": "balance_adjustment",
                    "description": "Moderate power imbalance detected",
                    "suggestions": [
                        {
                            "action": "incremental_support",
                            "details": "Provide slight advantages or alliances to weaker participants",
                            "target_participants": [
                                p for p, val in assessment["participant_power"].items() if 0.3 <= val < 0.5
                            ]
                        },
                        {
                            "action": "encourage_diversification",
                            "details": "Encourage strategies or developments that reduce reliance on a single strong participant",
                            "target_participants": [
                                p for p, val in assessment["participant_power"].items() 
                                if val > 0.7
                            ]
                        }
                    ]
                })
            elif balance_score < 0.6:
                # Moderate imbalance
                recommendations.append({
                    "priority": "medium",
                    "type": "balance_adjustment",
                    "description": "Moderate power imbalance detected",
                    "suggestions": [
                        {
                            "action": "situational_advantage",
                            "details": "Create situational advantages for weaker participants",
                            "target_participants": [
                                p for p, val in assessment["participant_power"].items()
                                if val < 0.4
                            ]
                        },
                        {
                            "action": "temporary_setback",
                            "details": "Introduce temporary setbacks for stronger participants",
                            "target_participants": [
                                p for p, val in assessment["participant_power"].items()
                                if val > 0.6
                            ]
                        }
                    ]
                })

            # Check for stagnation
            if len(assessment["power_shifts"]) < len(conflict.escalation_points) / 2:
                recommendations.append({
                    "priority": "medium",
                    "type": "dynamic_adjustment",
                    "description": "Limited power dynamics detected",
                    "suggestions": [
                        {
                            "action": "introduce_catalyst",
                            "details": "Add events that force power dynamic changes"
                        },
                        {
                            "action": "resource_shift",
                            "details": "Redistribute resources or advantages among participants"
                        }
                    ]
                })

            # Check for excessive volatility
            rapid_shifts = sum(
                1 for shift in assessment["power_shifts"]
                if any(abs(s["shift"]) > 0.3 for s in shift["shifts"].values())
            )
            if rapid_shifts > len(assessment["power_shifts"]) / 3:
                recommendations.append({
                    "priority": "medium",
                    "type": "stability_adjustment",
                    "description": "Excessive power volatility detected",
                    "suggestions": [
                        {
                            "action": "stabilize_changes",
                            "details": "Add elements that help stabilize rapid power shifts"
                        },
                        {
                            "action": "gradual_progression",
                            "details": "Plan more gradual power progression"
                        }
                    ]
                })

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating power balance recommendations: {str(e)}")
            return []

    async def _evaluate_resolutions(
        self,
        conflict: Conflict,
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate the feasibility and impact of possible resolutions."""
        try:
            evaluation = {
                "timestamp": datetime.utcnow().isoformat(),
                "resolution_options": [],
                "optimal_resolution": None,
                "risks": [],
                "requirements": []
            }

            # Evaluate each resolution option
            for option in conflict.resolution_options:
                option_evaluation = await self._evaluate_resolution_option(
                    option,
                    conflict,
                    story_bible
                )
                evaluation["resolution_options"].append(option_evaluation)

            # Determine optimal resolution
            if evaluation["resolution_options"]:
                optimal = max(
                    evaluation["resolution_options"],
                    key=lambda x: x["feasibility_score"] * x["satisfaction_score"]
                )
                evaluation["optimal_resolution"] = optimal

            # Identify common risks and requirements
            evaluation["risks"] = await self._identify_resolution_risks(
                conflict,
                evaluation["resolution_options"]
            )
            evaluation["requirements"] = await self._identify_resolution_requirements(
                conflict,
                evaluation["resolution_options"]
            )

            return evaluation

        except Exception as e:
            self.logger.error(f"Error evaluating resolutions: {str(e)}")
            raise

    async def _evaluate_resolution_option(
        self,
        option: Dict[str, Any],
        conflict: Conflict,
        story_bible: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single resolution option."""
        try:
            evaluation = {
                "option": option,
                "feasibility_score": 0.0,
                "satisfaction_score": 0.0,
                "character_impact": {},
                "plot_alignment": 0.0,
                "theme_alignment": 0.0,
                "risks": [],
                "requirements": []
            }

            # Calculate feasibility score
            evaluation["feasibility_score"] = await self._calculate_feasibility_score(
                option,
                conflict,
                story_bible
            )

            # Calculate satisfaction score
            evaluation["satisfaction_score"] = await self._calculate_satisfaction_score(
                option,
                conflict,
                story_bible
            )

            # Analyze character impact
            for participant in conflict.participants:
                evaluation["character_impact"][participant] = \
                    await self._analyze_resolution_character_impact(
                        participant,
                        option,
                        conflict,
                        story_bible
                    )

            # Calculate plot and theme alignment
            evaluation["plot_alignment"] = await self._calculate_plot_alignment(
                option,
                conflict,
                story_bible
            )
            evaluation["theme_alignment"] = await self._calculate_theme_alignment(
                option,
                story_bible
            )

            # Identify specific risks and requirements
            evaluation["risks"] = await self._identify_option_risks(
                option,
                conflict,
                story_bible
            )
            evaluation["requirements"] = await self._identify_option_requirements(
                option,
                conflict,
                story_bible
            )

            return evaluation

        except Exception as e:
            self.logger.error(f"Error evaluating resolution option: {str(e)}")
            raise

    async def _analyze_immediate_character_impact(
        self,
        character: str,
        option: Dict[str, Any],
        conflict: Conflict
    ) -> Dict[str, Any]:
        """Analyze the immediate impact of a resolution on a character."""
        try:
            impact = {
                "emotional_impact": await self._analyze_emotional_impact(
                    character,
                    option,
                    conflict
                ),
                "status_change": await self._analyze_status_change(
                    character,
                    option,
                    conflict
                ),
                "relationship_effects": await self._analyze_relationship_effects(
                    character,
                    option,
                    conflict
                )
            }
            return impact

        except Exception as e:
            self.logger.error(f"Error analyzing immediate character impact: {str(e)}")
            raise

    async def _analyze_emotional_impact(
        self,
        character: str,
        option: Dict[str, Any],
        conflict: Conflict
    ) -> Dict[str, Any]:
        """Analyze the emotional impact of a resolution on a character."""
        try:
            # Get character's stakes and motivations
            stakes = conflict.stakes.get(character, {})
            motivations = conflict.motivations.get(character, [])

            # Analyze how resolution affects stakes
            stakes_impact = self._evaluate_stakes_impact(
                stakes,
                option.get("consequences", {})
            )

            # Analyze motivation satisfaction
            motivation_satisfaction = self._evaluate_motivation_satisfaction(
                motivations,
                option.get("outcomes", {})
            )

            return {
                "stakes_impact": stakes_impact,
                "motivation_satisfaction": motivation_satisfaction,
                "overall_emotional_state": self._calculate_emotional_state(
                    stakes_impact,
                    motivation_satisfaction
                )
            }

        except Exception as e:
            self.logger.error(f"Error analyzing emotional impact: {str(e)}")
            raise

    def _evaluate_stakes_impact(
        self,
        stakes: Dict[str, Any],
        consequences: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate how resolution consequences affect character stakes."""
        try:
            impact = {}
            for stake, value in stakes.items():
                if stake in consequences:
                    impact[stake] = self._calculate_stake_impact(
                        value,
                        consequences[stake]
                    )
            return impact

        except Exception as e:
            self.logger.error(f"Error evaluating stakes impact: {str(e)}")
            return {}

    def _calculate_stake_impact(
        self,
        stake_value: Any,
        consequence: Any
    ) -> float:
        """Calculate the impact of a consequence on a stake."""
        try:
            # Convert to comparable values
            if isinstance(stake_value, bool):
                stake_value = 1.0 if stake_value else 0.0
            elif isinstance(stake_value, str):
                stake_value = self._evaluate_textual_value(stake_value)
            elif isinstance(stake_value, (int, float)):
                stake_value = float(stake_value)
            else:
                stake_value = 0.5

            if isinstance(consequence, bool):
                consequence = 1.0 if consequence else 0.0
            elif isinstance(consequence, str):
                consequence = self._evaluate_textual_value(consequence)
            elif isinstance(consequence, (int, float)):
                consequence = float(consequence)
            else:
                consequence = 0.5

            # Calculate impact (-1.0 to 1.0)
            return consequence - stake_value

        except Exception as e:
            self.logger.error(f"Error calculating stake impact: {str(e)}")
            return 0.0

    def _evaluate_textual_value(self, text: str) -> float:
        """Convert textual descriptions to numerical values."""
        try:
            # Define value mappings
            positive_terms = {
                "critical": 1.0,
                "vital": 0.9,
                "important": 0.8,
                "significant": 0.7,
                "moderate": 0.5,
                "minor": 0.3,
                "minimal": 0.2,
                "negligible": 0.1
            }

            text_lower = text.lower()
            
            # Check for negation
            is_negative = any(neg in text_lower for neg in ["not", "never", "none"])
            
            # Find highest matching value
            base_value = 0.5  # default
            for term, value in positive_terms.items():
                if term in text_lower:
                    base_value = value
                    break
            
            return -base_value if is_negative else base_value

        except Exception as e:
            self.logger.error(f"Error evaluating textual value: {str(e)}")
            return 0.5