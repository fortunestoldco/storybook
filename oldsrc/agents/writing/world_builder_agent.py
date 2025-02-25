# Complete the _are_relationship_types_compatible method
def _are_relationship_types_compatible(
    self,
    type1: str,
    type2: str
) -> bool:
    """Check if two relationship types are logically compatible."""
    try:
        # Define compatible relationship pairs
        compatible_pairs = {
            'contains': 'part_of',
            'parent': 'child',
            'ally': 'ally',
            'enemy': 'enemy',
            'ruler': 'subject',
            'creator': 'created_by',
            'teacher': 'student',
            'connected_to': 'connected_to',
            'influences': 'influenced_by',
            'predecessor': 'successor'
        }

        # Add reverse relationships
        reverse_pairs = {v: k for k, v in compatible_pairs.items()}
        compatible_pairs.update(reverse_pairs)

        # Check for symmetric relationships
        symmetric_relationships = {
            'ally', 'enemy', 'connected_to', 'sibling', 'spouse'
        }
        
        if type1 in symmetric_relationships and type2 in symmetric_relationships:
            return type1 == type2

        return compatible_pairs.get(type1) == type2

    except Exception as e:
        self.logger.error(f"Error checking relationship compatibility: {str(e)}")
        return False

# Add temporal analysis methods
async def _analyze_temporal_relationships(
    self,
    event: Dict[str, Any],
    story_bible: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze temporal relationships between world elements."""
    try:
        temporal_analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "concurrent_events": [],
            "causal_chains": [],
            "temporal_anomalies": [],
            "timeline_conflicts": []
        }

        # Find concurrent events
        temporal_analysis["concurrent_events"] = await self._find_concurrent_events(
            event,
            story_bible
        )

        # Analyze causal relationships
        temporal_analysis["causal_chains"] = await self._analyze_causal_chains(
            event,
            story_bible
        )

        # Check for temporal anomalies
        temporal_analysis["temporal_anomalies"] = await self._check_temporal_anomalies(
            event,
            story_bible
        )

        # Validate timeline consistency
        temporal_analysis["timeline_conflicts"] = await self._validate_timeline_consistency(
            event,
            story_bible
        )

        return temporal_analysis

    except Exception as e:
        self.logger.error(f"Error analyzing temporal relationships: {str(e)}")
        raise

async def _find_concurrent_events(
    self,
    event: Dict[str, Any],
    story_bible: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Find events that occur concurrently with the given event."""
    try:
        concurrent_events = []
        event_time = event.get("timestamp")
        
        if not event_time:
            return concurrent_events

        # Get all events from story bible
        all_events = story_bible.get("events", [])
        
        for other_event in all_events:
            if other_event.get("id") == event.get("id"):
                continue
                
            other_time = other_event.get("timestamp")
            if not other_time:
                continue

            # Check for temporal overlap
            if self._events_overlap(event_time, other_time):
                concurrent_events.append({
                    "event_id": other_event.get("id"),
                    "overlap_type": self._determine_overlap_type(event_time, other_time),
                    "overlap_duration": self._calculate_overlap_duration(event_time, other_time)
                })

        return concurrent_events

    except Exception as e:
        self.logger.error(f"Error finding concurrent events: {str(e)}")
        return []

def _events_overlap(
    self,
    time1: Dict[str, Any],
    time2: Dict[str, Any]
) -> bool:
    """Check if two events overlap in time."""
    try:
        # Handle different time formats
        if isinstance(time1, (str, int, float)) and isinstance(time2, (str, int, float)):
            return time1 == time2

        # Handle time ranges
        start1 = time1.get("start")
        end1 = time1.get("end")
        start2 = time2.get("start")
        end2 = time2.get("end")

        if all([start1, end1, start2, end2]):
            return not (end1 < start2 or start1 > end2)

        return False

    except Exception as e:
        self.logger.error(f"Error checking event overlap: {str(e)}")
        return False

def _determine_overlap_type(
    self,
    time1: Dict[str, Any],
    time2: Dict[str, Any]
) -> str:
    """Determine the type of temporal overlap between events."""
    try:
        if isinstance(time1, (str, int, float)) or isinstance(time2, (str, int, float)):
            return "exact" if time1 == time2 else "none"

        start1 = time1.get("start")
        end1 = time1.get("end")
        start2 = time2.get("start")
        end2 = time2.get("end")

        if not all([start1, end1, start2, end2]):
            return "undetermined"

        if start1 == start2 and end1 == end2:
            return "exact"
        elif start1 <= start2 and end1 >= end2:
            return "contains"
        elif start2 <= start1 and end2 >= end1:
            return "contained"
        elif start1 <= start2 and end1 <= end2 and end1 >= start2:
            return "overlaps_start"
        elif start2 <= start1 and end2 <= end1 and end2 >= start1:
            return "overlaps_end"
        else:
            return "none"

    except Exception as e:
        self.logger.error(f"Error determining overlap type: {str(e)}")
        return "error"

def _calculate_overlap_duration(
    self,
    time1: Dict[str, Any],
    time2: Dict[str, Any]
) -> Optional[float]:
    """Calculate the duration of overlap between events."""
    try:
        if isinstance(time1, (str, int, float)) or isinstance(time2, (str, int, float)):
            return 0.0 if time1 == time2 else None

        start1 = time1.get("start")
        end1 = time1.get("end")
        start2 = time2.get("start")
        end2 = time2.get("end")

        if not all([start1, end1, start2, end2]):
            return None

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_start <= overlap_end:
            return overlap_end - overlap_start
        return None

    except Exception as e:
        self.logger.error(f"Error calculating overlap duration: {str(e)}")
        return None

async def _analyze_causal_chains(
    self,
    event: Dict[str, Any],
    story_bible: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Analyze causal relationships between events."""
    try:
        causal_chains = []
        event_id = event.get("id")
        
        if not event_id:
            return causal_chains

        # Get all events from story bible
        all_events = story_bible.get("events", [])
        
        # Build cause-effect relationships
        for other_event in all_events:
            if other_event.get("id") == event_id:
                continue

            relationship = await self._determine_causal_relationship(
                event,
                other_event
            )
            
            if relationship:
                causal_chains.append({
                    "event_id": other_event.get("id"),
                    "relationship_type": relationship["type"],
                    "confidence": relationship["confidence"],
                    "evidence": relationship["evidence"]
                })

        return causal_chains

    except Exception as e:
        self.logger.error(f"Error analyzing causal chains: {str(e)}")
        return []

async def _determine_causal_relationship(
    self,
    event1: Dict[str, Any],
    event2: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Determine the causal relationship between two events."""
    try:
        # Check temporal order
        if not self._is_temporally_possible(event1, event2):
            return None

        # Analyze event properties for causal links
        causes = set(event1.get("effects", []))
        prerequisites = set(event2.get("prerequisites", []))
        
        # Calculate overlap
        common_elements = causes.intersection(prerequisites)
        
        if not common_elements:
            return None

        # Calculate confidence based on evidence
        confidence = self._calculate_causal_confidence(
            common_elements,
            event1,
            event2
        )

        return {
            "type": "causes" if confidence > 0.5 else "influences",
            "confidence": confidence,
            "evidence": list(common_elements)
        }

    except Exception as e:
        self.logger.error(f"Error determining causal relationship: {str(e)}")
        return None

def _calculate_causal_confidence(
    self,
    common_elements: Set[str],
    event1: Dict[str, Any],
    event2: Dict[str, Any]
) -> float:
    """Calculate confidence in a causal relationship."""
    try:
        confidence = 0.0
        
        # Base confidence from number of common elements
        confidence += len(common_elements) * 0.2

        # Adjust for temporal proximity
        temporal_factor = self._calculate_temporal_proximity(event1, event2)
        confidence += temporal_factor * 0.3

        # Adjust for direct references
        if event2.get("caused_by") == event1.get("id"):
            confidence += 0.3

        # Adjust for shared participants
        participant_overlap = self._calculate_participant_overlap(event1, event2)
        confidence += participant_overlap * 0.2

        return min(1.0, max(0.0, confidence))

    except Exception as e:
        self.logger.error(f"Error calculating causal confidence: {str(e)}")
        return 0.0

def _calculate_temporal_proximity(
    self,
    event1: Dict[str, Any],
    event2: Dict[str, Any]
) -> float:
    """Calculate temporal proximity between events."""
    try:
        time1 = event1.get("timestamp")
        time2 = event2.get("timestamp")
        
        if not (time1 and time2):
            return 0.0

        # Handle different time formats
        if isinstance(time1, (str, int, float)) and isinstance(time2, (str, int, float)):
            return 1.0 if time1 == time2 else 0.0

        # Calculate proximity for time ranges
        end1 = time1.get("end")
        start2 = time2.get("start")
        
        if not (end1 and start2):
            return 0.0

        time_diff = abs(start2 - end1)
        max_diff = 100.0  # Adjust based on your time scale
        
        return max(0.0, 1.0 - (time_diff / max_diff))

    except Exception as e:
        self.logger.error(f"Error calculating temporal proximity: {str(e)}")
        return 0.0

def _calculate_participant_overlap(
    self,
    event1: Dict[str, Any],
    event2: Dict[str, Any]
) -> float:
    """Calculate overlap in event participants."""
    try:
        participants1 = set(event1.get("participants", []))
        participants2 = set(event2.get("participants", []))
        
        if not (participants1 and participants2):
            return 0.0

        overlap = len(participants1.intersection(participants2))
        total = len(participants1.union(participants2))
        
        return overlap / total if total > 0 else 0.0

    except Exception as e:
        self.logger.error(f"Error calculating participant overlap: {str(e)}")
        return 0.0