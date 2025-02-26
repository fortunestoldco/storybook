# storybook/innovations/voice/adaptive_voice.py

"""
Adaptive Voice Resonance Framework for developing and maintaining distinctive authorial voice.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import uuid

from storybook.tools.nlp.voice_analyzer import VoiceAnalyzer

class VoiceMemoryBank:
    """Maintains a neural representation of the evolving authorial voice."""
    
    def __init__(self):
        """Initialize the voice memory bank."""
        self.core_patterns = {}
        self.adaptations = []
        self.voice_embedding = None
        self.successful_passages = []
        self.pattern_weights = defaultdict(float)
        
    def update_from_successful_passages(
        self, 
        passages: List[Dict[str, Any]], 
        feedback_scores: Dict[str, float]
    ) -> None:
        """
        Update voice memory using successful passages with high feedback scores.
        
        Args:
            passages: List of successful passages with their metadata
            feedback_scores: Dictionary mapping passage IDs to feedback scores
        """
        # Store successful passages
        for passage in passages:
            if passage["id"] in feedback_scores:
                self.successful_passages.append({
                    "text": passage["text"],
                    "score": feedback_scores[passage["id"]],
                    "context": passage.get("context", {}),
                    "id": passage["id"]
                })
        
        # Extract patterns from successful passages
        new_patterns = self._extract_patterns(passages, feedback_scores)
        
        # Update core patterns
        for pattern_type, patterns in new_patterns.items():
            if pattern_type not in self.core_patterns:
                self.core_patterns[pattern_type] = {}
                
            for pattern, stats in patterns.items():
                if pattern in self.core_patterns[pattern_type]:
                    # Update existing pattern
                    self.core_patterns[pattern_type][pattern]["count"] += stats["count"]
                    self.core_patterns[pattern_type][pattern]["total_score"] += stats["total_score"]
                    self.core_patterns[pattern_type][pattern]["contexts"].update(stats["contexts"])
                else:
                    # Add new pattern
                    self.core_patterns[pattern_type][pattern] = stats
                    
                # Update pattern weight
                avg_score = stats["total_score"] / max(1, stats["count"])
                self.pattern_weights[(pattern_type, pattern)] = avg_score
    
    def retrieve_voice_guidance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve voice patterns most appropriate for the given context.
        
        Args:
            context: The current context
            
        Returns:
            Voice guidance including patterns and examples
        """
        # Extract context features for matching
        context_type = context.get("type", "")
        tone = context.get("tone", "")
        pov = context.get("pov", "")
        
        # Find matching patterns
        matching_patterns = {}
        for pattern_type, patterns in self.core_patterns.items():
            matching_patterns[pattern_type] = []
            
            for pattern, stats in patterns.items():
                # Check context match
                context_match = False
                for ctx in stats["contexts"]:
                    if (
                        (not context_type or ctx.get("type") == context_type) and
                        (not tone or ctx.get("tone") == tone) and
                        (not pov or ctx.get("pov") == pov)
                    ):
                        context_match = True
                        break
                        
                if context_match or not stats["contexts"]:
                    # Get the average score for this pattern
                    avg_score = stats["total_score"] / max(1, stats["count"])
                    
                    matching_patterns[pattern_type].append({
                        "pattern": pattern,
                        "score": avg_score,
                        "count": stats["count"]
                    })
            
            # Sort patterns by score
            matching_patterns[pattern_type].sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to top patterns
            matching_patterns[pattern_type] = matching_patterns[pattern_type][:5]
        
        # Find examples that match the context
        examples = self._find_matching_examples(context)
        
        return {
            "patterns": matching_patterns,
            "examples": examples,
            "voice_profile": self._generate_voice_profile()
        }
        
    def _extract_patterns(
        self, 
        passages: List[Dict[str, Any]], 
        feedback_scores: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract patterns from passages.
        
        Args:
            passages: List of passages
            feedback_scores: Dictionary mapping passage IDs to feedback scores
            
        Returns:
            Dictionary of patterns
        """
        patterns = {
            "sentence_structure": {},
            "transition": {},
            "imagery": {},
            "dialogue": {}
        }
        
        for passage in passages:
            if passage["id"] not in feedback_scores:
                continue
                
            score = feedback_scores[passage["id"]]
            
            # Extract sentence structure patterns (simplified)
            sentences = passage["text"].split(".")
            for sentence in sentences:
                if len(sentence.strip()) < 5:
                    continue
                    
                # Simplified pattern: first 3 words
                words = sentence.strip().split()
                if len(words) >= 3:
                    pattern = " ".join(words[:3]).lower()
                    self._add_pattern(
                        patterns, "sentence_structure", pattern, 
                        score, passage.get("context", {})
                    )
                    
            # Extract transition patterns
            # (simplified - in real implementation would be more sophisticated)
            transition_words = ["however", "therefore", "meanwhile", "nevertheless", "furthermore"]
            for word in transition_words:
                if word in passage["text"].lower():
                    self._add_pattern(
                        patterns, "transition", word, 
                        score, passage.get("context", {})
                    )
                    
            # More pattern extraction would be implemented here
            
        return patterns
    
    def _add_pattern(
        self, 
        patterns: Dict[str, Dict[str, Any]], 
        pattern_type: str, 
        pattern: str, 
        score: float, 
        context: Dict[str, Any]
    ) -> None:
        """
        Add a pattern to the patterns dictionary.
        
        Args:
            patterns: Patterns dictionary
            pattern_type: Type of pattern
            pattern: The pattern
            score: Feedback score
            context: Context information
        """
        if pattern not in patterns[pattern_type]:
            patterns[pattern_type][pattern] = {
                "count": 0,
                "total_score": 0,
                "contexts": set()
            }
            
        patterns[pattern_type][pattern]["count"] += 1
        patterns[pattern_type][pattern]["total_score"] += score
        
        # Add context as a frozenset of items for hashability
        context_items = frozenset(context.items())
        patterns[pattern_type][pattern]["contexts"].add(context_items)
    
    def _find_matching_examples(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find examples that match the given context.
        
        Args:
            context: The context to match
            
        Returns:
            List of matching examples
        """
        context_type = context.get("type", "")
        tone = context.get("tone", "")
        pov = context.get("pov", "")
        
        matching_examples = []
        for passage in self.successful_passages:
            passage_context = passage.get("context", {})
            
            # Check context match
            if (
                (not context_type or passage_context.get("type") == context_type) and
                (not tone or passage_context.get("tone") == tone) and
                (not pov or passage_context.get("pov") == pov)
            ):
                matching_examples.append({
                    "text": passage["text"],
                    "score": passage["score"]
                })
                
        # Sort by score and limit
        matching_examples.sort(key=lambda x: x["score"], reverse=True)
        return matching_examples[:3]
    
    def _generate_voice_profile(self) -> Dict[str, Any]:
        """
        Generate a voice profile summarizing the voice characteristics.
        
        Returns:
            Voice profile
        """
        profile = {
            "patterns": {},
            "tendencies": {}
        }
        
        # Count patterns by type
        pattern_counts = defaultdict(int)
        for pattern_type, patterns in self.core_patterns.items():
            pattern_counts[pattern_type] = len(patterns)
            
            # Add top patterns to profile
            profile["patterns"][pattern_type] = []
            sorted_patterns = sorted(
                patterns.items(),
                key=lambda x: x[1]["total_score"] / max(1, x[1]["count"]),
                reverse=True
            )
            
            for pattern, stats in sorted_patterns[:3]:
                profile["patterns"][pattern_type].append({
                    "pattern": pattern,
                    "score": stats["total_score"] / max(1, stats["count"])
                })
        
        # Calculate tendencies
        if self.successful_passages:
            avg_sentence_length = np.mean([
                len(p["text"].split("."))
                for p in self.successful_passages
            ])
            
            avg_word_length = np.mean([
                len(word)
                for p in self.successful_passages
                for word in p["text"].split()
            ])
            
            profile["tendencies"] = {
                "avg_sentence_length": float(avg_sentence_length),
                "avg_word_length": float(avg_word_length),
                "pattern_diversity": sum(pattern_counts.values()) / max(1, len(self.successful_passages))
            }
        
        return profile


class VoiceModulator:
    """Dynamically modulates voice while maintaining core identity."""
    
    def __init__(self, voice_analyzer: Optional[VoiceAnalyzer] = None):
        """
        Initialize the voice modulator.
        
        Args:
            voice_analyzer: Optional voice analyzer
        """
        self.voice_analyzer = voice_analyzer
        
    def modulate_for_context(
        self, 
        base_voice: Dict[str, Any], 
        context_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt voice for specific context while preserving identity.
        
        Args:
            base_voice: Base voice profile
            context_requirements: Context requirements for modulation
            
        Returns:
            Modulated voice guidance
        """
        # Start with base voice
        modulated_voice = base_voice.copy()
        
        # Adjust for tone
        target_tone = context_requirements.get("tone", "")
        if target_tone:
            self._adjust_for_tone(modulated_voice, target_tone)
            
        # Adjust for POV
        target_pov = context_requirements.get("pov", "")
        if target_pov:
            self._adjust_for_pov(modulated_voice, target_pov)
            
        # Adjust for intensity
        target_intensity = context_requirements.get("intensity", 0.5)
        self._adjust_for_intensity(modulated_voice, target_intensity)
            
        return modulated_voice
    
    def ensure_uniqueness(
        self, 
        voice_pattern: Dict[str, Any], 
        market_examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ensure voice remains distinctive compared to market examples.
        
        Args:
            voice_pattern: Voice pattern to check
            market_examples: Market examples to compare against
            
        Returns:
            Updated voice pattern with uniqueness scores
        """
        if not self.voice_analyzer or not market_examples:
            return voice_pattern
            
        # Analyze market examples
        market_voices = []
        for example in market_examples:
            if "text" in example:
                voice_analysis = self.voice_analyzer.analyze_voice(example["text"])
                market_voices.append(voice_analysis)
                
        if not market_voices:
            return voice_pattern
            
        # Compare with market voices
        uniqueness_scores = {}
        for feature in voice_pattern.get("voice_profile", {}).get("tendencies", {}):
            if feature in voice_pattern["voice_profile"]["tendencies"]:
                feature_value = voice_pattern["voice_profile"]["tendencies"][feature]
                market_values = [
                    v.get("stylistic_features", {}).get(feature, 0)
                    for v in market_voices
                    if feature in v.get("stylistic_features", {})
                ]
                
                if market_values:
                    # Calculate how different this voice is from market average
                    market_avg = np.mean(market_values)
                    uniqueness = abs(feature_value - market_avg) / max(1, market_avg)
                    uniqueness_scores[feature] = float(uniqueness)
        
        # Add uniqueness scores to voice pattern
        voice_pattern["uniqueness_scores"] = uniqueness_scores
        voice_pattern["overall_uniqueness"] = np.mean(list(uniqueness_scores.values())) if uniqueness_scores else 0
        
        return voice_pattern
    
    def _adjust_for_tone(self, voice: Dict[str, Any], target_tone: str) -> None:
        """
        Adjust voice for target tone.
        
        Args:
            voice: Voice to adjust
            target_tone: Target tone
        """
        # Adjust pattern weights for the target tone
        for pattern_type in voice.get("patterns", {}):
            for i, pattern in enumerate(voice["patterns"][pattern_type]):
                # Simplified logic - in real implementation would be more sophisticated
                if target_tone == "formal":
                    # Emphasize formal patterns
                    if "therefore" in pattern["pattern"] or "however" in pattern["pattern"]:
                        pattern["score"] *= 1.2
                elif target_tone == "casual":
                    # Emphasize casual patterns
                    if "but" in pattern["pattern"] or "so" in pattern["pattern"]:
                        pattern["score"] *= 1.2
                        
            # Re-sort patterns based on adjusted scores
            voice["patterns"][pattern_type].sort(key=lambda x: x["score"], reverse=True)
    
    def _adjust_for_pov(self, voice: Dict[str, Any], target_pov: str) -> None:
        """
        Adjust voice for target point of view.
        
        Args:
            voice: Voice to adjust
            target_pov: Target POV
        """
        # Add POV guidance
        voice["pov_guidance"] = {
            "target_pov": target_pov,
            "recommendations": []
        }
        
        if target_pov == "first_person":
            voice["pov_guidance"]["recommendations"] = [
                "Use 'I' and 'we' consistently",
                "Convey subjective experiences directly",
                "Filter all observations through the narrator's perspective"
            ]
        elif target_pov == "third_person_limited":
            voice["pov_guidance"]["recommendations"] = [
                "Stay close to the viewpoint character's perspective",
                "Use 'he/she/they' consistently",
                "Only reveal what the viewpoint character would know"
            ]
        elif target_pov == "third_person_omniscient":
            voice["pov_guidance"]["recommendations"] = [
                "Move between characters' perspectives smoothly",
                "Use narrative voice for insights unavailable to characters",
                "Maintain consistent narrative distance"
            ]
    
    def _adjust_for_intensity(self, voice: Dict[str, Any], target_intensity: float) -> None:
        """
        Adjust voice for target emotional intensity.
        
        Args:
            voice: Voice to adjust
            target_intensity: Target intensity (0.0 to 1.0)
        """
        # Add intensity guidance
        voice["intensity_guidance"] = {
            "target_intensity": target_intensity,
            "sentence_length": "shorter" if target_intensity > 0.7 else "varied",
            "recommended_patterns": []
        }
        
        if target_intensity > 0.7:
            voice["intensity_guidance"]["recommended_patterns"] = [
                "Use sharp, vivid images",
                "Favor direct, forceful verbs",
                "Employ short, punchy sentences at key moments"
            ]
        elif target_intensity < 0.3:
            voice["intensity_guidance"]["recommended_patterns"] = [
                "Use flowing, lyrical descriptions",
                "Employ more complex sentence structures",
                "Allow for meandering observations where appropriate"
            ]
        else:
            voice["intensity_guidance"]["recommended_patterns"] = [
                "Balance vivid imagery with measured pacing",
                "Vary sentence length and structure for natural rhythm",
                "Intensify language selectively at emotional high points"
            ]


class ReaderFeedbackProcessor:
    """Processes reader feedback on voice and style."""
    
    def process(self, feedback: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Process reader feedback.
        
        Args:
            feedback: List of feedback items
            
        Returns:
            Dictionary mapping passage IDs to feedback scores
        """
        scores = {}
        
        for item in feedback:
            passage_id = item.get("passage_id")
            if not passage_id:
                continue
                
            # Calculate score based on feedback
            score = self._calculate_score(item)
            scores[passage_id] = score
            
        return scores
    
    def _calculate_score(self, feedback_item: Dict[str, Any]) -> float:
        """
        Calculate score for a feedback item.
        
        Args:
            feedback_item: Feedback item
            
        Returns:
            Score between 0 and 1
        """
        # Get explicit ratings if available
        if "rating" in feedback_item:
            rating = feedback_item["rating"]
            if isinstance(rating, (int, float)):
                # Normalize to 0-1
                return min(1.0, max(0.0, rating / 10.0))
                
        # Otherwise extract sentiment from text
        sentiment = feedback_item.get("sentiment", 0.5)
        return sentiment


class AdaptiveVoiceResonance:
    """Dynamic system for developing and maintaining distinctive authorial voice."""
    
    def __init__(self, voice_analyzer: Optional[VoiceAnalyzer] = None):
        """
        Initialize the adaptive voice resonance system.
        
        Args:
            voice_analyzer: Optional voice analyzer
        """
        self.voice_memory = VoiceMemoryBank()
        self.voice_modulator = VoiceModulator(voice_analyzer)
        self.reader_feedback_processor = ReaderFeedbackProcessor()
        
    def evolve_voice(
        self, 
        manuscript_progress: Dict[str, Any], 
        reader_feedback: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evolve the voice based on manuscript progress and feedback.
        
        Args:
            manuscript_progress: Dictionary with manuscript progress information
            reader_feedback: List of reader feedback items
            
        Returns:
            Evolved voice guidance
        """
        # Process reader feedback
        processed_feedback = self.reader_feedback_processor.process(reader_feedback)
        
        # Identify successful passages
        successful_passages = self._identify_successful_passages(
            manuscript_progress, processed_feedback)
            
        # Update voice memory
        self.voice_memory.update_from_successful_passages(
            successful_passages, processed_feedback)
            
        # Generate evolved voice guidance
        voice_guidance = self.voice_memory.retrieve_voice_guidance(
            manuscript_progress.get("current_context", {}))
            
        # Modulate voice for current context
        modulated_voice = self.voice_modulator.modulate_for_context(
            voice_guidance,
            manuscript_progress.get("current_context", {})
        )
        
        # Ensure uniqueness compared to market examples
        if "market_examples" in manuscript_progress:
            modulated_voice = self.voice_modulator.ensure_uniqueness(
                modulated_voice,
                manuscript_progress.get("market_examples", [])
            )
            
        return modulated_voice
    
    def _identify_successful_passages(
        self, 
        manuscript_progress: Dict[str, Any], 
        processed_feedback: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Identify successful passages based on feedback.
        
        Args:
            manuscript_progress: Manuscript progress information
            processed_feedback: Processed feedback scores
            
        Returns:
            List of successful passages
        """
        # Get passages from the manuscript
        passages = []
        
        # Get chapters
        chapters = manuscript_progress.get("chapters", [])
        
        for chapter in chapters:
            # Split chapter into passages (paragraphs)
            chapter_text = chapter.get("content", "")
            paragraphs = chapter_text.split("\n\n")
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 50:
                    continue
                    
                passage_id = f"{chapter.get('id', 'unknown')}_p{i}"
                
                passages.append({
                    "id": passage_id,
                    "text": paragraph,
                    "context": {
                        "chapter_id": chapter.get("id"),
                        "chapter_title": chapter.get("title"),
                        "type": "paragraph",
                        "pov": chapter.get("pov", "unknown"),
                        "tone": chapter.get("tone", "unknown")
                    }
                })
                
            # Also add dialogue passages
            import re
            dialogue_pattern = r'"[^"]+"|'[^']+'|"[^"]+"'
            dialogues = re.findall(dialogue_pattern, chapter_text)
            
            for i, dialogue in enumerate(dialogues):
                if len(dialogue.strip()) < 20:
                    continue
                    
                passage_id = f"{chapter.get('id', 'unknown')}_d{i}"
                
                passages.append({
                    "id": passage_id,
                    "text": dialogue,
                    "context": {
                        "chapter_id": chapter.get("id"),
                        "chapter_title": chapter.get("title"),
                        "type": "dialogue",
                        "pov": chapter.get("pov", "unknown"),
                        "tone": chapter.get("tone", "unknown")
                    }
                })
        
        # Filter for successful passages
        threshold = 0.7  # Success threshold
        successful_passages = []
        
        for passage in passages:
            # Check if passage has direct feedback
            if passage["id"] in processed_feedback:
                score = processed_feedback[passage["id"]]
                if score >= threshold:
                    successful_passages.append(passage)
                    
            # For passages without direct feedback, use chapter-level feedback
            elif passage["context"]["chapter_id"] in processed_feedback:
                score = processed_feedback[passage["context"]["chapter_id"]]
                if score >= threshold:
                    # Use a slightly reduced score for indirect feedback
                    processed_feedback[passage["id"]] = score * 0.9
                    successful_passages.append(passage)
        
        return successful_passages
