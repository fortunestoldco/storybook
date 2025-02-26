# storybook/tools/nlp/minilm_analyzer.py

"""
MiniLM-based text analysis tool for evaluating generated content.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class MiniLMAnalyzer:
    """Core MiniLM-based text analysis tool."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the MiniLM analyzer.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self) -> SentenceTransformer:
        """
        Load the MiniLM model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        try:
            return SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load MiniLM model: {e}")
        
    def analyze_output(self, text: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze text based on specified criteria.
        
        Args:
            text: The text to analyze
            criteria: Dictionary of criteria to evaluate against
            
        Returns:
            Analysis results
        """
        results = {}
        scores = []
        weights = []
        
        # Analyze each criterion
        for criterion, parameters in criteria.items():
            weight = parameters.get("weight", 1.0)
            if criterion == "emotional_impact":
                results[criterion] = self.analyze_emotional_impact(text, parameters)
            elif criterion == "voice_consistency":
                results[criterion] = self.analyze_voice_consistency(text, parameters)
            elif criterion == "thematic_coherence":
                results[criterion] = self.analyze_thematic_coherence(text, parameters)
            elif criterion == "relevance":
                results[criterion] = self.analyze_relevance(text, parameters)
            elif criterion == "completeness":
                results[criterion] = self.analyze_completeness(text, parameters)
            elif criterion == "coherence":
                results[criterion] = self.analyze_coherence(text, parameters)
            else:
                # Generic criterion
                results[criterion] = self.analyze_generic_criterion(text, parameters)
            
            scores.append(results[criterion]["score"])
            weights.append(weight)
        
        # Calculate overall weighted score
        if scores:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
            
        return {
            "criteria_results": results,
            "overall_score": float(overall_score),
            "message": self._generate_analysis_message(results, overall_score)
        }
    
    def analyze_emotional_impact(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze emotional impact of text.
        
        Args:
            text: The text to analyze
            parameters: Parameters for emotional impact analysis
            
        Returns:
            Analysis results
        """
        # Placeholder implementation
        target_emotion = parameters.get("target_emotion", "")
        intensity_level = parameters.get("intensity_level", 0.5)
        
        # In a real implementation, we would analyze the emotional content more thoroughly
        score = 0.75  # Placeholder score
        
        return {
            "score": score,
            "details": {
                "target_emotion": target_emotion,
                "detected_emotions": {"joy": 0.4, "sadness": 0.1, "anger": 0.05},
                "intensity": 0.7
            },
            "feedback": "The emotional content is generally effective but could be intensified."
        }
    
    def analyze_voice_consistency(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze voice consistency of text.
        
        Args:
            text: The text to analyze
            parameters: Parameters for voice consistency analysis
            
        Returns:
            Analysis results
        """
        # Placeholder implementation
        reference_text = parameters.get("reference_text", "")
        
        score = 0.8
        if reference_text:
            # If we have reference text, compare the embeddings
            text_embedding = self.model.encode(text)
            reference_embedding = self.model.encode(reference_text)
            similarity = np.dot(text_embedding, reference_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(reference_embedding)
            )
            score = float(similarity)
        
        return {
            "score": score,
            "details": {
                "style_markers": {
                    "sentence_length_variation": 0.65,
                    "vocabulary_richness": 0.7,
                    "distinctive_patterns": 0.75
                }
            },
            "feedback": "The voice is consistent but could use more distinctive stylistic markers."
        }
    
    def analyze_thematic_coherence(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze thematic coherence of text.
        
        Args:
            text: The text to analyze
            parameters: Parameters for thematic coherence analysis
            
        Returns:
            Analysis results
        """
        # Placeholder implementation
        themes = parameters.get("themes", [])
        
        score = 0.7
        detected_themes = {"love": 0.8, "betrayal": 0.6, "redemption": 0.3}
        
        return {
            "score": score,
            "details": {
                "target_themes": themes,
                "detected_themes": detected_themes,
                "thematic_progression": 0.65
            },
            "feedback": "The primary themes are present but could be developed more consistently."
        }
    
    def analyze_relevance(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze relevance of text to a reference or query.
        
        Args:
            text: The text to analyze
            parameters: Parameters for relevance analysis
            
        Returns:
            Analysis results
        """
        reference = parameters.get("reference", "")
        query = parameters.get("query", "")
        
        score = 0.8
        if reference or query:
            # If we have reference text or query, compare the embeddings
            text_embedding = self.model.encode(text)
            compare_text = reference or query
            compare_embedding = self.model.encode(compare_text)
            similarity = np.dot(text_embedding, compare_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(compare_embedding)
            )
            score = float(similarity)
        
        return {
            "score": score,
            "details": {
                "key_points_covered": 0.75,
                "off_topic_content": 0.1
            },
            "feedback": "The response is relevant but could address some key points more directly."
        }
    
    def analyze_completeness(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze completeness of text.
        
        Args:
            text: The text to analyze
            parameters: Parameters for completeness analysis
            
        Returns:
            Analysis results
        """
        # Placeholder implementation
        required_elements = parameters.get("required_elements", [])
        
        score = 0.85
        missing_elements = []
        
        return {
            "score": score,
            "details": {
                "required_elements": required_elements,
                "missing_elements": missing_elements,
                "depth_of_coverage": 0.8
            },
            "feedback": "The response is comprehensive but could explore some points in more depth."
        }
    
    def analyze_coherence(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze coherence of text.
        
        Args:
            text: The text to analyze
            parameters: Parameters for coherence analysis
            
        Returns:
            Analysis results
        """
        # Placeholder implementation
        score = 0.8
        
        return {
            "score": score,
            "details": {
                "logical_flow": 0.85,
                "transitions": 0.75,
                "internal_consistency": 0.8
            },
            "feedback": "The text has good logical flow but some transitions could be improved."
        }
    
    def analyze_generic_criterion(self, text: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze text for a generic criterion.
        
        Args:
            text: The text to analyze
            parameters: Parameters for the analysis
            
        Returns:
            Analysis results
        """
        # Placeholder implementation
        description = parameters.get("description", "")
        
        score = 0.75
        
        return {
            "score": score,
            "details": {},
            "feedback": f"The text scores 0.75 on: {description}"
        }
    
    def _generate_analysis_message(self, results: Dict[str, Any], overall_score: float) -> str:
        """
        Generate a human-readable summary of the analysis.
        
        Args:
            results: The analysis results for each criterion
            overall_score: The overall score
            
        Returns:
            Human-readable summary
        """
        message = f"Overall quality score: {overall_score:.2f}\n\n"
        
        for criterion, result in results.items():
            message += f"{criterion}: {result['score']:.2f} - {result['feedback']}\n"
        
        return message
