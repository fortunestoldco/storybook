# storybook/tools/nlp/voice_analyzer.py

"""
Voice analysis tool for evaluating narrative voice consistency and quality.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import re

class VoiceAnalyzer:
    """Tool for analyzing narrative voice and stylistic elements."""
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize the voice analyzer.
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
        """
        self.embedding_model = embedding_model
        
    def analyze_voice(self, text: str) -> Dict[str, Any]:
        """
        Analyze the voice characteristics of a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Voice analysis results
        """
        stylistic_features = self._extract_stylistic_features(text)
        sentence_patterns = self._analyze_sentence_patterns(text)
        word_choice = self._analyze_word_choice(text)
        pov_analysis = self._analyze_point_of_view(text)
        tone_analysis = self._analyze_tone(text)
        
        return {
            "stylistic_features": stylistic_features,
            "sentence_patterns": sentence_patterns,
            "word_choice": word_choice,
            "point_of_view": pov_analysis,
            "tone": tone_analysis,
            "voice_signature": self._generate_voice_signature(
                stylistic_features, sentence_patterns, word_choice, tone_analysis
            )
        }
    
    def compare_voice(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare the voice characteristics of two texts.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Voice comparison results
        """
        voice1 = self.analyze_voice(text1)
        voice2 = self.analyze_voice(text2)
        
        # Calculate embedding similarity if embedding model is available
        embedding_similarity = 0.0
        if self.embedding_model:
            embedding1 = self.embedding_model.encode(text1)
            embedding2 = self.embedding_model.encode(text2)
            embedding_similarity = float(np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            ))
        
        # Compare stylistic features
        feature_similarities = {}
        for feature in voice1["stylistic_features"]:
            if feature in voice2["stylistic_features"]:
                feature_similarities[feature] = 1.0 - abs(
                    voice1["stylistic_features"][feature] - 
                    voice2["stylistic_features"][feature]
                )
        
        # Calculate overall voice similarity
        if feature_similarities:
            feature_similarity = sum(feature_similarities.values()) / len(feature_similarities)
        else:
            feature_similarity = 0.0
            
        # Combine similarities
        if self.embedding_model:
            overall_similarity = (embedding_similarity + feature_similarity) / 2
        else:
            overall_similarity = feature_similarity
        
        return {
            "embedding_similarity": embedding_similarity,
            "feature_similarities": feature_similarities,
            "overall_similarity": overall_similarity,
            "voice1": voice1,
            "voice2": voice2,
            "assessment": self._generate_comparison_assessment(
                overall_similarity, feature_similarities, voice1, voice2
            )
        }
    
    def _extract_stylistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract stylistic features from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of stylistic features
        """
        # Split text into sentences and words
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate average sentence length
        avg_sentence_length = sum(len(re.findall(r'\b\w+\b', s)) for s in sentences) / max(1, len(sentences))
        
        # Calculate sentence length variation
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        sentence_length_variation = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Calculate lexical diversity (type-token ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(1, len(words))
        
        # Calculate average word length
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))
        
        # Calculate frequency of complex words (words with 3+ syllables)
        complex_words = [w for w in words if self._count_syllables(w) >= 3]
        complex_word_ratio = len(complex_words) / max(1, len(words))
        
        return {
            "avg_sentence_length": float(avg_sentence_length),
            "sentence_length_variation": float(sentence_length_variation),
            "lexical_diversity": float(lexical_diversity),
            "avg_word_length": float(avg_word_length),
            "complex_word_ratio": float(complex_word_ratio)
        }
    
    def _analyze_sentence_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentence patterns in text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Sentence pattern analysis
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Count question and exclamation sentences
        question_count = sum(1 for s in sentences if s.endswith('?'))
        exclamation_count = sum(1 for s in sentences if s.endswith('!'))
        
        # Analyze sentence beginnings
        beginnings = []
        for s in sentences:
            words = re.findall(r'\b\w+\b', s)
            if words:
                beginnings.append(words[0].lower())
        
        # Count unique beginnings
        unique_beginnings = set(beginnings)
        beginning_diversity = len(unique_beginnings) / max(1, len(beginnings))
        
        return {
            "question_ratio": question_count / max(1, len(sentences)),
            "exclamation_ratio": exclamation_count / max(1, len(sentences)),
            "beginning_diversity": float(beginning_diversity),
            "common_beginnings": self._get_most_common(beginnings, 3)
        }
    
    def _analyze_word_choice(self, text: str) -> Dict[str, Any]:
        """
        Analyze word choice patterns in text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Word choice analysis
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Analyze parts of speech (simplified)
        adjectives = self._count_word_types(words, ['beautiful', 'red', 'happy', 'large', 'small'])
        adverbs = self._count_word_types(words, ['quickly', 'very', 'suddenly', 'really', 'quite'])
        
        # Analyze common words
        common_words = self._get_most_common(words, 5)
        
        return {
            "adjective_ratio": adjectives / max(1, len(words)),
            "adverb_ratio": adverbs / max(1, len(words)),
            "common_words": common_words
        }
    
    def _analyze_point_of_view(self, text: str) -> Dict[str, Any]:
        """
        Analyze the point of view in text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Point of view analysis
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count personal pronouns
        first_person_singular = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
        first_person_plural = sum(1 for w in words if w in ['we', 'us', 'our', 'ours', 'ourselves'])
        second_person = sum(1 for w in words if w in ['you', 'your', 'yours', 'yourself', 'yourselves'])
        third_person_singular = sum(1 for w in words if w in ['he', 'she', 'it', 'him', 'her', 'his', 'hers', 'its', 'himself', 'herself', 'itself'])
        third_person_plural = sum(1 for w in words if w in ['they', 'them', 'their', 'theirs', 'themselves'])
        
        # Determine primary POV
        pov_counts = {
            "first_person_singular": first_person_singular,
            "first_person_plural": first_person_plural,
            "second_person": second_person,
            "third_person_singular": third_person_singular,
            "third_person_plural": third_person_plural
        }
        
        primary_pov = max(pov_counts, key=pov_counts.get)
        
        return {
            "pov_counts": pov_counts,
            "primary_pov": primary_pov,
            "pov_ratio": pov_counts[primary_pov] / max(1, sum(pov_counts.values()))
        }
    
    def _analyze_tone(self, text: str) -> Dict[str, Any]:
        """
        Analyze the tone of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Tone analysis
        """
        # Simplified tone analysis based on keyword presence
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Count tone indicators
        tone_indicators = {
            "formal": sum(1 for w in words if w in ['furthermore', 'subsequently', 'therefore', 'however', 'nevertheless']),
            "casual": sum(1 for w in words if w in ['yeah', 'cool', 'okay', 'stuff', 'guy', 'kind of']),
            "humorous": sum(1 for w in words if w in ['funny', 'laugh', 'joke', 'hilarious', 'amused']),
            "serious": sum(1 for w in words if w in ['grave', 'important', 'significant', 'critical', 'essential']),
            "optimistic": sum(1 for w in words if w in ['hope', 'happy', 'joy', 'positive', 'wonderful']),
            "pessimistic": sum(1 for w in words if w in ['sad', 'unfortunate', 'terrible', 'awful', 'negative'])
        }
        
        # Determine primary tone
        primary_tone = max(tone_indicators, key=tone_indicators.get) if any(tone_indicators.values()) else "neutral"
        
        return {
            "tone_indicators": tone_indicators,
            "primary_tone": primary_tone
        }
    
    def _generate_voice_signature(
        self, 
        stylistic_features: Dict[str, float],
        sentence_patterns: Dict[str, Any],
        word_choice: Dict[str, Any],
        tone_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a signature representing the voice.
        
        Args:
            stylistic_features: Stylistic features
            sentence_patterns: Sentence pattern analysis
            word_choice: Word choice analysis
            tone_analysis: Tone analysis
            
        Returns:
            Voice signature
        """
        # Combine the most distinctive elements into a signature
        signature = {
            "sentence_complexity": stylistic_features["avg_sentence_length"] * (1 + stylistic_features["sentence_length_variation"]),
            "vocabulary_richness": stylistic_features["lexical_diversity"] * (1 + stylistic_features["complex_word_ratio"]),
            "question_tendency": sentence_patterns["question_ratio"],
            "exclamation_tendency": sentence_patterns["exclamation_ratio"],
            "adjective_usage": word_choice["adjective_ratio"],
            "primary_pov": None,  # Will be filled if available
            "primary_tone": tone_analysis["primary_tone"]
        }
        
        return signature
    
    def _generate_comparison_assessment(
        self,
        overall_similarity: float,
        feature_similarities: Dict[str, float],
        voice1: Dict[str, Any],
        voice2: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable assessment of voice comparison.
        
        Args:
            overall_similarity: Overall similarity score
            feature_similarities: Similarities for individual features
            voice1: First voice analysis
            voice2: Second voice analysis
            
        Returns:
            Human-readable assessment
        """
        if overall_similarity > 0.9:
            assessment = "The voices are nearly identical, showing remarkable consistency."
        elif overall_similarity > 0.8:
            assessment = "The voices are very similar, with only minor variations."
        elif overall_similarity > 0.7:
            assessment = "The voices are similar but show noticeable differences."
        elif overall_similarity > 0.5:
            assessment = "The voices show substantial differences while maintaining some similarities."
        else:
            assessment = "The voices are significantly different and may not appear to be from the same source."
            
        # Add specific observations
        differences = []
        for feature, similarity in feature_similarities.items():
            if similarity < 0.7:
                differences.append(f"Different {feature} patterns")
                
        if voice1["point_of_view"]["primary_pov"] != voice2["point_of_view"]["primary_pov"]:
            differences.append(f"Different points of view: {voice1['point_of_view']['primary_pov']} vs {voice2['point_of_view']['primary_pov']}")
            
        if voice1["tone"]["primary_tone"] != voice2["tone"]["primary_tone"]:
            differences.append(f"Different tones: {voice1['tone']['primary_tone']} vs {voice2['tone']['primary_tone']}")
            
        if differences:
            assessment += " Key differences include: " + ", ".join(differences)
            
        return assessment
    
    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word (simplified).
        
        Args:
            word: The word to count syllables for
            
        Returns:
            Estimated number of syllables
        """
        # This is a simplified syllable counter
        word = word.lower()
        if len(word) <= 3:
            return 1
            
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Adjust for silent e
        if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
            count = max(1, count - 1)
            
        return max(1, count)
    
    def _count_word_types(self, words: List[str], examples: List[str]) -> int:
        """
        Count occurrences of words similar to example types.
        
        Args:
            words: List of words to check
            examples: List of example words of the desired type
            
        Returns:
            Count of matching words
        """
        # This is a simple approximation - in a real implementation, 
        # we would use a part-of-speech tagger
        count = 0
        for word in words:
            for example in examples:
                if word == example or word.endswith(example[-2:]):
                    count += 1
                    break
        return count
    
    def _get_most_common(self, items: List[str], n: int) -> List[Tuple[str, int]]:
        """
        Get the most common items in a list.
        
        Args:
            items: List of items
            n: Number of common items to return
            
        Returns:
            List of (item, count) tuples
        """
        from collections import Counter
        return Counter(items).most_common(n)
