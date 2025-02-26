import re
from collections import Counter
from typing import Any, Dict, List, Optional


class QualityMetrics:
    """Quality metrics for novel evaluation."""

    @staticmethod
    def calculate_thematic_coherence(themes: List[str], text: str) -> float:
        """
        Calculate thematic coherence score.

        This is a simplified implementation for demonstration purposes.
        A real implementation would use more sophisticated NLP techniques.
        """
        score = 0.0

        if not themes or not text:
            return score

        # Simple word frequency approach
        theme_related_terms = {}
        for theme in themes:
            # In a real implementation, we would use a thematic lexicon or embedding similarity
            theme_related_terms[theme] = [theme.lower()] + [
                word for word in theme.lower().split()
            ]

        text_lower = text.lower()
        total_terms = sum(len(terms) for terms in theme_related_terms.values())

        if total_terms == 0:
            return 0.0

        term_occurrences = 0
        for theme_terms in theme_related_terms.values():
            for term in theme_terms:
                term_occurrences += text_lower.count(term)

        # Normalize by text length to avoid favoring longer texts
        word_count = len(text.split())
        if word_count > 0:
            normalized_score = term_occurrences / (
                word_count * 0.01
            )  # 1 occurrence per 100 words = 1.0
            return min(normalized_score, 1.0)  # Cap at 1.0

        return 0.0

    @staticmethod
    def calculate_character_consistency(
        character: Dict[str, Any], dialogue: str
    ) -> float:
        """
        Calculate character consistency in dialogue.

        This is a simplified implementation for demonstration purposes.
        """
        if not character or not dialogue:
            return 0.0

        consistency_score = 0.7  # Default baseline

        # In a real implementation, we would:
        # 1. Extract character-specific dialogue patterns
        # 2. Check for consistency with established patterns
        # 3. Verify personality traits are reflected in dialogue
        # 4. Check for appropriate reactions based on character background

        # Simplified example: check if dialogue patterns are present
        if "dialogue_patterns" in character:
            pattern_matches = 0
            total_patterns = len(character["dialogue_patterns"])

            if total_patterns > 0:
                for pattern_type, pattern in character["dialogue_patterns"].items():
                    if isinstance(pattern, str) and pattern.lower() in dialogue.lower():
                        pattern_matches += 1

                pattern_score = pattern_matches / total_patterns
                consistency_score = 0.3 + (
                    0.7 * pattern_score
                )  # 30% baseline + 70% pattern matching

        return consistency_score

    @staticmethod
    def calculate_narrative_engagement(text: str) -> Dict[str, float]:
        """
        Calculate narrative engagement metrics.

        Returns dict with pacing_variance, emotional_arc_completeness, and plot_resolution_satisfaction.
        """
        result = {
            "pacing_variance": 0.0,
            "emotional_arc_completeness": 0.0,
            "plot_resolution_satisfaction": 0.0,
        }

        if not text:
            return result

        # Pacing variance - simplified implementation
        # In a real system, we'd analyze sentence length, paragraph structure, action vs. dialog, etc.
        sentences = re.split(r"[.!?]+", text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if sentence_lengths:
            mean_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum(
                (length - mean_length) ** 2 for length in sentence_lengths
            ) / len(sentence_lengths)
            normalized_variance = min(variance / 100, 1.0)  # Normalize, cap at 1.0
            result["pacing_variance"] = normalized_variance

        # Emotional arc - simplified implementation
        # In a real system, we'd use sentiment analysis across narrative segments
        emotion_words = {
            "positive": ["happy", "joy", "love", "excited", "hope", "triumph"],
            "negative": ["sad", "fear", "angry", "despair", "lost", "defeated"],
            "neutral": ["thought", "considered", "wondered", "observed"],
        }

        emotion_counts = {category: 0 for category in emotion_words}
        text_lower = text.lower()

        for category, words in emotion_words.items():
            for word in words:
                emotion_counts[category] += text_lower.count(word)

        total_emotion_words = sum(emotion_counts.values())

        if total_emotion_words > 0:
            # Measure emotional variety - higher is better
            emotion_distribution = [
                count / total_emotion_words for count in emotion_counts.values()
            ]
            emotion_variety = min(
                1.0,
                1
                - abs(emotion_distribution[0] - 0.33)
                - abs(emotion_distribution[1] - 0.33)
                - abs(emotion_distribution[2] - 0.33),
            )
            result["emotional_arc_completeness"] = emotion_variety

        # Plot resolution - simplified placeholder
        # In a real system, we'd track open plot threads and their resolution
        result["plot_resolution_satisfaction"] = 0.7  # Placeholder value

        return result

    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        if not text:
            return {
                "flesch_kincaid": 0.0,
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
            }

        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]

        total_sentences = len(sentences)
        words = text.split()
        total_words = len(words)
        total_syllables = 0

        # Very simplified syllable counter
        for word in words:
            word = word.lower().strip(".,;:!?-\"'()[]{}").replace("'s", "")
            if not word:
                continue

            syllable_count = 0
            vowels = "aeiouy"
            is_prev_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not is_prev_vowel:
                    syllable_count += 1
                is_prev_vowel = is_vowel

            if syllable_count == 0:
                syllable_count = 1
            total_syllables += syllable_count

        if total_sentences == 0 or total_words == 0:
            return {
                "flesch_kincaid": 0.0,
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
            }

        avg_sentence_length = total_words / total_sentences
        avg_word_length = sum(len(word) for word in words) / total_words

        # Flesch-Kincaid Grade Level
        flesch_kincaid = (
            0.39 * (total_words / total_sentences)
            + 11.8 * (total_syllables / total_words)
            - 15.59
        )

        return {
            "flesch_kincaid": flesch_kincaid,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
        }
