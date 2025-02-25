"""Analysis utilities for story quality evaluation in the Storybook application."""

import re
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import Counter

from storybook.text_utils import (
    preprocess_text, extract_keywords, extract_topics, analyze_text_sentiment,
    extract_entities, detect_dialogue, detect_narrative_perspective, detect_reading_level,
    detect_style_elements, calculate_text_stats
)

def analyze_thematic_coherence(text: str, themes: List[str]) -> Dict[str, Any]:
    """
    Analyze how well a text maintains thematic coherence with specified themes.
    
    Args:
        text: The text to analyze
        themes: List of themes to check for
        
    Returns:
        Dictionary with thematic coherence metrics
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Extract keywords and topics
    keywords = extract_keywords(text, max_keywords=20)
    topics = extract_topics(text, max_topics=10)
    
    # Generate related terms for each theme
    theme_related_terms = {}
    theme_presence = {}
    
    for theme in themes:
        # Basic implementation - in a real system, this would use 
        # more sophisticated semantic similarity or a thematic lexicon
        theme_terms = set([theme.lower()])
        
        # Add simple variations and related words
        # This is a simplified version - a real implementation would use
        # word embeddings or a thesaurus API
        if theme.lower() == "love":
            theme_terms.update(["romance", "affection", "passion", "heart", "emotional", "relationship"])
        elif theme.lower() == "betrayal":
            theme_terms.update(["treachery", "deception", "dishonesty", "treason", "disloyal", "backstab"])
        elif theme.lower() == "redemption":
            theme_terms.update(["forgiveness", "atonement", "salvation", "reconciliation", "amends"])
        elif theme.lower() == "justice":
            theme_terms.update(["fairness", "equality", "rights", "law", "moral", "ethics", "punishment"])
        elif theme.lower() == "power":
            theme_terms.update(["control", "authority", "dominance", "strength", "influence", "force"])
        # Add more themes as needed
        
        # Store related terms for this theme
        theme_related_terms[theme] = theme_terms
        
        # Count occurrences
        term_count = sum(1 for term in processed_text if term in theme_terms)
        
        # Calculate normalized presence (adjusted for text length)
        normalized_presence = term_count / max(1, len(processed_text) / 100)
        theme_presence[theme] = normalized_presence
    
    # Calculate overall thematic coherence score
    if themes:
        # Average theme presence
        avg_presence = sum(theme_presence.values()) / len(themes)
        
        # Variance in theme presence (lower is better for balance)
        variance = sum((presence - avg_presence) ** 2 for presence in theme_presence.values()) / len(themes)
        
        # Keywords and topics alignment with themes
        theme_terms_all = set()
        for terms in theme_related_terms.values():
            theme_terms_all.update(terms)
        
        keyword_alignment = sum(1 for keyword in keywords if keyword in theme_terms_all) / max(1, len(keywords))
        topic_alignment = sum(1 for topic in topics if topic in theme_terms_all) / max(1, len(topics))
        
        # Combined score (0-1 scale)
        coherence_score = (
            0.4 * avg_presence +
            0.3 * (1 - min(1.0, variance * 5)) +  # Convert variance to 0-1 scale and invert
            0.2 * keyword_alignment +
            0.1 * topic_alignment
        )
    else:
        coherence_score = 0.0
    
    return {
        "thematic_coherence_score": round(coherence_score, 2),
        "theme_presence": theme_presence,
        "keyword_alignment": keywords,
        "topic_alignment": topics,
        "themes_analyzed": themes
    }

def analyze_character_consistency(text: str, characters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze character consistency throughout a text.
    
    Args:
        text: The text to analyze
        characters: List of character dictionaries with name and traits
        
    Returns:
        Dictionary with character consistency metrics
    """
    character_mentions = {}
    character_dialogue = {}
    character_actions = {}
    
    # Split text into sentences for better context
    import nltk
    sentences = nltk.sent_tokenize(text)
    
    # Process each character
    for character in characters:
        name = character["name"]
        traits = character.get("traits", [])
        
        # Count mentions
        name_pattern = r'\b' + re.escape(name) + r'\b'
        mentions = sum(1 for sentence in sentences if re.search(name_pattern, sentence, re.IGNORECASE))
        character_mentions[name] = mentions
        
        # Extract dialogue associated with character
        dialogue_pattern = r'(?:' + re.escape(name) + r'\s*(?:said|replied|asked|exclaimed|whispered|shouted|answered),?\s*"|"\s*(?:said|replied|asked|exclaimed|whispered|shouted|answered)\s*' + re.escape(name) + r')'
        character_dialogues = []
        
        for i, sentence in enumerate(sentences):
            if re.search(dialogue_pattern, sentence, re.IGNORECASE):
                # Look for the actual dialogue
                dialogue_match = re.search(r'"([^"]*)"', sentence)
                if dialogue_match:
                    character_dialogues.append(dialogue_match.group(1))
                elif i > 0:  # Check previous sentence for dialogue
                    prev_dialogue = re.search(r'"([^"]*)"', sentences[i-1])
                    if prev_dialogue:
                        character_dialogues.append(prev_dialogue.group(1))
        
        character_dialogue[name] = character_dialogues
        
        # Find character actions
        action_pattern = r'' + re.escape(name) + r'\s+(?:\w+ed|\w+s)\b'
        character_action_list = []
        
        for sentence in sentences:
            action_matches = re.finditer(action_pattern, sentence, re.IGNORECASE)
            for match in action_matches:
                start = match.start()
                # Get some context around the action
                context_start = max(0, start - 50)
                context_end = min(len(sentence), start + 100)
                action_context = sentence[context_start:context_end]
                character_action_list.append(action_context.strip())
        
        character_actions[name] = character_action_list
    
    # Calculate consistency metrics
    consistency_scores = {}
    
    for character in characters:
        name = character["name"]
        traits = character.get("traits", [])
        
        if not traits or name not in character_dialogue or not character_dialogue[name]:
            consistency_scores[name] = 0.5  # Default middle score if no data
            continue
        
        # Check if dialogue reflects character traits
        trait_presence = []
        
        # This is a simplified implementation - in a production system,
        # this would use more sophisticated NLP techniques
        for trait in traits:
            trait_words = set([trait.lower()])
            
            # Add simple trait-related words
            if trait.lower() == "intelligent":
                trait_words.update(["smart", "clever", "logical", "rational", "thoughtful"])
            elif trait.lower() == "brave":
                trait_words.update(["courage", "fearless", "bold", "daring", "heroic"])
            elif trait.lower() == "kind":
                trait_words.update(["gentle", "compassionate", "caring", "helpful", "nice"])
            # Add more traits as needed
            
            # Check dialogues for trait indicators
            dialogues_text = " ".join(character_dialogue[name]).lower()
            trait_count = sum(1 for word in trait_words if word in dialogues_text)
            
            # Check actions for trait indicators
            actions_text = " ".join(character_actions[name]).lower()
            trait_action_count = sum(1 for word in trait_words if word in actions_text)
            
            # Calculate normalized presence
            dialogue_length = len(dialogues_text.split())
            action_length = len(actions_text.split())
            
            if dialogue_length + action_length > 0:
                normalized_presence = (trait_count + trait_action_count) / (dialogue_length + action_length) * 100
                trait_presence.append(normalized_presence)
        
        # Calculate consistency score
        if trait_presence:
            # Average trait presence
            avg_presence = sum(trait_presence) / len(trait_presence)
            
            # Variance in trait presence (lower is better for consistent portrayal)
            if len(trait_presence) > 1:
                variance = sum((presence - avg_presence) ** 2 for presence in trait_presence) / len(trait_presence)
            else:
                variance = 0
            
            # Combined score (0-1 scale)
            consistency_score = min(1.0, (
                0.6 * min(1.0, avg_presence * 0.5) +  # Scale avg_presence
                0.4 * (1 - min(1.0, variance * 2))    # Convert variance to 0-1 scale and invert
            ))
        else:
            consistency_score = 0.5  # Default middle score
        
        consistency_scores[name] = round(consistency_score, 2)
    
    # Overall character consistency
    overall_consistency = (
        sum(consistency_scores.values()) / max(1, len(consistency_scores))
        if consistency_scores else 0.5
    )
    
    return {
        "character_consistency_score": round(overall_consistency, 2),
        "individual_scores": consistency_scores,
        "character_mentions": character_mentions,
        "dialogue_samples": {name: dialogues[:3] for name, dialogues in character_dialogue.items() if dialogues},
        "action_samples": {name: actions[:3] for name, actions in character_actions.items() if actions}
    }

def analyze_narrative_engagement(text: str) -> Dict[str, Any]:
    """
    Analyze narrative engagement factors like pacing, emotional arc, and plot resolution.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with narrative engagement metrics
    """
    # Divide text into sections for pacing analysis
    import nltk
    import math
    
    sentences = nltk.sent_tokenize(text)
    
    # If the text is too short, return limited analysis
    if len(sentences) < 10:
        return {
            "pacing_variance": 0.0,
            "emotional_intensity": analyze_text_sentiment(text),
            "narrative_arc_completeness": 0.3,  # Assume incomplete for very short texts
            "engagement_score": 0.4
        }
    
    # Divide into sections (minimum 3, maximum 10)
    num_sections = min(10, max(3, math.ceil(len(sentences) / 15)))
    section_size = len(sentences) // num_sections
    sections = []
    
    for i in range(num_sections):
        start = i * section_size
        end = start + section_size if i < num_sections - 1 else len(sentences)
        section_text = ' '.join(sentences[start:end])
        sections.append(section_text)
    
    # Analyze pacing through sentence length variance in each section
    pacing_metrics = []
    
    for section in sections:
        section_sentences = nltk.sent_tokenize(section)
        sentence_lengths = [len(nltk.word_tokenize(s)) for s in section_sentences]
        
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
            pacing_metrics.append({
                "avg_sentence_length": avg_length,
                "variance": variance
            })
    
    # Calculate pacing variance across sections (higher variance indicates more dynamic pacing)
    section_avgs = [m["avg_sentence_length"] for m in pacing_metrics]
    if section_avgs:
        overall_avg = sum(section_avgs) / len(section_avgs)
        pacing_variance = sum((avg - overall_avg) ** 2 for avg in section_avgs) / len(section_avgs)
        normalized_pacing_variance = min(1.0, pacing_variance / 25)  # Normalize to 0-1 scale
    else:
        normalized_pacing_variance = 0.0
    
    # Analyze emotional arc by sentiment across sections
    emotional_arc = []
    
    for section in sections:
        sentiment = analyze_text_sentiment(section)
        emotional_arc.append({
            "sentiment": sentiment["sentiment"],
            "score": sentiment["score"]
        })
    
    # Calculate emotional range and shifts
    sentiment_scores = [e["score"] for e in emotional_arc]
    if sentiment_scores:
        emotional_range = max(sentiment_scores) - min(sentiment_scores)
        
        # Count sentiment shifts (changes from positive to negative or vice versa)
        sentiment_shifts = 0
        for i in range(1, len(sentiment_scores)):
            if (sentiment_scores[i] > 0 and sentiment_scores[i-1] < 0) or \
               (sentiment_scores[i] < 0 and sentiment_scores[i-1] > 0):
                sentiment_shifts += 1
                
        # Normalized emotional arc completeness
        emotional_arc_completeness = min(1.0, (
            0.5 * emotional_range +
            0.5 * (sentiment_shifts / max(1, len(sentiment_scores) - 1))
        ))
    else:
        emotional_range = 0.0
        sentiment_shifts = 0
        emotional_arc_completeness = 0.0
    
    # Estimate narrative arc completeness
    # This is a heuristic based on section sentiment patterns
    # In a real implementation, this would use more sophisticated analysis
    
    # Check if we have a complete arc (setup, confrontation, resolution pattern)
    if len(sentiment_scores) >= 3:
        # Simplified arc check:
        # - Beginning should be neutral or slightly positive
        # - Middle should have conflict (negative sentiment or shifts)
        # - End should show resolution (return to positive or conclusion of shifts)
        
        beginning_sentiment = sum(sentiment_scores[:len(sentiment_scores)//3]) / (len(sentiment_scores)//3)
        middle_sentiment = sum(sentiment_scores[len(sentiment_scores)//3:2*len(sentiment_scores)//3]) / (len(sentiment_scores)//3)
        end_sentiment = sum(sentiment_scores[2*len(sentiment_scores)//3:]) / (len(sentiment_scores) - 2*len(sentiment_scores)//3)
        
        has_beginning = abs(beginning_sentiment) < 0.3  # Close to neutral
        has_conflict = middle_sentiment < beginning_sentiment or sentiment_shifts > 0
        has_resolution = end_sentiment > middle_sentiment or abs(end_sentiment - beginning_sentiment) < 0.2
        
        narrative_arc_completeness = (
            0.3 * (1.0 if has_beginning else 0.5) +
            0.4 * (1.0 if has_conflict else 0.3) +
            0.3 * (1.0 if has_resolution else 0.2)
        )
    else:
        narrative_arc_completeness = 0.5  # Default for short texts
    
    # Calculate overall engagement score
    engagement_score = (
        0.3 * normalized_pacing_variance +
        0.4 * emotional_arc_completeness +
        0.3 * narrative_arc_completeness
    )
    
    return {
        "pacing_variance": round(normalized_pacing_variance, 2),
        "emotional_arc": emotional_arc,
        "emotional_range": round(emotional_range, 2),
        "sentiment_shifts": sentiment_shifts,
        "emotional_arc_completeness": round(emotional_arc_completeness, 2),
        "narrative_arc_completeness": round(narrative_arc_completeness, 2),
        "engagement_score": round(engagement_score, 2)
    }

def calculate_quality_improvement(original_text: str, revised_text: str) -> Dict[str, Any]:
    """
    Calculate quality improvement between original and revised text.
    
    Args:
        original_text: Original text before revision
        revised_text: Revised version of the text
        
    Returns:
        Dictionary with improvement metrics
    """
    # Get text statistics for both versions
    original_stats = calculate_text_stats(original_text)
    revised_stats = calculate_text_stats(revised_text)
    
    # Get reading level metrics
    original_reading = detect_reading_level(original_text)
    revised_reading = detect_reading_level(revised_text)
    
    # Get style elements
    original_style = detect_style_elements(original_text)
    revised_style = detect_style_elements(revised_text)
    
    # Calculate technical improvements
    word_count_change = revised_stats["word_count"] - original_stats["word_count"]
    word_count_change_percent = (word_count_change / original_stats["word_count"]) * 100 if original_stats["word_count"] > 0 else 0
    
    lexical_diversity_change = revised_stats["lexical_diversity"] - original_stats["lexical_diversity"]
    lexical_diversity_change_percent = (lexical_diversity_change / original_stats["lexical_diversity"]) * 100 if original_stats["lexical_diversity"] > 0 else 0
    
    # Calculate readability changes
    reading_ease_change = revised_reading["reading_ease"] - original_reading["reading_ease"]
    
    # Calculate style improvements
    style_complexity_change = revised_style["style_complexity"] - original_style["style_complexity"]
    
    # Dialogue improvement
    original_dialogue = detect_dialogue(original_text)
    revised_dialogue = detect_dialogue(revised_text)
    dialogue_percentage_change = revised_dialogue["dialogue_percentage"] - original_dialogue["dialogue_percentage"]
    
    # Calculate overall improvement score
    # This is a weighted combination of various metrics
    technical_improvement = min(1.0, max(0.0, (
        0.5 * (1 + (lexical_diversity_change_percent / 100)) +  # Normalize to 0-1
        0.5 * (abs(reading_ease_change) / 20)  # Normalize to 0-1, any change (up or down) could be improvement
    )))
    
    stylistic_improvement = min(1.0, max(0.0, (
        0.7 * (style_complexity_change + 0.5) +  # Normalize to 0-1, assuming -0.5 to +0.5 range
        0.3 * (abs(dialogue_percentage_change) / 10)  # Normalize to 0-1
    )))
    
    overall_improvement = (
        0.6 * technical_improvement +
        0.4 * stylistic_improvement
    )
    
    return {
        "word_count_change": word_count_change,
        "word_count_change_percent": round(word_count_change_percent, 1),
        "lexical_diversity_change": round(lexical_diversity_change, 3),
        "reading_ease_change": round(reading_ease_change, 1),
        "style_complexity_change": round(style_complexity_change, 2),
        "technical_improvement_score": round(technical_improvement, 2),
        "stylistic_improvement_score": round(stylistic_improvement, 2),
        "overall_improvement_score": round(overall_improvement, 2)
    }

def evaluate_story_quality(text: str, themes: List[str] = None, characters: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of story quality.
    
    Args:
        text: The story text to evaluate
        themes: Optional list of themes
        characters: Optional list of characters
        
    Returns:
        Dictionary with quality metrics
    """
    # If themes or characters not provided, try to extract them
    if not themes:
        themes = extract_topics(text, max_topics=5)
    
    if not characters:
        # Simple character extraction - in a real system this would be more sophisticated
        entities = extract_entities(text)
        characters = [{"name": name} for name in entities.get("people", [])]
    
    # Text statistics
    text_stats = calculate_text_stats(text)
    
    # Readability
    reading_level = detect_reading_level(text)
    
    # Style elements
    style_elements = detect_style_elements(text)
    
    # Perspective
    perspective = detect_narrative_perspective(text)
    
    # Thematic analysis
    thematic_analysis = analyze_thematic_coherence(text, themes)
    
    # Character consistency
    character_analysis = analyze_character_consistency(text, characters)
    
    # Narrative engagement
    engagement_analysis = analyze_narrative_engagement(text)
    
    # Calculate overall quality score
    quality_score = (
        0.25 * thematic_analysis["thematic_coherence_score"] +
        0.25 * character_analysis["character_consistency_score"] +
        0.30 * engagement_analysis["engagement_score"] +
        0.20 * style_elements["style_complexity"]
    )
    
    return {
        "overall_quality_score": round(quality_score, 2),
        "word_count": text_stats["word_count"],
        "reading_level": reading_level["flesch_kincaid_grade"],
        "reading_ease": reading_level["reading_ease"],
        "narrative_perspective": perspective["dominant_perspective"],
        "thematic_coherence": thematic_analysis["thematic_coherence_score"],
        "character_consistency": character_analysis["character_consistency_score"],
        "narrative_engagement": engagement_analysis["engagement_score"],
        "style_complexity": style_elements["style_complexity"],
        "detailed_metrics": {
            "text_stats": text_stats,
            "reading_level": reading_level,
            "style_elements": style_elements,
            "thematic_analysis": thematic_analysis,
            "character_analysis": character_analysis,
            "engagement_analysis": engagement_analysis
        }
    }
