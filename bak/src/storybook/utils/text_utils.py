"""Text processing utilities for the Storybook application."""

import re
import string
import random
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


def preprocess_text(text: str) -> List[str]:
    """Preprocess text for analysis."""
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def extract_keywords(text: str, max_keywords: int = 10, min_word_length: int = 3) -> List[str]:
    """Extract keywords from text."""
    # Preprocess text
    tokens = preprocess_text(text)

    # Filter by length
    tokens = [token for token in tokens if len(token) >= min_word_length]

    # Count token frequencies
    token_counts = Counter(tokens)

    # Get most common tokens as keywords
    keywords = [token for token, count in token_counts.most_common(max_keywords)]

    return keywords


def extract_key_phrases(
    text: str, max_phrases: int = 5, ngram_range: Tuple[int, int] = (2, 3)
) -> List[str]:
    """Extract key phrases (n-grams) from text."""
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stop words and punctuation
    stop_words = set(stopwords.words("english"))
    tokens = [
        token for token in tokens if token not in stop_words and token not in string.punctuation
    ]

    # Generate n-grams within the specified range
    all_ngrams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        all_ngrams.extend(list(ngrams(tokens, n)))

    # Convert n-gram tuples to strings
    ngram_strings = [" ".join(gram) for gram in all_ngrams]

    # Count n-gram frequencies
    ngram_counts = Counter(ngram_strings)

    # Get most common n-grams as key phrases
    key_phrases = [phrase for phrase, count in ngram_counts.most_common(max_phrases)]

    return key_phrases


def summarize_text(text: str, max_length: int = 200) -> str:
    """Create a summary of text using sentence extraction."""
    # Split into sentences
    sentences = sent_tokenize(text)

    # If text is already short, return it
    if len(text) <= max_length:
        return text

    # If only one sentence, truncate it
    if len(sentences) == 1:
        return sentences[0][:max_length] + "..."

    # Score sentences by position and keyword frequency
    tokens = preprocess_text(text)
    keyword_counts = Counter(tokens)
    keywords = set([kw for kw, count in keyword_counts.most_common(10)])

    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0

        # Position score (first sentences get higher score)
        position_score = 1.0 - (i / len(sentences))
        score += position_score * 0.5

        # Keyword score
        sentence_tokens = preprocess_text(sentence)
        keyword_score = sum(1 for token in sentence_tokens if token in keywords) / max(
            1, len(sentence_tokens)
        )
        score += keyword_score * 0.5

        scored_sentences.append((sentence, score))

    # Sort sentences by score
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select top sentences until max length is reached
    summary_sentences = []
    current_length = 0

    for sentence, score in scored_sentences:
        if current_length + len(sentence) <= max_length:
            summary_sentences.append(sentence)
            current_length += len(sentence) + 1  # +1 for space
        else:
            break

    # Sort sentences back to original order
    summary_sentences_with_index = []
    for sentence in summary_sentences:
        index = sentences.index(sentence)
        summary_sentences_with_index.append((index, sentence))

    summary_sentences_with_index.sort(key=lambda x: x[0])

    # Join sentences
    summary = " ".join([sentence for _, sentence in summary_sentences_with_index])

    # If still too long, truncate
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return summary


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text (positive, negative, neutral)."""
    # Simple rule-based sentiment analysis as a fallback
    # In a real implementation, use a proper sentiment analysis library

    positive_words = {
        "good",
        "great",
        "excellent",
        "wonderful",
        "amazing",
        "happy",
        "joy",
        "love",
        "beautiful",
        "positive",
        "success",
        "successful",
        "best",
        "win",
        "winning",
        "lovely",
        "outstanding",
        "fantastic",
        "terrific",
        "delightful",
        "pleased",
    }

    negative_words = {
        "bad",
        "terrible",
        "awful",
        "horrible",
        "sad",
        "unhappy",
        "hate",
        "dislike",
        "ugly",
        "negative",
        "fail",
        "failure",
        "worst",
        "lose",
        "losing",
        "poor",
        "disappointing",
        "unfortunate",
        "miserable",
        "unpleasant",
        "troubled",
    }

    tokens = preprocess_text(text)

    # Count positive and negative words
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)

    # Determine sentiment
    sentiment = "neutral"
    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"

    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total > 0:
        sentiment_score = (positive_count - negative_count) / total
    else:
        sentiment_score = 0.0

    return {
        "sentiment": sentiment,
        "score": sentiment_score,
        "positive_words": positive_count,
        "negative_words": negative_count,
    }


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities like people, places, organizations, etc."""
    # In a real implementation, use a proper NER library

    # Simplified entity extraction using regex patterns
    entities = {"people": [], "places": [], "organizations": [], "dates": []}

    # Simple regex for capitalized names (people and places)
    name_pattern = r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"
    names = re.findall(name_pattern, text)

    # Simple date patterns
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # MM/DD/YYYY
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",  # MM-DD-YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",  # Month DD, YYYY
    ]

    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))

    entities["dates"] = dates

    # Very basic classification of names into people vs places
    # This is oversimplified and would be better with actual NER
    for name in names:
        if name.lower().startswith(("mr", "mrs", "ms", "dr", "miss", "sir", "lady")):
            entities["people"].append(name)
        elif name.lower().endswith(("inc", "corp", "llc", "company", "co", "ltd")):
            entities["organizations"].append(name)
        elif name.lower().endswith(("city", "town", "village", "county", "state", "country")):
            entities["places"].append(name)
        else:
            # Default to people for capitalized names
            entities["people"].append(name)

    return entities


def extract_topics(text: str, max_topics: int = 5) -> List[str]:
    """Extract main topics from text."""
    # Preprocess text
    tokens = preprocess_text(text)

    # Count token frequencies
    token_counts = Counter(tokens)

    # Get most common tokens as topics
    topics = [token for token, count in token_counts.most_common(max_topics)]

    return topics


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate various statistics about a text."""
    # Tokenize
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Count words, excluding punctuation
    word_count = len([word for word in words if word not in string.punctuation])

    # Average sentence length
    avg_sentence_length = word_count / max(1, len(sentences))

    # Average word length
    avg_word_length = sum(len(word) for word in words if word not in string.punctuation) / max(
        1, word_count
    )

    # Lexical diversity (unique words / total words)
    unique_words = set(word.lower() for word in words if word not in string.punctuation)
    lexical_diversity = len(unique_words) / max(1, word_count)

    return {
        "sentence_count": len(sentences),
        "word_count": word_count,
        "character_count": len(text),
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity,
    }


def detect_dialogue(text: str) -> Dict[str, Any]:
    """Detect and analyze dialogue in text."""
    # Simple dialogue detection using quotes
    dialogue_pattern = r'"([^"]*)"'
    dialogues = re.findall(dialogue_pattern, text)

    # Alternative dialogue pattern with single quotes
    if not dialogues:
        dialogue_pattern = r"'([^']*)'"
        dialogues = re.findall(dialogue_pattern, text)

    # Calculate statistics
    dialogue_count = len(dialogues)
    dialogue_words = sum(len(word_tokenize(d)) for d in dialogues)

    # Estimate total words
    total_words = len(word_tokenize(text))

    # Calculate dialogue percentage
    dialogue_percentage = dialogue_words / max(1, total_words) * 100

    return {
        "dialogue_count": dialogue_count,
        "dialogue_excerpts": dialogues[:5],  # Return first 5 dialogues as examples
        "dialogue_words": dialogue_words,
        "dialogue_percentage": dialogue_percentage,
    }


def generate_title_suggestions(text: str, count: int = 5) -> List[str]:
    """Generate title suggestions based on text content."""
    # Extract keywords and phrases
    keywords = extract_keywords(text, max_keywords=15)
    phrases = extract_key_phrases(text, max_phrases=10)

    # Generate title templates
    templates = [
        "The {noun} of {noun2}",
        "{adjective} {noun}",
        "{noun}: A Story of {noun2}",
        "The {adjective} {noun} of {noun2}",
        "When {noun} Meets {noun2}",
        "{verb}ing the {noun}",
        "A {noun} in {time}",
        "The Last {noun}",
        "{noun}'s {noun2}",
        "The {noun} {verb}s",
    ]

    # Simple part-of-speech tagging
    nouns = [
        word
        for word in keywords
        if word.lower() not in ["the", "a", "an", "and", "or", "but", "in", "on", "at"]
    ]
    verbs = []
    adjectives = []

    # In a real implementation, use proper POS tagging
    for word in keywords:
        if word.endswith(("ed", "ing")):
            verbs.append(word)
        elif word.endswith(("ful", "ous", "ive", "al", "ic", "able", "ible")):
            adjectives.append(word)

    # Ensure we have some words in each category
    if not verbs:
        verbs = ["discover", "journey", "find", "create", "build", "explore"]
    if not adjectives:
        adjectives = ["hidden", "secret", "lost", "forgotten", "amazing", "incredible"]

    # Time-related words
    times = ["time", "darkness", "light", "dawn", "dusk", "summer", "winter", "night", "day"]

    # Generate titles
    titles = []
    for _ in range(min(count, 20)):  # Generate up to 20 titles, then select the requested number
        template = random.choice(templates)

        # Fill in template
        title = template
        if "{noun}" in title:
            title = title.replace("{noun}", random.choice(nouns))
        if "{noun2}" in title:
            title = title.replace("{noun2}", random.choice(nouns))
        if "{verb}" in title:
            title = title.replace("{verb}", random.choice(verbs))
        if "{adjective}" in title:
            title = title.replace("{adjective}", random.choice(adjectives))
        if "{time}" in title:
            title = title.replace("{time}", random.choice(times))

        titles.append(title.title())  # Title case

    # Add some titles based on extracted phrases
    for phrase in phrases[:3]:
        titles.append(phrase.title())

    # Remove duplicates and return requested number
    unique_titles = list(set(titles))
    return unique_titles[:count]


def detect_narrative_perspective(text: str) -> Dict[str, Any]:
    """Detect narrative perspective (first person, third person, etc.)."""
    # Look for first person pronouns
    first_person_pattern = r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b"
    first_person_matches = re.findall(first_person_pattern, text, re.IGNORECASE)
    first_person_count = len(first_person_matches)

    # Look for second person pronouns
    second_person_pattern = r"\b(you|your|yours|yourself|yourselves)\b"
    second_person_matches = re.findall(second_person_pattern, text, re.IGNORECASE)
    second_person_count = len(second_person_matches)

    # Look for third person pronouns
    third_person_pattern = r"\b(he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves)\b"
    third_person_matches = re.findall(third_person_pattern, text, re.IGNORECASE)
    third_person_count = len(third_person_matches)

    # Determine dominant perspective
    total_pronouns = first_person_count + second_person_count + third_person_count
    if total_pronouns == 0:
        dominant = "unknown"
        confidence = 0.0
    else:
        if first_person_count > second_person_count and first_person_count > third_person_count:
            dominant = "first_person"
            confidence = first_person_count / total_pronouns
        elif second_person_count > first_person_count and second_person_count > third_person_count:
            dominant = "second_person"
            confidence = second_person_count / total_pronouns
        else:
            dominant = "third_person"
            confidence = third_person_count / total_pronouns

    return {
        "dominant_perspective": dominant,
        "confidence": confidence,
        "first_person_count": first_person_count,
        "second_person_count": second_person_count,
        "third_person_count": third_person_count,
    }


def detect_reading_level(text: str) -> Dict[str, Any]:
    """Estimate reading level using Flesch-Kincaid and other metrics."""
    # Tokenize
    sentences = sent_tokenize(text)
    words = [word for word in word_tokenize(text) if word not in string.punctuation]

    # Count syllables (simplified approximation)
    def count_syllables(word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    total_syllables = sum(count_syllables(word) for word in words)

    # Calculate metrics
    num_sentences = len(sentences)
    num_words = len(words)

    if num_sentences == 0 or num_words == 0:
        return {
            "flesch_kincaid_grade": 0,
            "reading_ease": 100,
            "avg_syllables_per_word": 0,
            "complex_words_percentage": 0,
        }

    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59

    # Flesch Reading Ease
    reading_ease = (
        206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
    )

    # Complex words (words with 3+ syllables)
    complex_words = sum(1 for word in words if count_syllables(word) >= 3)
    complex_percentage = complex_words / num_words * 100

    return {
        "flesch_kincaid_grade": round(fk_grade, 1),
        "reading_ease": min(100, max(0, round(reading_ease, 1))),
        "avg_syllables_per_word": round(total_syllables / num_words, 2),
        "complex_words_percentage": round(complex_percentage, 1),
    }


def detect_style_elements(text: str) -> Dict[str, Any]:
    """Detect and analyze stylistic elements in text."""
    # Sentence types
    sentences = sent_tokenize(text)

    # Simple detection of sentence types
    question_count = sum(1 for s in sentences if s.strip().endswith("?"))
    exclamation_count = sum(1 for s in sentences if s.strip().endswith("!"))

    # Parenthetical expressions
    parenthetical_pattern = r"\([^)]*\)"
    parenthetical_count = len(re.findall(parenthetical_pattern, text))

    # Em dash usage
    em_dash_count = text.count("—")

    # Semicolon usage
    semicolon_count = text.count(";")

    # Sentence length variation
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    if sentence_lengths:
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        sentence_length_variance = sum(
            (length - avg_sentence_length) ** 2 for length in sentence_lengths
        ) / len(sentence_lengths)
    else:
        avg_sentence_length = 0
        sentence_length_variance = 0

    return {
        "question_percentage": round((question_count / max(1, len(sentences))) * 100, 1),
        "exclamation_percentage": round((exclamation_count / max(1, len(sentences))) * 100, 1),
        "parenthetical_count": parenthetical_count,
        "em_dash_count": em_dash_count,
        "semicolon_count": semicolon_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "sentence_length_variance": round(sentence_length_variance, 1),
        "style_complexity": calculate_style_complexity(text),
    }


def calculate_style_complexity(text: str) -> float:
    """Calculate a style complexity score based on various metrics."""
    # Get various text statistics
    stats = calculate_text_stats(text)
    reading_level = detect_reading_level(text)

    # Combine metrics to create a complexity score
    complexity = (
        0.3 * stats["lexical_diversity"]
        + 0.3 * (reading_level["flesch_kincaid_grade"] / 12)  # Normalize to 0-1 range
        + 0.2 * (reading_level["complex_words_percentage"] / 100)
        + 0.2 * min(1.0, stats["avg_sentence_length"] / 25)  # Cap at 25 words
    )

    return round(min(1.0, complexity), 2)  # Return a value between 0 and 1
