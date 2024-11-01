import re
from collections import Counter
import math
import spacy
from nltk.corpus import wordnet

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


def detect_template_phrases(response, template_phrases):
    total_phrases = len(template_phrases)
    phrase_count = 0
    for phrase in template_phrases:
        occurrences = len(re.findall(
            re.escape(phrase), response, re.IGNORECASE))
        phrase_count += occurrences
    # Normalize the score between 0 and 1
    return min(phrase_count / total_phrases, 1)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms


def check_keyword_relevancy(response, relevant_keywords):
    response_words = set(re.findall(r'\w+', response.lower()))
    keywords_present = 0
    total_keywords = len(relevant_keywords)
    if total_keywords == 0:
        return 0  # Assume full relevancy if no keywords are provided
    for keyword in relevant_keywords:
        keyword_synonyms = get_synonyms(keyword)
        keyword_synonyms.add(keyword.lower())
        if response_words & keyword_synonyms:
            keywords_present += 1
    # Calculate the proportion of relevant keywords present
    relevancy_score = 1 - (keywords_present / total_keywords)
    return relevancy_score


def measure_repetition(response):
    words = re.findall(r'\w+', response.lower())
    word_counts = Counter(words)
    total_words = len(words)
    if total_words == 0:
        return 1  # Fully templated if no words
    repeated_words = sum(
        count - 1 for count in word_counts.values() if count > 1)
    repetition_score = repeated_words / total_words
    return repetition_score


def calculate_uniqueness(response):
    words = re.findall(r'\w+', response.lower())
    unique_words = set(words)
    if len(words) == 0:
        return 1  # Fully templated if no words
    uniqueness_score = 1 - (len(unique_words) / len(words))
    return uniqueness_score


def structural_pattern_score(response):
    sentences = re.split(r'[.!?]', response)
    sentence_starters = []
    for sentence in sentences:
        tokens = sentence.strip().split()
        if tokens:
            sentence_starters.append(tokens[0].lower())
    starter_counts = Counter(sentence_starters)
    if not starter_counts:
        return 1  # Fully templated if no sentences
    most_common_starter, count = starter_counts.most_common(1)[0]
    pattern_score = count / len(sentence_starters)
    return pattern_score


def analyze_pos_patterns(response):
    doc = nlp(response)
    pos_patterns = [token.pos_ for token in doc]
    pos_counts = Counter(pos_patterns)
    total_tokens = len(pos_patterns)
    # Define expected POS patterns for templates (this can be adjusted)
    template_pos_patterns = ['PRON', 'AUX', 'DET', 'NOUN', 'VERB']
    pattern_score = sum(pos_counts.get(pos, 0)
                        for pos in template_pos_patterns) / total_tokens
    return pattern_score


def sentence_length_analysis(response):
    sentences = re.split(r'[.!?]', response)
    sentence_lengths = [len(re.findall(r'\w+', sentence))
                        for sentence in sentences if sentence.strip()]
    if not sentence_lengths:
        return 1  # Fully templated if no sentences
    avg_length = sum(sentence_lengths) / len(sentence_lengths)
    length_variance = sum((length - avg_length) **
                          2 for length in sentence_lengths) / len(sentence_lengths)
    # Normalize the variance (higher variance suggests originality)
    max_variance = avg_length ** 2  # Maximum possible variance
    normalized_variance = length_variance / max_variance if max_variance != 0 else 0
    # Invert so higher score indicates more likely templated
    return 1 - normalized_variance


def compute_confidence_score(response, template_phrases, relevant_keywords):
    # Individual scores
    template_phrase_score = detect_template_phrases(response, template_phrases)
    keyword_relevancy_score = check_keyword_relevancy(
        response, relevant_keywords)
    repetition_score = measure_repetition(response)
    uniqueness_score = calculate_uniqueness(response)
    structural_score = structural_pattern_score(response)
    pos_pattern_score = analyze_pos_patterns(response)
    sentence_length_score = sentence_length_analysis(response)

    # Combine scores with weights
    weights = {
        'template_phrase_score': 0.2,
        'keyword_relevancy_score': 0.2,
        'repetition_score': 0.15,
        'uniqueness_score': 0.1,
        'structural_score': 0.1,
        'pos_pattern_score': 0.15,
        'sentence_length_score': 0.1,
    }
    total_weight = sum(weights.values())
    combined_score = (
        weights['template_phrase_score'] * template_phrase_score +
        weights['keyword_relevancy_score'] * keyword_relevancy_score +
        weights['repetition_score'] * repetition_score +
        weights['uniqueness_score'] * uniqueness_score +
        weights['structural_score'] * structural_score +
        weights['pos_pattern_score'] * pos_pattern_score +
        weights['sentence_length_score'] * sentence_length_score
    ) / total_weight

    # Ensure score is between 0 and 1
    combined_score = min(max(combined_score, 0), 1)
    return combined_score


# Example usage
response = """
The image illustrates that there are several factors to consider.
As we can see from the graph, the numbers have increased over time.
In conclusion, it can be observed that the trend is upward.
"""

# List of template phrases to detect
template_phrases = [
    "can see",
    "element",
    "informative",
    "information regarding",
    "I",
    "i",
    "beautiful",
    "title",
    "colours",
    "color",
    "colorful",
    "colourful",
    "future reference",
    
    # Add more templated phrases here
]

# List of relevant keywords expected in the response
relevant_keywords = [
    "line chart",
    "bar graph",
    "trend",
    "increase",
    "decrease",
    "upward",
    "downward",
    "significant",
    "observed",
    "conclusion",
    
    # Add more relevant keywords here
]

confidence_score = compute_confidence_score(
    response, template_phrases, relevant_keywords)
percentage_score = confidence_score * 100
print(
    f"Confidence score that the response is templated: {percentage_score:.2f}%")

# Interpret the result
if confidence_score >= 0.7:
    print("The response is likely templated.")
elif confidence_score >= 0.4:
    print("The response may be partially templated.")
else:
    print("The response is likely original.")
