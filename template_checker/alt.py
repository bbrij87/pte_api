import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from nltk.corpus import wordnet
import numpy as np

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Step 1: Extract dynamic N-grams and store them


def extract_ngrams(response, n=2):
    words = re.findall(r'\w+', response.lower())
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# Step 2: Detect common phrases without a predefined template pool


def detect_template_phrases(response, stored_responses):
    ngrams_list = extract_ngrams(response, n=2)
    all_ngrams = Counter()
    for stored_response in stored_responses:
        all_ngrams.update(extract_ngrams(stored_response, n=2))

    common_phrases = [phrase for phrase,
                      freq in all_ngrams.items() if freq > 3]
    template_phrases_score = sum(
        1 for phrase in ngrams_list if phrase in common_phrases) / max(len(ngrams_list), 1)
    return template_phrases_score

# Step 3: Compute similarity to past responses (cosine similarity)


def calculate_similarity_score(response, stored_responses):
    all_responses = stored_responses + [response]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_responses)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return max(similarities[0])

# Reuse your existing functions with enhancements


def measure_repetition(response):
    words = re.findall(r'\w+', response.lower())
    word_counts = Counter(words)
    total_words = len(words)
    repeated_words = sum(
        count - 1 for count in word_counts.values() if count > 1)
    repetition_score = repeated_words / total_words
    return repetition_score


def calculate_uniqueness(response):
    words = re.findall(r'\w+', response.lower())
    unique_words = set(words)
    uniqueness_score = 1 - (len(unique_words) / len(words))
    return uniqueness_score


def structural_pattern_score(response):
    sentences = re.split(r'[.!?]', response)
    sentence_starters = []
    for sentence in sentences:
        tokens = sentence.strip().split()
        if tokens:  # Check if tokens list is not empty
            sentence_starters.append(tokens[0].lower())
    starter_counts = Counter(sentence_starters)
    if not starter_counts:
        return 1  # Fully templated if no sentences
    most_common_starter, count = starter_counts.most_common(1)[0]
    pattern_score = count / len(sentence_starters)
    return pattern_score


# Adding updated confidence score calculation


def compute_confidence_score(response, stored_responses, relevant_keywords):
    # Extract template phrases dynamically
    template_phrases_score = detect_template_phrases(
        response, stored_responses)
    similarity_score = calculate_similarity_score(response, stored_responses)
    repetition_score = measure_repetition(response)
    uniqueness_score = calculate_uniqueness(response)
    structural_score = structural_pattern_score(response)

    # Combine scores with weights
    weights = {
        'template_phrases_score': 0.25,
        'similarity_score': 0.25,
        'repetition_score': 0.15,
        'uniqueness_score': 0.15,
        'structural_score': 0.2
    }
    combined_score = (
        weights['template_phrases_score'] * template_phrases_score +
        weights['similarity_score'] * similarity_score +
        weights['repetition_score'] * repetition_score +
        weights['uniqueness_score'] * uniqueness_score +
        weights['structural_score'] * structural_score
    )

    return min(max(combined_score, 0), 1)


# Example usage
stored_responses = [
    "This graph shows an increasing trend over time, indicating positive growth.",
    "The image illustrates several factors, and the trend is upwards as shown.",
    "It is observed that numbers have increased significantly, suggesting an upward trend."
]

response = """
The image illustrates that there are several factors to consider.
As we can see from the graph, the numbers have increased over time.
In conclusion, it can be observed that the trend is upward.
"""

confidence_score = compute_confidence_score(
    response, stored_responses, relevant_keywords=[])
percentage_score = confidence_score * 100
print(
    f"Confidence score that the response is templated: {percentage_score:.2f}%")

# Interpretation
if confidence_score >= 0.7:
    print("The response is likely templated.")
elif confidence_score >= 0.4:
    print("The response may be partially templated.")
else:
    print("The response is likely original.")
