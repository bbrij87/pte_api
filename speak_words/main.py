import re
from collections import Counter
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Template pool for known templated responses
template_pool = [
    "The given image represents a . There must have been a popular debate about information present in the image regarding. Moreover, some details about and are given. Additionally, some facts about and can be seen. From the image, it is clear that the maximum value seems to be constant, which further states the importance of the  in the image. In conclusion, the given image includes complicated data and analysis with all the sufficient data.",
    "I have got an interesting and beautiful picture in front of me. Let me have a closer look at this image. By looking closely, I can see that several trends are emerging. The best part of the question is that I have to speak only for 27 seconds. Let me begin with the topic. I can see some words like . There are some numbers in the picture like . There are some beautiful colours in the image like .",
    # Add more known templated responses here
]

# Define a dictionary of vocabulary for each image type
image_type_vocab = {
    "bar_chart": ["bar chart", "bar graph", "bars", "categories", "vertical", "horizontal"],
    "line_graph": ["line graph", "line chart", "trend line", "plot", "rise", "fall"],
    "pie_chart": ["pie chart", "pie graph", "slices", "portion", "sector", "percentage"],
    "table": ["table", "grid", "rows", "columns", "data table", "matrix"],
    "scatter_plot": ["scatter plot", "scatter chart", "points", "distribution", "scatter", "correlation"],
    "histogram": ["histogram", "frequency", "bins", "distribution", "bar"],
    "diagram": ["diagram", "illustration", "visual", "representation", "model"]
}

# List of common relationship words
relationship_words = [
    "relationship", "association", "correlation", "connection", "trend", "comparison"
]

# List of template phrases
template_phrases = [
    "can see", "element", "informative", "information regarding", "I", "i", "beautiful", "title", "future reference"
]

# List of relevant keywords
relevant_keywords = ["trend", "increase", "decrease",
                     "significant", "observed", "conclusion"]


# Set up TF-IDF vectorizer for the template pool
vectorizer = TfidfVectorizer().fit(template_pool)


def check_template_pool_similarity(response, template_pool, threshold=0.7):
    """
    Check if the response is similar to any templates in the template pool using cosine similarity.
    """
    # Transform the response and template pool using the TF-IDF vectorizer
    response_vector = vectorizer.transform([response])
    template_vectors = vectorizer.transform(template_pool)

    # Compute cosine similarity between the response and each template
    similarities = cosine_similarity(
        response_vector, template_vectors).flatten()
    max_similarity = np.max(similarities)

    # Log the maximum similarity
    print(f"Max Cosine Similarity with Template Pool: {max_similarity:.2f}")

    # Return whether the similarity exceeds the threshold and the max similarity score
    return max_similarity >= threshold, max_similarity


def compute_template_score(response, image_type, tags, template_phrases, relevant_keywords, template_matching_threshold=0.7, use_template_matching=True):
    """
    Compute a template confidence score for a given response.

    Parameters:
    - response: The response text to evaluate.
    - image_type: The type of image (e.g., 'bar_chart').
    - tags: List or comma-separated string of relevant tags.
    - template_phrases: List of phrases common in templated responses.
    - relevant_keywords: List of keywords relevant to the content.
    - template_matching_threshold: Threshold for template matching similarity.
    - use_template_matching: Boolean to enable or disable template matching.

    Returns:
    - template_confidence: A score between 0 and 1 indicating the likelihood that the response is templated.
    """
    # Ensure tags is a list (in case it's provided as a comma-separated string)
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",")]

    # First, check if template matching is enabled
    if use_template_matching:
        # Check similarity with template pool
        is_similar, similarity_score = check_template_pool_similarity(
            response, template_pool, threshold=template_matching_threshold)

        # Log the similarity score
        print(f"Template Matching Similarity Score: {similarity_score:.2f}")

        if is_similar:
            print("Response is very similar to a known template.")
            template_confidence = 1.0  # Set template_confidence to 1
            return template_confidence  # Stop further processing

    # Define helper functions

    def detect_template_phrases(response, template_phrases):
        matches = [phrase for phrase in template_phrases if re.search(
            re.escape(phrase), response, re.IGNORECASE)]
        score = min(len(matches) / len(template_phrases), 1)
        print(f"Template Phrase Matches: {matches}")
        return score

    def check_keyword_relevancy(response, relevant_keywords):
        response_lemmas = {token.lemma_ for token in nlp(response)}
        matches = [keyword for keyword in relevant_keywords if response_lemmas & {
            lemma.lemma_ for lemma in nlp(keyword)}]
        score = 1 - (len(matches) / len(relevant_keywords))
        print(f"Relevant Keyword Matches: {matches}")
        return score

    def measure_repetition(response):
        words = re.findall(r'\w+', response.lower())
        word_counts = Counter(words)
        repeated_words = {word: count for word,
                          count in word_counts.items() if count > 1}
        score = sum(count - 1 for count in repeated_words.values()) / len(words)
        print(f"Repeated Words: {repeated_words}")
        return score

    def calculate_uniqueness(response):
        words = re.findall(r'\w+', response.lower())
        unique_words = set(words)
        score = 1 - (len(unique_words) / len(words))
        print(f"Unique Words: {unique_words}")
        return score

    def calculate_lexical_diversity(response):
        words = re.findall(r'\w+', response.lower())
        unique_words = set(words)
        score = len(unique_words) / len(words)
        print(f"Lexical Diversity - Unique Words: {unique_words}")
        return score

    def structural_pattern_score(response):
        sentences = re.split(r'[.!?]', response)
        sentence_starters = [sentence.strip().split()[0].lower()
                             for sentence in sentences if sentence.strip()]
        starter_counts = Counter(sentence_starters)
        most_common_starter, count = starter_counts.most_common(
            1)[0] if starter_counts else ('', 0)
        score = count / len(sentence_starters) if sentence_starters else 1
        print(
            f"Sentence Starters: {sentence_starters}, Most Common Starter: '{most_common_starter}'")
        return score

    def analyze_pos_patterns(response):
        doc = nlp(response)
        pos_patterns = [token.pos_ for token in doc]
        pos_counts = Counter(pos_patterns)
        template_pos_patterns = ['PRON', 'AUX', 'DET', 'NOUN', 'VERB']
        score = sum(pos_counts.get(pos, 0)
                    for pos in template_pos_patterns) / len(pos_patterns)
        print(f"POS Pattern Matches: {pos_counts}")
        return score

    def sentence_length_analysis(response):
        sentences = re.split(r'[.!?]', response)
        sentence_lengths = [len(re.findall(r'\w+', sentence))
                            for sentence in sentences if sentence.strip()]
        avg_length = sum(sentence_lengths) / \
            len(sentence_lengths) if sentence_lengths else 0
        variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / \
            len(sentence_lengths) if sentence_lengths else 0
        max_variance = avg_length ** 2
        score = 1 - (variance / max_variance if max_variance != 0 else 0)
        print(f"Sentence Lengths: {sentence_lengths}, Variance: {variance}")
        return score

    def check_image_type_and_relationship(response, image_type_vocab, relationship_words, max_score=0.15, score_per_match=0.05):
        """
        Check for the presence of image type vocabulary and relationship words in the response
        and calculate a reduction score with a cap.
        """
        # Convert response to lowercase for case-insensitive matching
        response = response.lower()

        # Find matches for image type vocabulary
        image_type_terms = image_type_vocab.get(image_type, [])
        image_type_matches = set(term for term in image_type_terms if re.search(
            r'\b' + re.escape(term) + r'\b', response))

        # Find matches for relationship words
        relationship_matches = set(word for word in relationship_words if re.search(
            r'\b' + re.escape(word) + r'\b', response))

        # Calculate total matches and cap the score
        total_matches = len(image_type_matches) + len(relationship_matches)
        score = min(total_matches * score_per_match, max_score)

        print(
            f"Image Type Matches: {image_type_matches}, Relationship Word Matches: {relationship_matches}")
        print(f"Reduction Score: {score:.2f} (from {total_matches} matches)")

        return score

    def content_relevancy_score(response, tags_list):
        response_lemmas = {token.lemma_ for token in nlp(response)}
        tags_lemmas = {token.lemma_ for token in nlp(" ".join(tags_list))}
        matched_tags = response_lemmas & tags_lemmas
        score = 1 - (len(matched_tags) / len(tags_list)) if tags_list else 1
        print(f"Content Relevant Tags Matched: {matched_tags}")
        return score

    def sentence_penalty(response, tags_list):
        penalty = 0.0
        sentences = re.split(r'[.!?]', response)
        tag_lemmas = {token.lemma_ for token in nlp(" ".join(tags_list))}
        penalized_sentences = []

        for sentence in sentences:
            if sentence.strip():
                sentence_lemmas = {token.lemma_ for token in nlp(sentence)}
                if not sentence_lemmas & tag_lemmas:
                    penalty += 0.1
                    penalized_sentences.append(sentence.strip())

        print(f"Penalized Sentences (No Tag Matches): {penalized_sentences}")
        print(f"Penalty Score for Non-Matching Sentences: {penalty:.2f}")
        return penalty

    # Calculate individual scores
    template_phrase_score = detect_template_phrases(response, template_phrases)
    keyword_relevancy_score = check_keyword_relevancy(
        response, relevant_keywords)
    repetition_score = measure_repetition(response)
    uniqueness_score = calculate_uniqueness(response)
    lexical_diversity_score = calculate_lexical_diversity(response)
    structural_score = structural_pattern_score(response)
    pos_pattern_score = analyze_pos_patterns(response)
    sentence_length_score = sentence_length_analysis(response)
    reduction_score = check_image_type_and_relationship(
        response, image_type_vocab, relationship_words)
    content_score = content_relevancy_score(response, tags)

    # Calculate penalty for sentences with no tag matches
    penalty_score = sentence_penalty(response, tags)

    # Weights for each score
    weights = {
        'template_phrase_score': 0.1,
        'keyword_relevancy_score': 0.1,
        'repetition_score': 0.1,
        'uniqueness_score': 0.1,
        'lexical_diversity_score': 0.1,
        'structural_score': 0.1,
        'pos_pattern_score': 0.1,
        'sentence_length_score': 0.1,
        'content_score': 0.1
    }

    # Calculate weighted scores
    weighted_template_phrase_score = weights['template_phrase_score'] * \
        template_phrase_score
    weighted_keyword_relevancy_score = weights['keyword_relevancy_score'] * \
        keyword_relevancy_score
    weighted_repetition_score = weights['repetition_score'] * repetition_score
    weighted_uniqueness_score = weights['uniqueness_score'] * uniqueness_score
    weighted_lexical_diversity_score = weights['lexical_diversity_score'] * \
        lexical_diversity_score
    weighted_structural_score = weights['structural_score'] * structural_score
    weighted_pos_pattern_score = weights['pos_pattern_score'] * \
        pos_pattern_score
    weighted_sentence_length_score = weights['sentence_length_score'] * \
        sentence_length_score
    weighted_content_score = weights['content_score'] * content_score

    # Print weighted scores
    print("\nWeighted Scores:")
    print(
        f"Weighted Template Phrase Score: {weighted_template_phrase_score:.2f}")
    print(
        f"Weighted Keyword Relevancy Score: {weighted_keyword_relevancy_score:.2f}")
    print(f"Weighted Repetition Score: {weighted_repetition_score:.2f}")
    print(f"Weighted Uniqueness Score: {weighted_uniqueness_score:.2f}")
    print(
        f"Weighted Lexical Diversity Score: {weighted_lexical_diversity_score:.2f}")
    print(
        f"Weighted Structural Pattern Score: {weighted_structural_score:.2f}")
    print(f"Weighted POS Pattern Score: {weighted_pos_pattern_score:.2f}")
    print(
        f"Weighted Sentence Length Score: {weighted_sentence_length_score:.2f}")
    print(f"Weighted Content Relevancy Score: {weighted_content_score:.2f}")

    # Calculate the combined score using weighted values
    combined_score = (
        weighted_template_phrase_score +
        weighted_keyword_relevancy_score +
        weighted_repetition_score +
        weighted_uniqueness_score +
        weighted_lexical_diversity_score +
        weighted_structural_score +
        weighted_pos_pattern_score +
        weighted_sentence_length_score -
        weighted_content_score
    ) / sum(weights.values())

    # Apply reduction and final normalization
    combined_score = max(0, combined_score - reduction_score + penalty_score)
    combined_score = min(combined_score, 1)

    print(
        f"\nCombined Weighted Score (before reduction): {combined_score + reduction_score:.2f}")
    print(
        f"Reduction Score (Image Type and Relationship): {reduction_score:.2f}")
    print(f"Penalty Score (No Tag Matches): {penalty_score:.2f}")
    print(f"Final Combined Score: {combined_score:.2f}")

    # Set the template confidence as the final combined score
    template_confidence = combined_score
    return template_confidence


# Example API call
response_text = """
I have got an interesting and beautiful picture in front of me. Let me have a closer look at this image. By looking closely, I can see that several trends are emerging. The best part of the question is that I have to speak only for 27 seconds. Let me begin with the topic. The title of the image is Vehicle colour involved in total-loss collision. I can see some words like vehicle, colour, collision, frequency. There are some numbers in the picture like 55, 50, 30. There are some beautiful colours in the image like red, green, blue, black, and white.
"""

# best for template till now
# response_text = """
# The bar chart depicts the relationship between vehicle color and total-loss collision frequency. Green vehicles show the highest accident frequency, while blue vehicles have the lowest. This trend suggests that certain colors may have a higher propensity for involvement in accidents.
# """

# response_text = """
# The given bar chart gives information regarding vehicle colors involved in total-loss collisions. It is evident from the bar chart that the highest frequency of total-loss collisions is for green-colored vehicles, whereas the lowest frequency is for blue-colored vehicles. Moreover, it shows a variation in collision frequency based on color involvement. Overall, it is an informative image and can be used for analyzing accident statistics and traffic safety trends.
# """

# response_text = """
# The given image gives the information about vehicle color involvement in total-loss collisions. There are many elements in this image. From the image, I can see that the highest value is for collisions involving green-colored vehicles, which is indicated by the tallest bar. Now the second highest value is for collisions involving red-colored vehicles, which is indicated by a slightly shorter bar.If we talked about the lowest value, that is for blue-colored vehicles, which is indicated by the shortest bar. To conclude, this image is very informative and interesting, providing insights into traffic safety and accident data analysis
# """

# response_text = """
# The given image represents a bar graph. There must have been a popular debate about information present in the image regarding vehicle safety, collision frequency, and colour impact on accidents. Moreover, some details aboutvehicle colours and accident rates are given. Additionally, some facts about different colours and safety trends can be seen. From the image, it is clear that the maximum value seems to be constant, which further states the importance of the accident data in the image. In conclusion, the given image includes complicated data and analysis with all the sufficient data.
# """
# non templated response
# response_text = """
# According to the bar chart, green-colored vehicles are involved in the most total-loss collisions, while blue vehicles have the fewest. Red and black colors also show relatively high accident frequencies. This data might suggest an association between vehicle color and accident likelihood.
# """

# my tempalte
# response_text = """
# The bar graph image is about vehicle colour involved in total-loss collision. From it, it is noticeable that green has the highest occurrence, with a value of 55. This is followed closely by red and black, which also show relatively high values of 45 and 40. On the other end, blue has the lowest value at 30. Overall, image provides useful insights into vehicle collisions by colour, which could be valuable for safety analysis.
# """

image_type = "bar_chart"

tags = "vehicle, collision, frequency, accident, blue, green, red, white, black, grey, high, low, comparison, involvement, statistics, traffic, safety, analysis"

# Compute the template confidence score
template_confidence = compute_template_score(
    response_text, image_type, tags, template_phrases, relevant_keywords, template_matching_threshold=0.7, use_template_matching=False)

percentage_score = template_confidence * 100
print(
    f"\nConfidence score that the response is templated: {percentage_score:.2f}%")

# Interpret the result
if template_confidence >= 0.53:
    print("The response is most probably templated.")
elif template_confidence >= 0.45:
    print("The response may be likely templated.")
elif template_confidence >= 0.38:
    print("The response may be templated.")
else:
    print("The response is likely original.")
