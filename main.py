import os
import json
import time
import logging
from collections import defaultdict
from operator import itemgetter
from flask import Flask, request, jsonify, render_template
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from math import log
import nltk
import re
from nltk.stem import PorterStemmer

app = Flask(__name__)

ROOT_DIR = "DEV"
index_lock = Lock()  # for thread-safe access to inverted index
inverted_index = defaultdict(list)
term_cache = {}
nltk.download('punkt')  # punkt = stemmer data
stemmer = PorterStemmer()
# Set up logging
logging.basicConfig(level=logging.INFO)


# save partial inverted index
def save_partial_index(partial_index, part_num):
    partial_index_file = f"partial_inv_idx_{part_num}.json"
    with open(partial_index_file, "w", encoding="utf-8") as f:
        json.dump(partial_index, f, indent=4)
    logging.info(f"Saved partial index {part_num} to {partial_index_file}.")


# Function to load a partial index
def load_partial_index(part_num):
    partial_index_file = f"partial_inv_idx_{part_num}.json"
    if os.path.exists(partial_index_file):
        with open(partial_index_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Merge two indexes (used after loading partial indexes)
def merge_indices(global_index, local_index):
    with index_lock:
        for token, postings in local_index.items():
            if token not in global_index:
                global_index[token] = []
            global_index[token].extend(postings)


# Function to tokenize document text
def tokenize_document(text, ngram=0):
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words without punctuation
    ngrams = []

    # If ngram is 0, set it to the length of words + 1 to generate all possible ngrams
    if ngram == 0:
        ngram = len(words) + 1

    # Generate non-stemmed n-grams (preserve multi-word phrases)
    for n in range(2, 5):  # 2-grams and 4-grams
        ngrams.extend([" ".join(words[i:i + n]) for i in range(len(words) - n + 1)])

    # Porter stemming process for individual tokens (don't stem n-grams)
    token_stemming = []
    try:
        token_stemming = [stemmer.stem(token.text) for token in words]
    except AttributeError as e:
        token_stemming = [stemmer.stem(token) for token in words]

    # Generate stemmed n-grams
    for n in range(2, 5):  # 2-grams and 4-grams for stemmed words
        ngrams.extend([" ".join(token_stemming[i:i + n]) for i in range(len(token_stemming) - n + 1)])

    # Combine individual words (non-stemmed) with n-grams (both stemmed and non-stemmed)
    tokens = words + ngrams
    return tokens


def tokenize_query(query, ngram=0):
    # Tokenize the query into words using spacy and basic preprocessing
    words = re.findall(r'\b\w+\b', query.lower())  # Extract words without punctuation
    ngrams = []

    # If ngram is 0, set it to the length of words + 1 to generate all possible ngrams
    if ngram == 0:
        ngram = len(words) + 1

    # Generate non-stemmed n-grams (preserve multi-word phrases) for the query
    for n in range(2, 5):  # Ensure we're capturing 2-grams (like "data science")
        ngrams.extend([" ".join(words[i:i + n]) for i in range(len(words) - n + 1)])

    # Combine individual words (non-stemmed) with n-grams (both stemmed and non-stemmed)
    tokens = words + ngrams
    return tokens


# Update the search query handling route to use the new tokenize_query function
@app.route('/search', methods=['GET'])
def search_query():
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query parameter provided"}), 400

    # Use tokenize_query for search query
    search_time = time.time()
    SEARCH_TERMS = tokenize_query(query)  # Tokenize the query differently
    search_results, query_doc_count = search_terms(SEARCH_TERMS, inverted_index)

    # Calculate the frequency and tf-idf for the search results
    top_urls = ranking(SEARCH_TERMS, inverted_index, query_doc_count, top_n=5)
    # top_urls = calculate_frequency_idf(SEARCH_TERMS, inverted_index, query_doc_count)
    finish_search = time.time()

    result_data = set()
    for term, postings in top_urls.items():
        for posting in postings:
            result_data.add((
                posting['id'],
            ))
    result_data = [{"term": posting_id} for posting_id in result_data]
    logging.info(f"Search time {finish_search - search_time} seconds")
    return jsonify(result_data)


def ranking(terms, index, doc_count, top_n):
    # get top n tfidf, already sorted
    top_urls = calculate_frequency_tfidf(terms, index, doc_count, top_n)

    # calcjaccard
    top_jaccard = []
    for key, v in top_urls.items():
        for k in v:
            jaccard_coefficient = calculate_jaccard(key, terms)
            k['jaccard'] = jaccard_coefficient
            # print("V0", k)

    # sort by jaccard
    print(top_jaccard)
    for key in top_urls:
        top_urls[key].sort(key=lambda x: x.get('jaccard', 0), reverse=True)
    print(top_urls)
    return top_urls


# A ^ B / A U B
def calculate_jaccard(posting, terms):
    intersection = 0
    for term in terms:
        intersection = len(set(term) & set(posting))
    union = len(set(terms) | set(posting))
    return intersection / union


def calculate_frequency_tfidf(terms, index, doc_count, top_n=5):
    top_urls = defaultdict(list)
    terms = [term.lower() for term in terms]

    # Calculate IDF for each term
    idf = {term: log(doc_count / (len(index.get(term, [])) or 1)) for term in terms}
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                tf = posting['frq']
                posting['tfidf'] = tf * idf[term]  # Compute TF-IDF

            sorted_postings = sorted(postings, key=itemgetter('tfidf'), reverse=True)
            top_urls[term] = sorted_postings[:top_n]  # Get the top_n postings
    return top_urls


def get_html(path):
    if path not in term_cache:
        with open(path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            term_cache[path] = soup.get_text()
    return term_cache[path]


def get_html_tags(path):
    # this function returns a list of HTML tags corresponding to each token in the document.
    # The tags are aligned with the tokens (words) in the document text.
    with open(path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    tags = []

    # Traverse all text elements and associate each token with its parent tag
    for element in soup.descendants:
        if isinstance(element, str):  # Only process text nodes
            text = element.strip()
            if text:
                # Tokenize the text into words, split by whitespace
                tokens = text.split()
                parent_tag = element.parent.name if element.parent else None
                # Append the tag to match the number of tokens
                tags.extend([parent_tag] * len(tokens))

    return tags


def get_token_freq(tokens):
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    return freq


# Processing a file and constructing its local index
def process_file(doc_path, root, doc_count):
    doc_id = os.path.relpath(doc_path, root)
    doc_text = get_html(doc_path)
    tokens = tokenize_document(doc_text, 4)
    tf = get_token_freq(tokens)

    token_tag = get_html_tags(doc_path)
    tag_weightage = {
        "h1": 2,
        "h2": 2,
        "h3": 2,
        "strong": 2,
        "b": 2,
        "p": 1,
    }

    # accumulate term frequencies
    local_index = defaultdict(list)

    for position, (token, token_tag) in enumerate(zip(tokens, token_tag)):
        weight = tag_weightage.get(token_tag, 1)
        local_index[token].append({
            "id": doc_id,
            "frq": tf[token] * weight,
            "positions": [position]
        })

    # Second pass: calculate tf-idf for each token
    for token, postings in local_index.items():
        term_freq = len(postings)  # Number of documents containing this token
        if 0 < term_freq <= doc_count:
            idf = log(doc_count / term_freq)
        else:
            idf = 0  # If the term doesn't appear in any document, set IDF to 0

        # Update the tf-idf value for each posting
        for posting in postings:
            posting["tfidf"] = posting["frq"] * idf

    return local_index


# Indexing the entire collection
def inv_index(root, max_workers=4, split_count=15):
    doc_count = 0
    all_files = []
    for s, _, fs in os.walk(root):
        for file in fs:
            all_files.append(os.path.join(s, file))

    partial_index = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file_path, root, doc_count): file_path for file_path in all_files}
        for future in as_completed(futures):
            local_index = future.result()
            merge_indices(partial_index, local_index)
            doc_count += 1

            # Offload the index to disk at regular intervals
            if doc_count % (len(all_files) // split_count) == 0:
                part_num = doc_count // (len(all_files) // split_count)
                save_partial_index(partial_index, part_num)
                partial_index.clear()  # Clear the partial index after saving

            if doc_count % 100 == 0:
                logging.info(f"Processed {doc_count} files")

    # Save any remaining partial index at the end
    if partial_index:
        part_num = split_count
        save_partial_index(partial_index, part_num)

    return doc_count


# Merging all partial indexes into the final inverted index
def merge_partial_indexes():
    global inverted_index
    inverted_index = defaultdict(list)
    part_num = 1
    while os.path.exists(f"partial_inv_idx_{part_num}.json"):
        partial_index = load_partial_index(part_num)
        merge_indices(inverted_index, partial_index)
        part_num += 1
    logging.info(f"Merged {part_num - 1} partial indexes into the final index.")


# Saving the final index
def make_file(file_name="inv_idx.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)
    logging.info(f"Final index saved to {file_name}.")


# load the final index
def load_index():
    global inverted_index
    if os.path.exists("inv_idx.json"):
        with open("inv_idx.json", "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        logging.info("Loaded inverted index from file.")
    else:
        logging.info("Inverted index file not found. Running indexing...")
        index_documents()


# Indexing documents
def index_documents():
    logging.info("Starting indexing...")
    index_time = time.time()
    doc_count = inv_index(ROOT_DIR, max_workers=4, split_count=3)
    save_time = time.time()
    logging.info(f"Indexing time: {save_time - index_time} seconds")
    logging.info("Indexing complete. Merging partial indexes...")
    merge_partial_indexes()
    logging.info("Merging complete. Saving final index...")
    make_file()
    end_time = time.time()
    logging.info(f"Save time: {end_time - save_time} seconds")


@app.route('/')
def home():
    return render_template('index.html')


# Searching terms in the index
def search_terms(terms, index):
    results = defaultdict(list)
    doc_matches = defaultdict(int)
    exact_phrases = []  # To store exact phrases found in the query
    total_docs = len([doc for postings in index.values() for doc in postings])

    # Create n-grams from the query terms
    ngrams = [" ".join(terms[i:i + 2]) for i in range(len(terms) - 1)]  # 2-grams
    ngrams += [" ".join(terms[i:i + 3]) for i in range(len(terms) - 2)]  # 3-grams

    # Check for exact phrase matches in the index
    for ngram in ngrams:
        if ngram in index:
            exact_phrases.append(ngram)

    # Search for exact matches first
    for phrase in exact_phrases:
        postings = index[phrase]
        for posting in postings:
            doc_matches[posting['id']] += 1  # Match with exact phrase

    # Search for individual terms if necessary
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                doc_matches[posting['id']] += 1  # Match with individual term

    # Filter docs that match all query terms (exact phrase or individual terms)
    result_docs = {doc_id: count for doc_id, count in doc_matches.items() if count >= len(terms)}

    # Retrieve the postings for documents that matched
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                if posting['id'] in result_docs:
                    results[term].append(posting)

    return results, total_docs


# Start the indexing process in the background only if the index doesn't exist
def load_or_index_documents():
    if not os.path.exists("inv_idx.json"):
        logging.info("Inverted index not found. Starting indexing in the background.")
        indexing_thread = Thread(target=index_documents)
        indexing_thread.daemon = True  # Ensure it exits when the main program exits
        indexing_thread.start()
    else:
        logging.info("Inverted index found. Loading from file.")
        load_index()  # Load the index if it already exists


if __name__ == "__main__":
    load_or_index_documents()  # Start indexing if needed, or load the index
    app.run(debug=False)
