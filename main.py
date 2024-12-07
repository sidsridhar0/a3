import heapq
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
import spacy
import nltk
import re
from nltk.stem import PorterStemmer

app = Flask(__name__)

ROOT_DIR = "ANALYST"
index_lock = Lock()  # for thread-safe access to inverted index
inverted_index = defaultdict(list)
term_cache = {}
create_tokens = spacy.load("en_core_web_sm")
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
    # Tokenize the text into words using spacy
    doc = create_tokens(text.lower())
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words without punctuation
    ngrams = []

    # If ngram is 0, set it to the length of words + 1 to generate all possible ngrams
    if ngram == 0:
        ngram = len(words) + 1

    # Generate non-stemmed n-grams (preserve multi-word phrases)
    for n in range(2, 5):  # 2-grams and 4-grams
        ngrams.extend([" ".join(words[i:i + n]) for i in range(len(words) - n + 1)])

    # Porter stemming process for individual tokens (don't stem n-grams)
    token_stemming = [stemmer.stem(token.text) for token in doc]

    # Generate stemmed n-grams
    for n in range(2, 5):  # 2-grams and 4-grams for stemmed words
        ngrams.extend([" ".join(token_stemming[i:i + n]) for i in range(len(token_stemming) - n + 1)])

    # Combine individual words (non-stemmed) with n-grams (both stemmed and non-stemmed)
    tokens = words + ngrams
    return tokens


def tokenize_query(query, ngram=0):
    # Tokenize the query into words using spacy and basic preprocessing
    doc = create_tokens(query.lower())
    words = re.findall(r'\b\w+\b', query.lower())  # Extract words without punctuation
    ngrams = []

    # If ngram is 0, set it to the length of words + 1 to generate all possible ngrams
    if ngram == 0:
        ngram = len(words) + 1

    # Generate non-stemmed n-grams (preserve multi-word phrases) for the query
    for n in range(2, 5):
        ngrams.extend([" ".join(words[i:i + n]) for i in range(len(words) - n + 1)])

    # Combine individual words (non-stemmed) with n-grams (both stemmed and non-stemmed)
    tokens = words + ngrams
    return tokens


# Update the search query handling route to use the new `tokenize_query` function
@app.route('/search', methods=['GET'])
def search_query():
    search_time = time.time()
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query parameter provided"}), 400

    # Use tokenize_query for search query
    SEARCH_TERMS = tokenize_query(query)  # Tokenize the query differently
    search_results, query_doc_count = search_terms(SEARCH_TERMS, inverted_index)

    # Calculate the frequency and tf-idf for the search results
    top_urls = calculate_frequency_tfidf(SEARCH_TERMS, inverted_index, query_doc_count)

    # Apply Jaccard ranking on top URLs based on query terms
    ranked_docs, doc_scores = ranking(SEARCH_TERMS, inverted_index)
    print(f"Ranked docs: {ranked_docs} \n Docked scores: {doc_scores}")
    # Create a result set with the documents ranked by Jaccard similarity
    result_data = set()
    for doc_id, _ in ranked_docs:
        for term, postings in top_urls.items():
            for posting in postings:
                if posting['id'] == doc_id:
                    result_data.add({
                        'term': term,
                        'doc_id': doc_id,
                        'jaccard_score': doc_scores[doc_id]
                    })

    finish_search = time.time()
    logging.info(f"Search time {finish_search - search_time} seconds")
    return jsonify(list(result_data))


def ranking(terms, index, top_n=5, alpha=0.7):
    doc_scores = defaultdict(float)

    # Loop through the terms and compute Jaccard similarity for each posting
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                # Calculate Jaccard similarity between the query and the document
                jaccard_score = calculate_jaccard(posting, terms)

                # Combine Jaccard score with the document's TF-IDF score
                tfidf_score = posting.get('tf-idf', 0)
                final_score = alpha * tfidf_score + (1 - alpha) * jaccard_score

                # Sum the final scores to rank the document based on total similarity
                doc_scores[posting['id']] += final_score

    # Sort the documents by their total score and get the top_n
    ranked_docs = heapq.nlargest(top_n, doc_scores.items(), key=itemgetter(1))

    return ranked_docs, doc_scores


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
                posting['tf-idf'] = tf * idf[term]  # Compute TF-IDF

            sorted_postings = sorted(postings, key=itemgetter('tf-idf'), reverse=True)
            top_urls[term] = sorted_postings[:top_n]  # Get the top_n postings
    # print(calculate_jaccard(posting, terms))
    return top_urls


# A ^ B / A U B
def calculate_jaccard(posting, terms):
    # Convert both posting terms and query terms to sets
    document_terms = set(posting['id'].split())
    query_terms = set(terms)

    # Calculate intersection and union
    intersection = len(document_terms & query_terms)
    union = len(document_terms | query_terms)

    if union == 0:
        return 0  # Avoid division by zero if both sets are empty
    return intersection / union


def get_html(path):
    if path not in term_cache:
        with open(path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            term_cache[path] = soup.get_text()
    return term_cache[path]


def get_token_freq(tokens):
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    return freq


# Processing a file and constructing its local index
# Process each document and calculate TF-IDF
def process_file(doc_path, root, doc_count):
    doc_id = os.path.relpath(doc_path, root)
    doc_text = get_html(doc_path)
    tokens = tokenize_document(doc_text, 4)
    tf = get_token_freq(tokens)

    local_index = defaultdict(list)

    # First pass: accumulate term frequencies
    for position, token in enumerate(tokens):
        local_index[token].append({
            "id": doc_id,
            "frq": tf[token],
            "positions": [position]
        })

    # Second pass: calculate tf-idf for each token
    for token, postings in local_index.items():
        term_freq = len(postings)  # Number of documents containing this token
        idf = log(doc_count / (term_freq or 1))  # Avoid division by zero
        for posting in postings:
            posting["tfidf"] = posting["frq"] * idf

    return local_index


# Indexing the entire collection
def inv_index(root, max_workers=4, split_count=3):
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


def new_search_terms(terms, index):
    res = defaultdict(list)
    for t in terms:
        if t in index:
            res[t] = heapq.nlargest(5, index[t], key=lambda x: x['tf-idf'])
    return res


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
