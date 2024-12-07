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


def term_doc_matrix(search_query):
    # IGNORE FOR NOW
    # iterate through each word in the query
    # dict[url] = number of appearances, start at 0
    # for each word
    #   get inv idx
    #       add all urls or ++ to urls that exist
    # now we have dict of unique urls that exist
    # check frequency of urls with number == to number of terms

    # OR

    # iterate through search terms, look for ngrams of 3
    # do same as before except iterate through each 3-gram instead of word

    # returns dict with list of urls and frequency of full query
    pass


# tokenizing using spacy (with permission from professor)
def tokenize(text):
    doc = create_tokens(text.lower())
    words = re.findall(r'\b\w+\b', text.lower())
    ngrams = []

    # porter stemming process for each token in the doc
    token_stemming = [stemmer.stem(token.text) for token in doc]
    # Generate 2-grams and 4-grams
    # eventually change to 2 and 3-grams
    for n in range(2, 5):
        ngrams.extend([" ".join(token_stemming[i: i + n]) for i in range(len(token_stemming) - n + 1)])
        ngrams.extend([" ".join(words[i: i + n]) for i in range(len(words) - n + 1)])
    # Combine individual words with n-grams
    tokens = token_stemming + ngrams
    return tokens


# def tokenize_without_stemming(text):
#
#     # Generate 2-grams and 4-grams
#     for n in range(2, 5):
#         ngrams.extend([" ".join(words[i: i + n]) for i in range(len(words) - n + 1)])
#
#     # Combine individual words with n-grams
#     tokens = words + ngrams
#     return tokens

def ranking(terms, index, top_n):
    #calcfreq
    #calcjaccard
    #sort
    pass

def calculate_frequency_tfidf(terms, index, doc_count, top_n=5):
    postings = defaultdict(list)
    top_urls = defaultdict(list)
    terms = [term.lower() for term in terms]

    # Calculate IDF for each term
    idf = {term: log(doc_count / (len(index.get(term, [])) + 1)) for term in terms}
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                tf = posting['frq']
                posting['tf-idf'] = tf * idf[term]  # Compute TF-IDF
            sorted_postings = sorted(postings, key=itemgetter('tf-idf'), reverse=True)
            top_urls[term] = sorted_postings[:top_n]  # Get the top_n postings

    return top_urls



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
def process_file(doc_path, root, doc_count):
    doc_id = os.path.relpath(doc_path, root)
    doc_text = get_html(doc_path)
    tokens = tokenize(doc_text)
    tf = get_token_freq(tokens)

    local_index = defaultdict(list)
    for position, token in enumerate(tokens):
        local_index[token].append({
            "id": doc_id,
            "frq": tf[token],
            "positions": [position],
            "tfidf": tf[token] * log(doc_count / (len(local_index.get(token, [])) + 1))
        })
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


# Function to load the final index
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


# Searching through the index
@app.route('/search', methods=['GET'])
def search_query():
    search_time = time.time()
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query parameter provided"}), 400
    SEARCH_TERMS = tokenize(query)
    search_results, query_doc_count = search_terms(SEARCH_TERMS, inverted_index)

    # dont need?
    top_urls = calculate_frequency_tfidf(SEARCH_TERMS, inverted_index, query_doc_count)

    result_data = set()
    for term, postings in top_urls.items():
        for posting in postings:
            result_data.add((
                term,
                posting['id'],
                # posting['frq']
            ))
    result_data = [{"term": posting_id} for posting_id in result_data]
    finish_search = time.time()
    logging.info(f"Search time {finish_search - search_time} seconds")
    return jsonify(result_data)


# Searching terms in the index
def search_terms(terms, index):
    results = defaultdict(list)
    # keep track of how many query terms match each document
    doc_matches = defaultdict(int)
    # doc_coverages = defaultdict(int)
    total_docs = len([doc for postings in index.values() for doc in postings])  # Count total docs

    # Iterate through each term in the query
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                doc_matches[posting['id']] += 1  # increment match count for each doc

    # filter out the docs that matched all terms in the query want documents that match both terms
    result_docs = {doc_id: count for doc_id, count in doc_matches.items() if count == len(terms)}

    # Retrieve the postings for those documents that matched all terms
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
