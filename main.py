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
from transformers import DistilBertTokenizer
import torch
from dotenv import load_dotenv
import numpy as np

app = Flask(__name__)


HF_TOKEN = os.getenv("HF_TOKEN")
ROOT_DIR = "DEV"
inverted_index = defaultdict(list)
index_lock = Lock()  # for thread-safe access to inverted index
token = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
term_cache = {}

# Set up logging
logging.basicConfig(level=logging.INFO)


def calculate_frequency_tfidf(terms, index, doc_count, top_n=5):
    top_urls = defaultdict(list)
    terms = [term.lower() for term in terms]

    # Calculate IDF for each term
    idf = {term: log(doc_count / (len(index.get(term, [])) + 1)) for term in terms}
    for term in terms:
        if term in index:
            postings = index[term]
            for posting in postings:
                tf = posting['frq']
                posting['tf-idf'] = tf * idf[term]
            sorted_postings = sorted(postings, key=itemgetter('tf-idf'), reverse=True)
            top_urls[term] = sorted_postings[:top_n]

    return top_urls


def tokenize(text):
    tokens = token.tokenize(text.lower())
    return tokens


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


def process_file(doc_path, root):
    doc_id = os.path.relpath(doc_path, root)
    doc_text = get_html(doc_path)
    tokens = tokenize(doc_text)
    tf = get_token_freq(tokens)

    local_index = defaultdict(list)
    for t, f in tf.items():
        local_index[t].append({"id": doc_id, "frq": f})
    return local_index


def merge_indices(global_index, local_index):
    # ensure thread-safe modification of global index
    with index_lock:
        for token, postings in local_index.items():
            if token not in global_index:
                global_index[token] = []  # Initialize as an empty list if not exists
            global_index[token].extend(postings)


def inv_index(root, max_workers=4):
    doc_count = 0
    all_files = []
    local_index = defaultdict(list)
    for s, _, fs in os.walk(root):
        for file in fs:
            all_files.append(os.path.join(s, file))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file_path, root): file_path for file_path in all_files}
        for future in as_completed(futures):
            local_index = future.result()
            merge_indices(inverted_index, local_index)
            doc_count += 1
            if doc_count % 100 == 0:
                logging.info(f"Processed {doc_count} files")

    return doc_count


def make_file(file_name="inv_idx.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)


def load_index():
    global inverted_index
    if os.path.exists("inv_idx.json"):
        with open("inv_idx.json", "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        logging.info("Loaded inverted index from file.")
    else:
        logging.info("Inverted index file not found. Running indexing...")
        index_documents()


def index_documents():
    logging.info("Starting indexing...")
    index_time = time.time()
    doc_count = inv_index(ROOT_DIR, max_workers=4)
    save_time = time.time()
    logging.info(f"Indexing time: {save_time - index_time} seconds")
    logging.info("Indexing complete. Saving index to file...")
    make_file()
    end_time = time.time()
    logging.info(f"Save time: {end_time - save_time} seconds")


@app.route('/search', methods=['GET'])
def search_query():
    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "No query parameter provided"}), 400
    SEARCH_TERMS = tokenize(query)
    search_results, query_doc_count = search_terms(SEARCH_TERMS, inverted_index)
    top_urls = calculate_frequency_tfidf(SEARCH_TERMS, inverted_index, query_doc_count)

    result_data = []
    for term, postings in top_urls.items():
        for posting in postings:
            result_data.append({
                "term": term,
                "url": posting['id'],
                "frequency": posting['frq']
            })

    return jsonify(result_data)

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in your templates folder


def search_terms(terms, index):
    results = defaultdict(list)
    total_docs = len([doc for postings in index.values() for doc in postings])  # Count total docs
    for term in terms:
        if term in index:
            postings = index[term]
            results[term] = postings
    return results, total_docs


def start_indexing_in_background():
    indexing_thread = Thread(target=index_documents)
    indexing_thread.daemon = True  # Ensure the thread exits when the main program exits
    indexing_thread.start()


if __name__ == "__main__":
    load_index()  # index is loaded
    start_indexing_in_background()  # indexing in the background
    app.run(debug=False)  # enable threading for Flask
