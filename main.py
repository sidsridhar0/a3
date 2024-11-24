import math
import os
import json
import re
from operator import itemgetter
from bs4 import BeautifulSoup
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# implement tf-idf: TF(t,d)×IDF(t)


# For default values in dict
ROOT_DIR = "ANALYST"
inverted_index = defaultdict(list)


def calculate_frequency_tfidf(terms, index, doc_count, top_n=5):
    # To store the top results for each search term
    top_urls = defaultdict(list)
    # make sure all terms are lowercase
    terms = [term.lower() for term in terms]

    idf = {term: math.log(doc_count / (len(index.get(term, [])) + 1)) for term in terms}
    # For each search term, get the documents where it appears
    for term in terms:
        if term in index:
            postings = index[term]

            for posting in postings:
                tf = posting['frq']
                posting['tf-idf'] = tf * idf[term]
            # Sort by frequency (highest first)
            sorted_postings = sorted(postings, key=itemgetter('tf-idf'), reverse=True)
            # Get the top N URLs
            top_urls[term] = sorted_postings[:top_n]

    return top_urls


def tokenize(text):
    # Tokenize based on whitespace, punctuation, and mult-word phrases
    tokens = re.findall(r'\"(.*?)\"|\b\w+\b', text.lower())

    return tokens


def get_html(path):
    with open(path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        return soup.get_text()


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
    for token, postings in local_index.items():
        global_index[token].extend(postings)


def inv_index(root, max_workers=4):
    doc_count = 0
    all_files = []
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
                print(f"Processed {doc_count} files")

    return doc_count


def make_file(file_name="inv_idx.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)


def search_terms(terms, index):
    results = defaultdict(list)
    doc_count = defaultdict(list)

    # Convert all search terms to lowercase
    terms = [term.lower() for term in terms]

    # Search the index for the terms
    for term in terms:
        if term in index:
            postings = index[term]
            results[term] = index[term]
            doc_count[term] = len(postings)

    return results, doc_count


def print_search_results(results):
    if not results:
        print("No results found.")
    else:
        for term, postings in results.items():
            print(f"Results for '{term}':")
            for posting in postings:
                print(f"Document ID: {posting['id']} (Frequency: {posting['frq']})")


if __name__ == "__main__":
    query = input("What would you like to know?\n")
    # tokenize the query
    SEARCH_TERMS = tokenize(query)

    print("Starting indexing...")
    index_time = time.time()
    doc_count = inv_index(ROOT_DIR, max_workers=4)
    save_time = time.time()
    print("Indexing time:", save_time - index_time)
    print("Indexing complete. Saving index to file...")
    make_file()
    end_time = time.time()
    print("Save time:", end_time - save_time)

    # Load the index from file for searching
    with open("inv_idx.json", "r", encoding="utf-8") as f:
        inverted_index = json.load(f)

    # Search for the specified terms in the index
    print("\nStarting search...")
    search_results, query_doc_count = search_terms(SEARCH_TERMS, inverted_index)
    print_search_results(search_results)

    # Final report
    print("\nFinal Report:")
    # Get the top URLs for each search term
    print("\nTop URLs for each query:")
    top_urls = calculate_frequency_tfidf(SEARCH_TERMS, inverted_index, doc_count)

    for term, postings in top_urls.items():
        print(f"\nTop 5 URLs for '{term}':")
        for posting in postings:
            print(f"URL: {posting['id']} (Frequency: {posting['frq']})")
