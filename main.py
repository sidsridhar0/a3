import os
import json
import re
from bs4 import BeautifulSoup

from collections import defaultdict

STOP_WORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", 
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", 
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", 
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", 
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", 
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", 
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves", "a", "b", "c", "d", "e", "f", "g", "h", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]


# may need to use file system instead of dict if memory error
ROOT_DIR = "ANALYST"

inverted_index = defaultdict(list)

def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
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

def inv_index(root):
    doc_count = 0
    for s, _, fs in os.walk(root):
        for file in fs:

            if doc_count % 100 == 0:
                print(doc_count)

            data = None
            doc_path = os.path.join(s, file)
            doc_id = os.path.relpath(doc_path, root)
            with open (doc_path, "r") as f:
                data = json.load(f)
            #print(data)
            #print(data["content"])
            #print(data["url"])
            doc_text = get_html(doc_path)
            #print(doc_text)
            tokens = tokenize(doc_text)
            #print(tokens)
            tf = get_token_freq(tokens)
            #print(tf)
            for t, f in tf.items():
                if t in STOP_WORDS:  # Skip stop words
                    continue
                inverted_index[t].append({"id" : doc_id, "frq" : f})
            
            doc_count += 1
    return doc_count

def make_file(file_name="inv_idx.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)


if __name__ == "__main__":
    doc_count = inv_index(ROOT_DIR)
    print("making file")
    make_file()
    
    #final report
    print("number of unique tokens: ", len(inverted_index))
    print("Total size of index on disk: ", os.path.getsize("inv_idx.json") / 1024)
    print("Document Count: ", doc_count)