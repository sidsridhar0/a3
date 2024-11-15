import os
import json
import re
from bs4 import BeautifulSoup

from collections import defaultdict

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
            #print(f)
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
                inverted_index[t].append({"id" : doc_id, "frq" : f})
            
            doc_count += 1
    return doc_count

def make_file(file_name="inv_idx.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)


if __name__ == "__main__":
    inv_index(ROOT_DIR)
    make_file()
    print(len(inverted_index))
    print(os.path.getsize("inv_idx.json") / 1024)