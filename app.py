import os
import logging
import math
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

def load_text_files(folder_path):
    data = {}
    doc_id_to_filename = {}
    doc_id = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                data[doc_id] = file.read()
                doc_id_to_filename[doc_id] = filename
                logging.info(f"Loaded file: {filename} with doc_id: {doc_id}")
                doc_id += 1
    return data, doc_id_to_filename

def tokenize(text):
    return text.lower().split()

def term_frequency(term, document):
    return document.count(term) / len(document)

def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))

def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 * norm_vec2 != 0 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    folder_path = "static"  # Path to the folder with your text files

    # Load and process documents
    docs, doc_id_to_filename = load_text_files(folder_path)
    tokenized_docs = [tokenize(doc) for doc in docs.values()]
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))

    # Compute TF-IDF vectors for documents
    doc_tfidf_vectors = [compute_tfidf(doc, tokenized_docs, vocab) for doc in tokenized_docs]

    # Process the query
    tokenized_query = tokenize(query)
    query_tfidf_vector = compute_tfidf(tokenized_query, tokenized_docs, vocab)

    # Compute cosine similarity
    similarities = []
    for doc_id, doc_vector in enumerate(doc_tfidf_vectors):
        similarity = cosine_similarity(query_tfidf_vector, doc_vector)
        similarities.append((doc_id, similarity))

    # Sort by similarity and get the top 5 most similar documents
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_similar = similarities[:5]  # Get the top 5

    results = [{"document": doc_id_to_filename[doc_id], "similarity": round(similarity, 4)} for doc_id, similarity in top_5_similar]

    # Pass results to the results.html template
    return render_template('result.html', query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)