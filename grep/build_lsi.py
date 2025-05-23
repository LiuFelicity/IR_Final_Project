import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Directory containing the text files
dir_path = os.path.join(os.path.dirname(__file__), 'activity_data_text')
files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

# Read all documents
documents = []
file_names = []
for filename in files:
    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
        text = file.read().lower()
        documents.append(text)
        file_names.append(filename)

# Build term-document matrix (TF-IDF)
# Only include tokens with at least two consecutive letters (no numbers or single letters)
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b', stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply Latent Semantic Indexing (LSI) using SVD
n_components = 100  # You can adjust this value
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_lsi = svd.fit_transform(X)

# Save term vectors
terms = vectorizer.get_feature_names_out()
term_vectors = svd.components_.T  # shape: (n_terms, n_components)

# Save to file
with open(os.path.join(os.path.dirname(__file__), 'term_vectors_lsi.txt'), 'w', encoding='utf-8') as out:
    for idx, term in enumerate(terms):
        vec_str = ' '.join(f'{v:.6f}' for v in term_vectors[idx])
        out.write(f'{term} {vec_str}\n')

print(f"LSI term vectors saved for {len(terms)} terms in 'term_vectors_lsi.txt'.")

# Save document vectors (LSI embeddings)
doc_vectors_path = os.path.join(os.path.dirname(__file__), 'doc_vectors_lsi.txt')
with open(doc_vectors_path, 'w', encoding='utf-8') as out:
    for idx, vec in enumerate(X_lsi):
        vec_str = ' '.join(f'{v:.6f}' for v in vec)
        out.write(f'{file_names[idx]} {vec_str}\n')

print(f"LSI document vectors saved for {len(file_names)} documents in 'doc_vectors_lsi.txt'.")
