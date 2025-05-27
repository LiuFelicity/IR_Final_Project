import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

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

# Save terms and term vectors to a single .npz file
term_data_path = os.path.join(os.path.dirname(__file__), 'term_data_lsi.npz')
np.savez(term_data_path, terms=terms, vectors=term_vectors)
print(f"LSI terms and vectors saved for {len(terms)} terms in '{os.path.basename(term_data_path)}'.")

# Save document file names and vectors to a single .npz file
doc_data_path = os.path.join(os.path.dirname(__file__), 'doc_data_lsi.npz')
np.savez(doc_data_path, file_names=np.array(file_names), vectors=X_lsi)
print(f"LSI document file names and vectors saved for {len(file_names)} documents in '{os.path.basename(doc_data_path)}'.")
