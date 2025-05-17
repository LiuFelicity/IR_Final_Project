import numpy as np

def load_doc_vectors(path):
    doc_vectors = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            fname = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            doc_vectors[fname] = vec
    return doc_vectors

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_top_k(query_fname, doc_vectors, k=5):
    if query_fname not in doc_vectors:
        print(f"File '{query_fname}' not found in doc_vectors_lsi.txt.")
        return
    query_vec = doc_vectors[query_fname]
    sims = []
    for fname, vec in doc_vectors.items():
        if fname == query_fname:
            continue
        sim = cosine_similarity(query_vec, vec)
        sims.append((sim, fname))
    sims.sort(reverse=True)
    print(f"Top {k} most relevant documents to '{query_fname}':")
    for i in range(min(k, len(sims))):
        print(f"{i+1}. {sims[i][1]} (similarity: {sims[i][0]:.4f})")

def main():
    doc_vectors = load_doc_vectors('doc_vectors_lsi.txt')
    query_fname = input("Enter a filename from activity_data_text (e.g. commonwealth-distance-learning-scholarships.txt): ").strip()
    find_top_k(query_fname, doc_vectors, k=5)

if __name__ == '__main__':
    main()
