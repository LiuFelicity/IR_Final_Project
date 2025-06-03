import os
import argparse
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
import time
import random
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def sigma(x):
    return 1 / (1 + math.exp(-x))

def load_user_scores(file_path):
    """
    Load user scores from a JSONL file and store them in the format {user_id, item, score}.
    Each user will have up to 15 pairs.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        dict: A dictionary where keys are user IDs and values are lists of (item, score) pairs.
    """
    user_scores = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data["user"]
            recommendations = data["rec"]

            # 將推薦項目和分數轉換為 (item, score) 格式
            for item, score in recommendations.items():
                if len(user_scores[user_id]) < 15:  # 限制每個用戶最多存 15 個推薦項目
                    user_scores[user_id].append((item, score))

    return user_scores

def load_and_copy_item_vectors():
    """
    Dynamically compute LSI vectors for documents in `activity_data_text`.

    Returns:
        tuple: A tuple containing the original and real item vectors, and a mapping from file names (without .txt) to integer indices.
    """
    dir_path = os.path.join(os.path.dirname(__file__), '../grep/activity_data_text')
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
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b', stop_words='english')
    X = vectorizer.fit_transform(documents)

    # Apply Latent Semantic Indexing (LSI) using SVD
    n_components = 100  # You can adjust this value
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsi = svd.fit_transform(X)

    # Create file name to index mapping
    cleaned_file_names = [file_name.strip().lower().replace('.txt', '') for file_name in file_names]
    file_to_index = {file_name: idx for idx, file_name in enumerate(cleaned_file_names)}

    return X_lsi, X_lsi.copy(), file_to_index

def initialize_user_vector(num_users=5000, vector_dim=100):
    """
    Initialize the user vector as a 5000x100 matrix filled with zeros.

    Args:
        num_users (int): Number of users (default: 5000).
        vector_dim (int): Dimension of the user vector (default: 100).

    Returns:
        np.ndarray: A matrix of shape (num_users, vector_dim) filled with zeros.
    """
    return np.zeros((num_users, vector_dim))

def BPR_gradient(rate=0.01, iterations=30, train_num=4000, lam=0.009, user_scores=None, file_to_vector=None, user_vector=None, item_vector=None, item_vector_original=None, lambda_tfidf=0):
    """
    Perform BPR gradient descent using user scores and item vectors.

    Args:
        rate (float): Learning rate.
        iterations (int): Number of iterations.
        train_num (int): Number of training samples per iteration.
        lam (float): Regularization parameter.
        user_scores (dict): User scores in the format {user_id: [(item, score), ...]}.
        file_to_vector (dict): Mapping from item names to vectors.
        user_vector (np.ndarray): User vector matrix.
        item_vector (dict): Mapping from item names to their vectors.
    """
    print("BPR gradient...")

    # 在 BPR_gradient 函數中記錄缺失的項目
    missing_items = []

    for _ in range(iterations):
        if _ % 10 == 0:
            print(f"Iteration {_+1}/{iterations}, BPR loss = {BPR_evluate(train_num, user_scores, lam, user_vector, item_vector)}")
        for user in range(1, train_num+1):
            # 隨機選擇一個 user
            # user = random.choice(list(user_scores.keys()))
            user_items = user_scores[user]

            # 隨機選擇兩個不同的 item，確保分數不同
            max_attempts = 100  # 限制迴圈次數
            attempts = 0
            while attempts < max_attempts:
                item1, score1 = random.choice(user_items)
                item2, score2 = random.choice(user_items)
                if item1 != item2 and score1 != score2:
                    break
                attempts += 1
            else:
                # print(f"Warning: Unable to find two distinct items for user {user} after {max_attempts} attempts. Skipping.")
                continue

            # 每次迴圈輸出進度
            # if user % 100 == 0:
            #     print(f"Processing user {user}/{train_num}...")

            # 確定 label
            if score1 > score2:
                high_item, low_item = item1, item2
                label_high, label_low = 1, -1
            else:
                high_item, low_item = item2, item1
                label_high, label_low = 1, -1
            # print(f"User: {user}, High Item: {high_item}, Low Item: {low_item}, Labels: {label_high}, {label_low}")
            # 獲取對應的向量
            try:
                high_item_idx = file_to_vector[high_item.lower()]
                low_item_idx = file_to_vector[low_item.lower()]

                user_vec = user_vector[user]
                high_item_vec = item_vector[high_item_idx]
                low_item_vec = item_vector[low_item_idx]

                # 計算梯度並更新
                # loss_grad_high = label_high * sigma(-1 * label_high * np.dot(user_vec, high_item_vec))
                # loss_grad_low = label_low * sigma(-1 * label_low * np.dot(user_vec, low_item_vec))

                # user_vector[user] += rate * (loss_grad_high * high_item_vec + loss_grad_low * low_item_vec)
                # item_vector[high_item_idx] += rate * loss_grad_high * user_vec
                # item_vector[low_item_idx] += rate * loss_grad_low * user_vec

                # 真正的
                tmp = 1-sigma(np.dot(user_vec, high_item_vec) - np.dot(user_vec, low_item_vec))
                user_vector[user] += rate * (np.dot(tmp, (high_item_vec - low_item_vec)) - 2* lam * user_vec)
                high_item_vec += rate * (tmp * user_vec - 2 * lam * high_item_vec - 2 * lambda_tfidf * (high_item_vec - item_vector_original[high_item_idx]))
                low_item_vec += rate * (-tmp * user_vec - 2 * lam * low_item_vec  - 2 * lambda_tfidf * (low_item_vec  - item_vector_original[low_item_idx]))

            except KeyError as e:
                # print(f"KeyError: {e}. Skipping this pair.")
                missing_items.append(str(e))
                continue

    print("BPR gradient done.")

    # 在程式結尾輸出缺失的項目
    if missing_items:
        print("Missing items:", set(missing_items))

def BPR_evluate(train_num, user_scores, lam, user_vector, item_vector):
    """
    Evaluate the BPR loss by iterating over users from 1 to train_num and their item pairs.

    Args:
        train_num (int): Number of users to evaluate (from 1 to train_num).
        user_scores (dict): User scores in the format {user_id: [(item, score), ...]}.
        lam (float): Regularization parameter.
        user_vector (np.ndarray): User vector matrix.
        item_vector (np.ndarray): Item vector matrix.

    Returns:
        float: The calculated BPR loss.
    """
    BPRloss = 0

    for user in range(1, train_num + 1):
        if user not in user_scores:
            continue

        user_vec = user_vector[user]
        user_items = user_scores[user]

        # Iterate over all pairs of items for the user
        for i in range(len(user_items)):
            for j in range(len(user_items)):
                if i == j:
                    continue

                item_p, score_p = user_items[i]
                item_n, score_n = user_items[j]

                # Skip pairs with the same score
                if score_p == score_n:
                    continue

                # Ensure item_p has a higher score than item_n
                if score_p < score_n:
                    item_p, item_n = item_n, item_p

                # Skip items not found in file_to_vector
                if item_p.lower() not in file_to_vector or item_n.lower() not in file_to_vector:
                    continue

                item_vec_p = item_vector[file_to_vector[item_p.lower()]]
                item_vec_n = item_vector[file_to_vector[item_n.lower()]]

                # Calculate the BPR loss for the given user and items
                BPRloss += -math.log(sigma(np.dot(user_vec, item_vec_p) - np.dot(user_vec, item_vec_n)))

    # Add regularization term
    for user_vec in user_vector:
        BPRloss += lam * np.dot(user_vec, user_vec)
    for item_vec in item_vector:
        BPRloss += lam * np.dot(item_vec, item_vec)

    return BPRloss

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='IR Final Project')
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()
    model = args.model
    np.random.seed(817)
    
    if model == "original":
    # 預處理
        user_vector = initialize_user_vector()
        user_scores = load_user_scores('../grep/train_data/user_scores.jsonl')
        # print("Loaded user scores:", user_scores)
        item_vector_original, item_vector_real, file_to_vector = load_and_copy_item_vectors()
        # print(file_to_vector)
        # train
        BPR_gradient(
            rate=0.01,
            iterations=100,
            train_num=4000,
            lam=0.009,
            user_scores=user_scores,
            file_to_vector=file_to_vector,
            user_vector=user_vector,
            item_vector=item_vector_real,
            item_vector_original = item_vector_original,
            lambda_tfidf=0 # only for item_cold_start model
        )
        # test
    elif model == "item_cold_start":
        user_vector = initialize_user_vector()
        user_scores = load_user_scores('../grep/train_data/user_scores.jsonl')
        item_vector_original, item_vector_real, file_to_vector = load_and_copy_item_vectors()
        # print(file_to_vector)
        # train
        BPR_gradient(
            rate=0.01,
            iterations=100,
            train_num=4000,
            lam=0.009,
            user_scores=user_scores,
            file_to_vector=file_to_vector,
            user_vector=user_vector,
            item_vector=item_vector_original,
            item_vector_original = item_vector_original,
            lambda_tfidf=0.001 # only for item_cold_start model
        )
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} sec")