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
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gui.constants import DEPARTMENTS_OPTIONS, AGES_OPTIONS

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

def add_user(user_name, vector_dim=100, user_file='user_vectors.pkl'):
    """
    Add and initialize a new user, automatically loading and saving the users dictionary.

    Args:
        user_name (str): The name of the user to add.
        vector_dim (int): Dimension of the user vector.
        user_file (str): File path to load and save user vectors (default: 'user_vectors.pkl').

    Returns:
        User: The newly created User instance.
    """
    user_file = os.path.join(os.path.dirname(__file__), user_file)
    # Load existing users
    try:
        user_vectors, _, _ = load_vectors(user_file=user_file)
    except FileNotFoundError:
        user_vectors = {}

    # Add the user if not already present
    if user_name not in user_vectors:
        user_vectors[user_name] = np.zeros(vector_dim)

    # Save updated users
    with open(user_file, 'wb') as f:
        pickle.dump(user_vectors, f)

    # print(f"User {user_name} added and saved to {user_file}")
    return user_vectors[user_name]

def initialize_users(user_scores, vector_dim=100, train_num=4000):
    """
    Initialize users as instances of the User class.

    Args:
        user_scores (dict): User scores in the format {user_name: [(item, score), ...]}.
        vector_dim (int): Dimension of the user vector.

    Returns:
        dict: A dictionary where keys are user names and values are User instances.
    """
    users = {}
    for user_name in list(user_scores.keys())[:train_num]:
        users[user_name] = User(name=user_name, vector_dim=vector_dim)
    return users

def save_vectors(users, item_vector, file_to_vector, user_file='user_vectors.pkl', item_file='item_vectors.pkl', file_to_vector_file='file_to_vector.pkl'):
    """
    Save user vectors, item vectors, and file_to_vector to files.

    Args:
        users (dict): A dictionary of User instances.
        item_vector (np.ndarray): Item vector matrix.
        file_to_vector (dict): Mapping from file names to vector indices.
        user_file (str): File path to save user vectors (default: 'user_vectors.pkl').
        item_file (str): File path to save item vectors (default: 'item_vectors.pkl').
        file_to_vector_file (str): File path to save file_to_vector (default: 'file_to_vector.pkl').
    """
    user_vectors = {user_name: user.vector for user_name, user in users.items()}
    with open(user_file, 'wb') as f:
        pickle.dump(user_vectors, f)

    with open(item_file, 'wb') as f:
        pickle.dump(item_vector, f)

    with open(file_to_vector_file, 'wb') as f:
        pickle.dump(file_to_vector, f)

    print(f"User vectors saved to {user_file}")
    print(f"Item vectors saved to {item_file}")
    print(f"File-to-vector mapping saved to {file_to_vector_file}")

def load_vectors(user_file='user_vectors.pkl', item_file='item_vectors.pkl', file_to_vector_file='file_to_vector.pkl'):
    """
    Load user vectors, item vectors, and file_to_vector from files.

    Args:
        user_file (str): File path to load user vectors (default: 'user_vectors.pkl').
        item_file (str): File path to load item vectors (default: 'item_vectors.pkl').
        file_to_vector_file (str): File path to load file_to_vector (default: 'file_to_vector.pkl').

    Returns:
        tuple: A tuple containing user vectors (dict), item vectors (np.ndarray), and file_to_vector (dict).
    """
    user_file = os.path.join(os.path.dirname(__file__), user_file)
    item_file = os.path.join(os.path.dirname(__file__), item_file)
    file_to_vector_file = os.path.join(os.path.dirname(__file__), file_to_vector_file)

    with open(user_file, 'rb') as f:
        user_vectors = pickle.load(f)

    with open(item_file, 'rb') as f:
        item_vector = pickle.load(f)

    with open(file_to_vector_file, 'rb') as f:
        file_to_vector = pickle.load(f)

    # print(f"User vectors loaded from {user_file}")
    # print(f"Item vectors loaded from {item_file}")
    # print(f"File-to-vector mapping loaded from {file_to_vector_file}")

    return user_vectors, item_vector, file_to_vector

def BPR_gradient(rate=0.01, iterations=30, train_num=4000, lam=0.009, user_scores=None, file_to_vector=None, users=None, item_vector=None, item_vector_original=None, lambda_tfidf=0, M = None, b = None, lambda_user = 0):
        """
        Perform BPR gradient descent using user scores and item vectors.

        Args:
            rate (float): Learning rate.
            iterations (int): Number of iterations.
            train_num (int): Number of training samples per iteration.
            lam (float): Regularization parameter.
            user_scores (dict): User scores in the format {user_name: [(item, score), ...]}.
            file_to_vector (dict): Mapping from item names to vectors.
            users (dict): A dictionary of User instances.
            item_vector (dict): Mapping from item names to their vectors.
        """
        print("BPR gradient...")

        missing_items = []

        for _ in range(iterations):
            if _ % 10 == 0:
                print(f"Iteration {_+1}/{iterations}, BPR loss = {BPR_evluate(train_num, user_scores, lam, users, item_vector)}")
            for user_name in list(user_scores.keys())[:train_num]:
                user = users[user_name]
                user_items = user_scores[user_name]

                max_attempts = 100
                attempts = 0
                while attempts < max_attempts:
                    item1, score1 = random.choice(user_items)
                    item2, score2 = random.choice(user_items)
                    if item1 != item2 and score1 != score2:
                        break
                    attempts += 1
                else:
                    continue

                if score1 > score2:
                    high_item, low_item = item1, item2
                    label_high, label_low = 1, -1
                else:
                    high_item, low_item = item2, item1
                    label_high, label_low = 1, -1

                try:
                    high_item_idx = file_to_vector[high_item.lower()]
                    low_item_idx = file_to_vector[low_item.lower()]

                    high_item_vec = item_vector[high_item_idx]
                    low_item_vec = item_vector[low_item_idx]

                    tmp = 1 - sigma(np.dot(user.vector, high_item_vec) - np.dot(user.vector, low_item_vec))
                    user.vector += rate * (np.dot(tmp, (high_item_vec - low_item_vec)) - 2 * lam * user.vector - 2 * lambda_user * ( user.vector - (M @ user.onehot.transpose() + b)))
                    high_item_vec += rate * (tmp * user.vector - 2 * lam * high_item_vec - 2 * lambda_tfidf * (high_item_vec - item_vector_original[high_item_idx]))
                    low_item_vec += rate * (-tmp * user.vector - 2 * lam * low_item_vec - 2 * lambda_tfidf * (low_item_vec - item_vector_original[low_item_idx]))
                    b -= rate * ( 2*lambda_user * (M @ user.onehot.transpose() + b - user.vector) + 2 * lam * b )

                except KeyError as e:
                    missing_items.append(str(e))
                    continue
        save_vectors(users, item_vector, file_to_vector)
        print("BPR gradient done.")

        if missing_items:
            print("Missing items:", set(missing_items))

def BPR_evluate(train_num, user_scores, lam, users, item_vector):
    """
    Evaluate the BPR loss by iterating over users from 1 to train_num and their item pairs.

    Args:
        train_num (int): Number of users to evaluate (from 1 to train_num).
        user_scores (dict): User scores in the format {user_name: [(item, score), ...]}.
        lam (float): Regularization parameter.
        users (dict): A dictionary of User instances.
        item_vector (np.ndarray): Item vector matrix.

    Returns:
        float: The calculated BPR loss.
    """
    BPRloss = 0

    for user_name in list(user_scores.keys())[:train_num]:
        user = users[user_name]
        user_vec = user.vector  # Access the vector attribute of the User object
        user_items = user_scores[user_name]

        for i in range(len(user_items)):
            for j in range(len(user_items)):
                if i == j:
                    continue

                item_p, score_p = user_items[i]
                item_n, score_n = user_items[j]

                if score_p == score_n:
                    continue

                if score_p < score_n:
                    item_p, item_n = item_n, item_p

                if item_p.lower() not in file_to_vector or item_n.lower() not in file_to_vector:
                    continue

                item_vec_p = item_vector[file_to_vector[item_p.lower()]]
                item_vec_n = item_vector[file_to_vector[item_n.lower()]]

                BPRloss += -math.log(sigma(np.dot(user_vec, item_vec_p) - np.dot(user_vec, item_vec_n)))

    for user in users.values():
        BPRloss += lam * np.dot(user.vector, user.vector)
    for item_vec in item_vector:
        BPRloss += lam * np.dot(item_vec, item_vec)

    return BPRloss

class User:
    """
    A class to represent a user with a name and vector.

    Attributes:
        name (str): The name of the user.
        vector (np.ndarray): The vector representing the user.
    """
    def __init__(self, name, vector_dim=100, department=None, age=None):
        self.name = name
        self.vector = np.zeros(vector_dim)
        self.onehot = np.zeros(len(DEPARTMENTS_OPTIONS) + len(AGES_OPTIONS))
        if department in DEPARTMENTS_OPTIONS:
            self.onehot[DEPARTMENTS_OPTIONS.index(department)] = 1
        if age in AGES_OPTIONS:
            self.onehot[len(DEPARTMENTS_OPTIONS) + AGES_OPTIONS.index(age)] = 1
        add_user(name, vector_dim=vector_dim)

    def recommend(self, user_name, user_file='user_vectors.pkl', item_file='item_vectors.pkl', top_k=5):
        """
        Recommend items for a given user based on inner product. Reload vectors before recommending.

        Args:
            user_name (str): Name of the user to recommend items for.
            user_file (str): File path to load user vectors (default: 'user_vectors.pkl').
            item_file (str): File path to load item vectors (default: 'item_vectors.pkl').
            top_k (int): Number of top recommendations to return (default: 10).

        Returns:
            list: A list of top_k recommended item indices.
        """
        user_file = os.path.join(os.path.dirname(__file__), user_file)
        item_file = os.path.join(os.path.dirname(__file__), item_file)

        user_name = self.name
        # Reload vectors
        user_vectors, item_vector, file_to_vector = load_vectors(user_file, item_file)
        
        if user_name not in user_vectors:
            print(f"User {user_name} not found.")
            return []

        user_vec = user_vectors[user_name]
        scores = np.dot(item_vector, user_vec)  # 計算內積
        # load user interaction, and remove items that the user has already interacted with by setting their scores to -inf
        if os.path.exists('user_ratings.jsonl'):
            user_scores = load_user_scores('user_ratings.jsonl')
            interacted_items = {item.lower() for item, _ in user_scores.get(user_name, [])}
            for item in interacted_items:
                if item.lower() in file_to_vector:
                    scores[file_to_vector[item.lower()]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:top_k]  # 按分數排序並取前 top_k
        index_to_file = {v: k for k, v in file_to_vector.items()}  # 反轉字典
        top_items = [index_to_file[idx] for idx in top_indices]

        return top_items
    
    def update_user_scores(self, item_scores, vector_dim=100, user_file='user_vectors.pkl', item_file='item_vectors.pkl', file_to_vector_file='file_to_vector.pkl', user_rating_file='user_ratings.jsonl'):
        """
        Update the user vector based on multiple scores for specific items and save the updated ratings to a JSONL file.

        Args:
            user_name (str): Name of the user to update.
            item_scores (dict): Dictionary of item names and their corresponding scores.
            vector_dim (int): Dimension of the user vector.
            user_file (str): File path to load user vectors (default: 'user_vectors.pkl').
            item_file (str): File path to load item vectors (default: 'item_vectors.pkl').
            file_to_vector_file (str): File path to load file-to-vector mapping (default: 'file_to_vector.pkl').
            user_rating_file (str): File path to save user ratings (default: 'user_ratings.jsonl').

        Returns:
            None
        """
        user_name = self.name
        user_file = os.path.join(os.path.dirname(__file__), user_file)
        item_file = os.path.join(os.path.dirname(__file__), item_file)
        file_to_vector_file = os.path.join(os.path.dirname(__file__), file_to_vector_file)
        user_rating_file = os.path.join(os.path.dirname(__file__), user_rating_file)

        # Reload vectors
        user_vectors, item_vector, file_to_vector = load_vectors(user_file, item_file, file_to_vector_file)

        if user_name not in user_vectors:
            print(f"User {user_name} not found.")
            return

        user_vec = user_vectors[user_name]

        # Load existing ratings if the file exists
        existing_ratings = {}
        if os.path.exists(user_rating_file):
            with open(user_rating_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    existing_ratings[data['user']] = data['rec']

        # Update user vector and ratings
        updated_ratings = existing_ratings.get(user_name, {})
        for item_name, score in item_scores.items():
            if item_name.lower() not in file_to_vector:
                print(f"Item {item_name} not found. Skipping.")
                continue
            
        # Save updated user vectors
        with open(user_file, 'wb') as f:
            pickle.dump(user_vectors, f)

        # Save updated ratings to JSONL file
        existing_ratings[user_name] = updated_ratings
        with open(user_rating_file, 'w', encoding='utf-8') as f:
            for user, rec in existing_ratings.items():
                json.dump({'user': user, 'rec': rec}, f)
                f.write('\n')

        print(f"User {user_name}'s vector and ratings updated with items and scores: {item_scores}.")


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='IR Final Project')
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()
    model = args.model
    np.random.seed(817)

    M = np.zeros((100, len(DEPARTMENTS_OPTIONS) + len(AGES_OPTIONS)))
    b = np.zeros(100)
    if model == "original":
        # 預處理
        user_scores = load_user_scores(os.path.join(os.path.dirname(__file__), '../grep/train_data/user_scores.jsonl'))
        users = initialize_users(user_scores, train_num=4000)  # 使用 User class 初始化
        item_vector_original, item_vector_real, file_to_vector = load_and_copy_item_vectors()

        # 訓練
        BPR_gradient(
            rate=0.01,
            iterations=100,
            train_num=4000,
            lam=0.009,
            user_scores=user_scores,
            file_to_vector=file_to_vector,
            users=users,
            item_vector=item_vector_real,
            item_vector_original=item_vector_original,
            lambda_tfidf=0,  # 只針對 item_cold_start 模型
            M = M,
            b = b
        )

    elif model == "item_cold_start":
        user_scores = load_user_scores(os.path.join(os.path.dirname(__file__), '../grep/train_data/user_scores.jsonl'))
        users = initialize_users(user_scores, train_num=4000)  # 使用 User class 初始化
        item_vector_original, item_vector_real, file_to_vector = load_and_copy_item_vectors()

        # 訓練
        BPR_gradient(
            rate=0.01,
            iterations=100,
            train_num=4000,
            lam=0.009,
            user_scores=user_scores,
            file_to_vector=file_to_vector,
            users=users,
            item_vector=item_vector_original,
            item_vector_original=item_vector_original,
            lambda_tfidf=0.001,  # 只針對 item_cold_start 模型
            M = M,
            b = b
        )
    elif model == "recommandation":
        #recommand for 1
        add_user(user_name='Felicity')
        top_k_recommendations = User.recommend(
            user_name="Felicity",
            top_k=5
        )
        print(f"Top 5 recommendations for {"Felicity"}: {top_k_recommendations}")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} sec")