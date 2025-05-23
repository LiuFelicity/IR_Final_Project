import numpy as np
import os
import random
import sys
import json

def load_doc_vectors(path):
    doc_vectors = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            fname = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            doc_vectors[fname] = vec
    return doc_vectors

def load_users(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users(users, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(users, f)

def prompt_ratings(activity_names):
    print("Please rate the following activities from 1 (worst) to 5 (best):")
    ratings = []
    for name in activity_names:
        # Display the full text of the activity with title highlighted
        txt_path = os.path.join('activity_data_text', name)
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    title = lines[0].strip()
                    print("=" * max(10, len(title)))
                    print(title)
                    print("=" * max(10, len(title)))
                    for line in lines[1:]:
                        print(line, end='')
                else:
                    print(f"[Empty file: {txt_path}]")
        else:
            print(f"[File not found: {txt_path}]")
        while True:
            try:
                r = int(input(f"\nYour rating for this activity: "))
                if 1 <= r <= 5:
                    ratings.append(r)
                    break
                else:
                    print("Rating must be between 1 and 5.")
            except ValueError:
                print("Please enter an integer between 1 and 5.")
    return np.array(ratings, dtype=float)

def initialize_user_vector(activity_vecs, ratings):
    # Use pseudo-inverse to solve for user vector: ratings = A @ u
    A = np.stack(activity_vecs)
    u = np.linalg.pinv(A) @ ratings
    return u.tolist()

def main():
    users_path = 'users.json'
    doc_vectors = load_doc_vectors('doc_vectors_lsi.txt')
    users = load_users(users_path)
    user_name = input("Enter your name: ").strip()
    if user_name in users:
        print("we do not serve returning user at this point srry")
        sys.exit(0)
    # New user: sample 10 activities
    all_activities = list(doc_vectors.keys())
    sampled = random.sample(all_activities, 10)
    ratings = prompt_ratings(sampled)
    activity_vecs = [doc_vectors[name] for name in sampled]
    user_vec = initialize_user_vector(activity_vecs, ratings)
    users[user_name] = {
        'ratings': {name: float(r) for name, r in zip(sampled, ratings)},
        'vector': user_vec
    }
    save_users(users, users_path)
    print("User vector initialized. we do not serve returning user at this point srry")

if __name__ == '__main__':
    main()
