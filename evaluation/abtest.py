"""
Use NDCG@5 as evaluation metric for A/B test with 3 rounds per user.
"""

import json
import numpy as np

def dcg(scores):
    """Discounted Cumulative Gain"""
    return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores)])

def ndcg(recommended_scores, ideal_scores):
    """Compute NDCG for a single recommendation list (scores in ranked order)"""
    if sum(ideal_scores) == 0:
        return 0.0
    return dcg(recommended_scores) / dcg(ideal_scores)

def split_rounds(rec_dict):
    """Split a dict of 15 items into 3 lists of 5 (order preserved)."""
    items = list(rec_dict.items())
    return [items[i*5:(i+1)*5] for i in range(3)]

def load_user_rounds(filename):
    """Load user recommendations and split into rounds."""
    user_rounds = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user = obj["user"]
            recs = obj["rec"]
            rounds = split_rounds(recs)
            user_rounds[user] = rounds
    return user_rounds

def evaluate_ndcg_abtest(our_rounds, baseline_rounds):
    """Evaluate NDCG@5 for each round, using top 5 of both models' ratings as IDCG."""
    ndcgs_our = [[] for _ in range(3)]
    ndcgs_baseline = [[] for _ in range(3)]
    for user in our_rounds:
        if user not in baseline_rounds:
            continue
        for i in range(3):
            our_items = our_rounds[user][i]
            base_items = baseline_rounds[user][i]
            # Get scores in order
            our_scores = [score for _, score in our_items]
            base_scores = [score for _, score in base_items]
            # Pool for ideal
            all_scores = our_scores + base_scores
            ideal_scores = sorted(all_scores, reverse=True)[:5]
            ndcgs_our[i].append(ndcg(our_scores, ideal_scores))
            ndcgs_baseline[i].append(ndcg(base_scores, ideal_scores))
    return ndcgs_our, ndcgs_baseline

def main():
    our_file = "method2/user_ratings.jsonl"
    baseline_file = "method2/baseline_user_ratings.jsonl"
    our_rounds = load_user_rounds(our_file)
    baseline_rounds = load_user_rounds(baseline_file)

    ndcgs_our, ndcgs_baseline = evaluate_ndcg_abtest(our_rounds, baseline_rounds)

    print("=== NDCG@5 by Round (higher is better, max=1.0) ===")
    for i in range(3):
        mean_our = np.mean(ndcgs_our[i]) if ndcgs_our[i] else float('nan')
        mean_base = np.mean(ndcgs_baseline[i]) if ndcgs_baseline[i] else float('nan')
        print(f"Round {i+1}:")
        print(f"  Our Model      NDCG@5 = {mean_our:.4f}")
        print(f"  Baseline Model NDCG@5 = {mean_base:.4f}")

    print("\n=== Overall Mean NDCG@5 ===")
    print(f"Our Model:      {np.mean([v for l in ndcgs_our for v in l]):.4f}")
    print(f"Baseline Model: {np.mean([v for l in ndcgs_baseline for v in l]):.4f}")

if __name__ == "__main__":
    main()