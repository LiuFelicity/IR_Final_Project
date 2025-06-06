"""
Use NDCG@5 as evaluation metric
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

def evaluate_ndcg(results, model_key):
    """Compute mean NDCG@5 for all user-rounds for a given model, using top 5 of all 10 ratings as IDCG"""
    ndcgs = []
    for r in results:
        # Collect all 10 ratings
        all_scores = list(r["our_model"].values()) + list(r["baseline_model"].values())
        ideal_scores = sorted(all_scores, reverse=True)[:5]
        scores = list(r[model_key].values())
        ndcgs.append(ndcg(scores, ideal_scores))
    return np.mean(ndcgs), np.std(ndcgs)

def evaluate_ndcg_by_round(results, model_key):
    """Compute mean NDCG@5 for each round for a given model, using top 5 of all 10 ratings as IDCG"""
    from collections import defaultdict
    round_ndcgs = defaultdict(list)
    for r in results:
        all_scores = list(r["our_model"].values()) + list(r["baseline_model"].values())
        ideal_scores = sorted(all_scores, reverse=True)[:5]
        scores = list(r[model_key].values())
        round_ndcgs[r["round"]].append(ndcg(scores, ideal_scores))
    return {rnd: np.mean(vals) for rnd, vals in round_ndcgs.items()}

def main():
    filename = "gemini_evaluation_results.jsonl"
    with open(filename, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    print("=== NDCG@5 Evaluation (higher is better, max=1.0) ===")
    for model_key in ["our_model", "baseline_model"]:
        mean_ndcg, std_ndcg = evaluate_ndcg(results, model_key)
        print(f"{model_key}: mean NDCG@5 = {mean_ndcg:.4f} (std={std_ndcg:.4f})")

    print("\n=== NDCG@5 by Round ===")
    for model_key in ["our_model", "baseline_model"]:
        round_ndcgs = evaluate_ndcg_by_round(results, model_key)
        print(f"{model_key}:")
        for rnd in sorted(round_ndcgs):
            print(f"  Round {rnd}: NDCG@5 = {round_ndcgs[rnd]:.4f}")

if __name__ == "__main__":
    main()