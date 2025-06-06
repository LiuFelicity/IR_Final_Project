"""
Use NDCG@5 as evaluation metric, grouped by department
"""

import json
import numpy as np
from collections import defaultdict

def dcg(scores):
    """Discounted Cumulative Gain"""
    return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores)])

def ndcg(recommended_scores, ideal_scores):
    """Compute NDCG for a single recommendation list (scores in ranked order)"""
    if sum(ideal_scores) == 0:
        return 0.0
    return dcg(recommended_scores) / dcg(ideal_scores)

def load_user_departments(filename):
    """Load user -> department mapping from a JSON array file."""
    user2dept = {}
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)  # load the whole array
        for obj in data:
            user2dept[str(obj["id"])] = obj["department"]
    return user2dept

def evaluate_ndcg_by_department(results, model_key, user2dept):
    """Compute mean NDCG@5 for each department for a given model."""
    dept_ndcgs = defaultdict(list)
    for r in results:
        user = str(r["user"])
        dept = user2dept.get(user, "Unknown")
        all_scores = list(r["our_model"].values()) + list(r["baseline_model"].values())
        ideal_scores = sorted(all_scores, reverse=True)[:5]
        scores = list(r[model_key].values())
        dept_ndcgs[dept].append(ndcg(scores, ideal_scores))
    return {dept: np.mean(vals) for dept, vals in dept_ndcgs.items()}

def main():
    filename = "evaluation/gemini_evaluation_results.jsonl"
    dept_file = "grep/train_data/pseudo_user_profiles.json"
    ndcg_file = "evaluation/major_result.txt"
    with open(filename, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    user2dept = load_user_departments(dept_file)

    print("=== NDCG@5 by Department ===")
    with open(ndcg_file, "w", encoding="utf-8") as out:
        for model_key in ["our_model", "baseline_model"]:
            dept_ndcgs = evaluate_ndcg_by_department(results, model_key, user2dept)
            out.write(f"\n{model_key}:\n")
            print(f"\n{model_key}:")
            for dept, ndcg_val in dept_ndcgs.items():
                line = f"NDCG@5 = {ndcg_val:.4f} {dept}\n"
                print(line.strip())
                out.write(line)

if __name__ == "__main__":
    main()