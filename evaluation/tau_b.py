import os
import json
import sys
from scipy.stats import kendalltau
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from method2.baseline import User, load_user_scores

# Paths
USER_SCORE_PATH = os.path.join(os.path.dirname(__file__), "../grep/train_data/user_scores.jsonl")
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "../grep/train_data/pseudo_user_profiles.json")

# Load user profiles
with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
    user_profiles = {str(profile["id"]): profile for profile in json.load(f)}

# Load user scores
user_scores = load_user_scores(USER_SCORE_PATH)

taus = []
for user_id, item_score_list in user_scores.items():
    if int(user_id) <= 4000:
        continue
    if len(item_score_list) < 2:
        continue  # Need at least 2 items for tau

    profile = user_profiles.get(str(user_id), {})
    department = profile.get("department")
    age = profile.get("age")
    print(f"Evaluating user {user_id} with department {department} and age {age}")
    # user = User(name=user_id, department=department, age=age)
    user = User.load(name=user_id) if User.exists(user_id) else User(name=user_id, age=age, department=department)
    items = [item for item, _ in item_score_list]
    gt_scores = [score for _, score in item_score_list]
    pred_scores = [user.getscore(item) for item in items]

    # Sort by ground truth (descending, 5 is best)
    gt_rank = [x for _, x in sorted(zip(gt_scores, range(len(gt_scores))), reverse=True)]
    # Sort by predicted score (descending)
    pred_rank = [x for _, x in sorted(zip(pred_scores, range(len(pred_scores))), reverse=True)]

    # Compute Kendall's tau
    tau, _ = kendalltau(gt_rank, pred_rank)
    taus.append(tau)

print(f"Average Kendall's tau: {sum(taus)/len(taus):.4f} over {len(taus)} users")