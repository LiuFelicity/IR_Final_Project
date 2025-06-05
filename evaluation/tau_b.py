import os
import json
import sys
from scipy.stats import kendalltau, rankdata
import numpy as np
# 加入上層目錄到 sys.path
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

    user = User.load(name=user_id) if User.exists(user_id) else User(name=user_id, age=age, department=department)

    items = [item for item, _ in item_score_list]
    gt_scores = [score for _, score in item_score_list]
    pred_scores = [user.getscore(item) for item in items]

    # 過濾掉 None 或 nan
    if any(s is None or (isinstance(s, float) and np.isnan(s)) for s in pred_scores):
        continue
    # 全部分數一樣也跳過
    if len(set(gt_scores)) == 1 or len(set(pred_scores)) == 1:
        continue

    # 使用 rankdata 正確處理同分，分數越高排名越前，所以加負號
    gt_rank = rankdata([-s for s in gt_scores], method="average")
    pred_rank = rankdata([-s for s in pred_scores], method="average")

    tau, _ = kendalltau(gt_rank, pred_rank)
    if tau is not None and not np.isnan(tau):
        taus.append(tau)

# 平均 tau
if taus:
    print(f"Average Kendall's tau: {sum(taus)/len(taus):.4f} over {len(taus)} users")
else:
    print("No valid users to evaluate.")
