import os
import json
import sys
from scipy.stats import kendalltau, rankdata
import numpy as np

# 加入上層目錄到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from method2.cold_start_model import User, load_user_scores

# Paths
USER_SCORE_PATH = os.path.join(os.path.dirname(__file__), "../grep/train_data/user_scores.jsonl")
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "../grep/train_data/pseudo_user_profiles.json")

# Load user profiles
with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
    user_profiles = {str(profile["id"]): profile for profile in json.load(f)}

# Load user scores
user_scores = load_user_scores(USER_SCORE_PATH)

taus = []
significant_count = 0

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
    # 全部分數一樣也跳過（無法比較排名）
    if len(set(gt_scores)) == 1 or len(set(pred_scores)) == 1:
        continue

    # 使用 rankdata 處理同分，分數高者排名前（加負號）
    gt_rank = rankdata([-s for s in gt_scores], method="average")
    pred_rank = rankdata([-s for s in pred_scores], method="average")

    tau, p_value = kendalltau(gt_rank, pred_rank)
    if tau is not None and not np.isnan(tau):
        print(f"  -> Kendall's tau = {tau:.4f}, p = {p_value:.4g}")
        taus.append(tau)
        if p_value < 0.05:
            significant_count += 1

# 統計與輸出
if taus:
    avg_tau = sum(taus) / len(taus)
    sig_ratio = significant_count / len(taus)
    print(f"\n✅ Average Kendall's tau: {avg_tau:.4f} over {len(taus)} users")
    print(f"🔍 Significant (p < 0.05): {significant_count}/{len(taus)} users ({sig_ratio:.2%})")
else:
    print("No valid users to evaluate.")
