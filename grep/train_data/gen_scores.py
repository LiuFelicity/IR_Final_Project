import os
import sys
import json
import time
import random
import google.generativeai as genai

# 設定 Gemini API 金鑰
genai.configure(api_key="AIzaSyCcWvPylo8gOiiuNu7X9pKdoM7NRpDgPAM")
model = genai.GenerativeModel('gemini-2.0-flash')

# 載入用戶資料
with open(os.path.join('train_data', 'pseudo_user_profiles.json'), 'r', encoding='utf-8') as f:
    user_profiles = json.load(f)

# 讀取活動資料夾中所有 .txt 活動檔案
activity_dir = './activity_data_text'
activity_files = [f for f in os.listdir(activity_dir) if f.endswith('.txt')]
activities = []

for filename in activity_files:
    with open(os.path.join(activity_dir, filename), 'r', encoding='utf-8') as f:
        content = f.read().strip()
        activities.append({
            'title': filename[:-4],  # 去掉 .txt 後綴
            'content': content
        })

def recommend_activities_for_user(user, activities):
    prompt = f"""You are an intelligent recommender system.
A user has the following profile:
- Age: {user['age']}
- Department: {user['department']}
- Interests: {', '.join(user['interests'])}
- Interested regions: {', '.join(user['countries'])}

You are given {len(activities)} activity descriptions. Your task is:
1. Assign each activity a score from 1 (least relevant) to 5 (most relevant) based on how well it matches the user's profile.
2. Return ONLY a dictionary in this format:
   <file name>: <score>
Here are the activities in <file name>: <content> format:\n""" + '\n'.join([
        f"{a['title']}: {a['content']}..." for a in activities
    ])

    try:
        response = model.generate_content(prompt)
        cleaned_text = response.text
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:].strip()
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3].strip()
        result = json.loads(cleaned_text)
        print(result)
        if isinstance(result, dict):
            return result
        else:
            print(f"[!] Unexpected format for user {user['id']}")
            return []
    except Exception as e:
        print(f"[!] Error for user {user['id']}: {e}")
        return []

recommendations = []

for j in range(0, 5000, 20):
    for i in range(j, j+20):
        user = user_profiles[i]
        print(f"({i+1}/{len(user_profiles)}) Recommending for {user['id']}")
        selected_activities = random.sample(activities, 15)
        recs = recommend_activities_for_user(user, selected_activities)
        recommendations.append({
            'user_id': user['id'],
            'recommendations': recs
        })
    time.sleep(60)

with open('train_data/user_scores.json', 'w', encoding='utf-8') as f:
    json.dump(recommendations, f, ensure_ascii=False, indent=4)
