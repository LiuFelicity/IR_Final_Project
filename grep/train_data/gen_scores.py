import os
import sys
import json
import time
import random
import google.generativeai as genai

# 設定 Gemini API 金鑰
genai.configure(api_key="")
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

def gen_scores():
    output_path = 'train_data/user_scores.jsonl'  # Use .jsonl for JSON Lines format
    for j in range(2411, 5000, 20):
        for i in range(j, j+20):
            user = user_profiles[i]
            print(f"({i+1}/{len(user_profiles)}) Recommending for {user['id']}")
            selected_activities = random.sample(activities, 15)
            recs = recommend_activities_for_user(user, selected_activities)
            record = {
                'user': user['id'],
                'rec': recs
            }
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        time.sleep(60)

def fix_missing_scores():
    input_path = 'train_data/user_scores.jsonl'
    output_path = 'train_data/user_scores_fixed.jsonl'

    def get_user_by_id(user_profiles, user_id):
        for user in user_profiles:
            if user['id'] == user_id:
                return user
        return None

    num_of_requests = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                obj = json.loads(line)
                if obj.get('rec', []) == []:
                    user_id = obj['user']
                    user = get_user_by_id(user_profiles, user_id)
                    if user is not None:
                        print(f"Regenerating recommendation for user {user_id}")
                        selected_activities = random.sample(activities, 15)
                        recs = recommend_activities_for_user(user, selected_activities)
                        num_of_requests += 1
                        obj['rec'] = recs
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")
                fout.write(line)  # Write the original line if error
            if num_of_requests % 20 == 0:
                time.sleep(60) # avoid rate limit


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fix':
        fix_missing_scores()
    else:
        gen_scores()    