"""
1. Use Gemini agent as an user. First, enter his/her name, department, and age. 
   Use the last 1000 profiles from /grep/train_data/psedo_user_profiles. Use user_id as username.
2. Recommend the user 5 activities using both models.
   The baseline model is in /method2/baseline.py, and our model is in /method2/orginal.py
3. Ask the user to rate each activity from score 1 to 5. 
   Note that in the program, you should pass the content of the recommended activity to gemini as well.
   For example, if the recommended activity is <act>, then the content is in /grep/activity_data_text/<act>.txt
4. Store the ratings into a file for future evaluation. The format should be <user>: {<activity>:<score>}
5. Repeat this for 1000 times
"""


import os
import json
import sys
import time
from dotenv import load_dotenv
# Add the parent directory to the Python path to ensure the method2 module can be imported.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from method2.orginal import User
from method2.baseline_eval import BaseUser
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

ACTIVITY_DATA_PATH = os.path.join(os.path.dirname(__file__), "../grep/activity_data_text")
USER_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "../grep/train_data/pseudo_user_profiles.json")

def get_activity_content(filename):
    file_path = os.path.join(ACTIVITY_DATA_PATH, f"{filename}.txt")
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def ask_gemini_to_rate(user_profile, activities_with_content):
    prompt = (
        f"You are a user with the following profile:\n"
        f"User ID: {user_profile['id']}\n"
        f"Department: {user_profile.get('department', 'N/A')}\n"
        f"Age: {user_profile.get('age', 'N/A')}\n"
        f"Interests: {', '.join(user_profile.get('interests', []))}\n"
        f"Please rate the following activities from 1 (worst) to 5 (best) based on your interest. "
        f"Return your ratings as a comma-separated list of integers in order, e.g., 4,2,5,3,1\n\n"
    )
    for idx, (activity, content) in enumerate(activities_with_content, 1):
        prompt += f"{idx}. {activity}\nDescription: {content[:300]}...\n\n"

    response = model.generate_content(prompt)
    ratings = []
    for token in response.text.strip().split(','):
        try:
            ratings.append(int(token.strip()))
        except Exception:
            import random
            ratings.append(random.randint(1, 5))
            print("gemini output format is wrong")
    return ratings

def evaluate_model(user_profile, user_instance):
    user_id = user_profile.get("id")
    user_name=str(user_id)

    # 2. Get recommendations (like in app.py)
    recommended_items = user_instance.recommend(user_name=user_name, top_k=5)
    activities_with_content = [(item, get_activity_content(item)) for item in recommended_items]

    # 3. Ask Gemini to rate
    ratings = ask_gemini_to_rate(user_profile, activities_with_content)
    item_ratings = {item: rating for item, rating in zip(recommended_items, ratings)}

    # 4. Update user scores (like in app.py)
    user_instance.update_user_scores(item_ratings)

    return item_ratings

def main():
    with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
        user_profiles = json.load(f)

    # Open the output file in append mode
    with open("gemini_evaluation_results.jsonl", "a", encoding="utf-8") as f:
        for user_profile in user_profiles[-432:]:
            user_id = user_profile.get("id")
            user_name=str(user_id)
            department = user_profile.get("department", "wrong department")
            age = str(user_profile.get("age", "wrong age"))
            print(f"Evaluating for user: {user_id}")
            # to avoid exceeding API quota
            if user_id % 2:
                time.sleep(60)

            for round in range(3):
                # 1. Create or load user
                user_instance_original = User(name=user_name)
                user_instance_baseline = BaseUser(name=user_name)

                # new user that has never registered
                if user_instance_original.department is None:
                    user_instance_original.__init__(name=user_name, age=age, department=department) 
                if user_instance_baseline.department is None:
                    user_instance_baseline.__init__(name=user_name, age=age, department=department)
                
                # Evaluate both models
                our_model_result = evaluate_model(user_profile, user_instance_original)
                baseline_result = evaluate_model(user_profile, user_instance_baseline)

                result = {
                    "user": user_id,
                    "round": round+1,
                    "our_model": our_model_result,
                    "baseline_model": baseline_result
                }

                # Write one user result per line
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                # re-train the model based on the feedback (active learning)
                cleaned_ratings = our_model_result
                user_instance_original.update_user_scores(cleaned_ratings)
                cleaned_ratings = baseline_result
                user_instance_baseline.update_user_scores(cleaned_ratings)


if __name__ == "__main__":
    main()