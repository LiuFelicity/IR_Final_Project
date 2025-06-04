import os
import json
import requests
import time
import random

# File paths
user_scores_file = "grep/train_data/user_scores.jsonl"
activity_data_text_folder = "grep/activity_data_text"
activity_data_folder = "grep/activity_data"
base_url = "https://www.opportunitiescircle.com/"

# Ensure output folder exists
os.makedirs(activity_data_folder, exist_ok=True)

# Load existing activity data filenames
existing_activities = set(os.listdir(activity_data_text_folder))

# Read user scores and extract activity names
with open(user_scores_file, "r", encoding="utf-8") as f:
    activities_to_fetch = set()
    for line in f:
        data = json.loads(line)
        recommendations = data.get("rec", {})
        for activity_name in recommendations.keys():
            if f"{activity_name}.txt" not in existing_activities:
                activities_to_fetch.add(activity_name)

# Fetch and save missing activities
for idx, activity_name in enumerate(activities_to_fetch, 1):
    try:
        url = f"{base_url}{activity_name}"
        filename = os.path.join(activity_data_folder, f"{activity_name}.html")

        # Skip if already fetched
        if os.path.exists(filename):
            print(f"⏩ Already exists, skipping: {filename}")
            continue

        print(f"[{idx}/{len(activities_to_fetch)}] Fetching {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"✅ Saved: {filename}")
        time.sleep(random.uniform(1, 3))  # Avoid being banned

    except Exception as e:
        print(f"⚠️ Failed to fetch: {url}, Error: {e}")
