import os
import requests
from urllib.parse import urlparse
import time
import random

input_file = "activity_data/opportunity_links.txt"
output_folder = "activity_data"

# 確保資料夾存在
os.makedirs(output_folder, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

for idx, url in enumerate(urls, 1):
    try:
        # 從網址取得最後一段作為檔名
        parsed_url = urlparse(url)
        slug = parsed_url.path.strip("/").split("/")[-1]
        filename = f"{output_folder}/{slug}.html"

        # 若已存在則跳過（可移除此段讓它每次都重新抓）
        if os.path.exists(filename):
            print(f"⏩ 已存在，跳過：{filename}")
            continue

        print(f"[{idx}/{len(urls)}] 抓取 {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"✅ 已儲存：{filename}")
        time.sleep(random.uniform(1, 3))  # 緩衝一下避免被 ban

    except Exception as e:
        print(f"⚠️ 抓取失敗：{url}，錯誤訊息：{e}")