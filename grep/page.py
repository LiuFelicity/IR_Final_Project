import requests
import time
import os

base_url = "https://www.opportunitiescircle.com/explore-opportunities/?sf_paged="

# 範圍：第 1 到第 25 頁
for page_num in range(1, 26):
    url = f"{base_url}{page_num}"
    try:
        print(f"正在抓取第 {page_num} 頁：{url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 若不是 200 OK 會丟出例外

        # 儲存 HTML
        filename = f"page_data/page_{page_num}.html"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"✅ 已儲存：{filename}")
        time.sleep(1.5)  # 避免對網站造成過大壓力

    except requests.RequestException as e:
        print(f"⚠️ 第 {page_num} 頁抓取失敗：{e}")
