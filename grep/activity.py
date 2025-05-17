from bs4 import BeautifulSoup
import os

input_folder = "page_data"
output_folder = "activity_data"
output_links = []

# 確保 output 資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 處理 page_1.html 到 page_25.html
for page_num in range(1, 26):
    filename = f"{input_folder}/page_{page_num}.html"
    if not os.path.exists(filename):
        print(f"❌ 找不到檔案：{filename}")
        continue

    with open(filename, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        h3_tags = soup.find_all("h3", class_="elementor-heading-title elementor-size-default")
        for h3 in h3_tags:
            a_tag = h3.find("a")
            if a_tag and a_tag.get("href"):
                output_links.append(a_tag["href"])

# 去除重複連結
output_links = list(set(output_links))
output_links.sort() # for reproducibility

# 儲存到 activity_data 資料夾下
output_file = f"{output_folder}/opportunity_links.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for link in output_links:
        f.write(link + "\n")

print(f"✅ 已儲存 {len(output_links)} 筆連結到 {output_file}")