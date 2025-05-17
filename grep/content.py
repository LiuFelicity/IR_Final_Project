import os
from bs4 import BeautifulSoup

input_folder = "activity_data"
output_folder = "activity_data_text"

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".html"):
        continue

    filepath = os.path.join(input_folder, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    title = soup.title.string.strip() if soup.title else "No Title"

    content_blocks = soup.select(".elementor-widget-text-editor")

    content_texts = []
    for block in content_blocks:
        text = block.get_text(separator="\n", strip=True)
        if not text: continue
        if "Increase your chances of acceptance for the program by using our" in text: continue
        if "Join Our OC Community" in text: continue
        if text in content_texts: continue
        if "disclaimer" in text.lower(): break
        content_texts.append(text)

    full_text = f"Title: {title}\n\n" + "\n\n".join(content_texts)

    # 儲存成 .txt
    txt_filename = filename.replace(".html", ".txt")
    output_path = os.path.join(output_folder, txt_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ 已產出：{output_path}")