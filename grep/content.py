import os
from bs4 import BeautifulSoup

input_folder = "activity_data"
output_folder = "activity_data_text"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".html"):
        continue

    filepath = os.path.join(input_folder, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    title = soup.title.string.strip() if soup.title else "No Title"

    # 擷取 Funding/Region/Type 資訊
    info_items = soup.select(".elementor-widget-post-info li")
    extra_info = []
    for li in info_items:
        label = li.select_one(".elementor-post-info__item-prefix")
        value = li.select_one(".elementor-post-info__terms-list")
        if label and value:
            key = label.get_text(strip=True).rstrip(":")
            val = value.get_text(separator=", ", strip=True)
            extra_info.append(f"{key}: {val}")

    content_texts = []

    # 擷取主內容 (包含 theme-post-content widget 的段落)
    main_content = soup.select_one(".elementor-widget-theme-post-content")
    if main_content:
        paragraphs = main_content.find_all(["p", "li"])
        for para in paragraphs:
            text = para.get_text(strip=True)
            if not text:
                continue
            if "Increase your chances of acceptance for the program by using our" in text:
                continue
            if "Join Our OC Community" in text:
                continue
            if "Application Process" in text:
                break
            if "disclaimer" in text.lower():
                break
            if text not in content_texts:
                content_texts.append(text)

    # 擷取其他 text-editor 內容
    text_editors = soup.select(".elementor-widget-text-editor")
    for block in text_editors:
        text = block.get_text(separator="\n", strip=True)
        if not text:
            continue
        if "Increase your chances of acceptance for the program by using our" in text:
            continue
        if "Join Our OC Community" in text:
            continue
        if "Application Process" in text:
            break
        if "disclaimer" in text.lower():
            break
        if text not in content_texts:
            content_texts.append(text)

    # 組合完整內容
    full_text = title + "\n\n"
    if extra_info:
        full_text += " / ".join(extra_info) + "\n\n"
    full_text += "\n\n".join(content_texts)

    # 儲存成 .txt
    txt_filename = filename.replace(".html", ".txt")
    output_path = os.path.join(output_folder, txt_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ 已產出：{output_path}")
