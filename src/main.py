import os
import pandas as pd

folder_path = "/Users/hyunjee/Desktop/4-1/meeting/original"
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

data = []

for file in txt_files:
    file_path = os.path.join(folder_path, file)
    filename = os.path.splitext(file)[0]
    try:
        unique_id, prompt_num, country, level = filename.split("_")
    except ValueError:
        print(f"⚠️ 파일 이름 형식이 다름: {filename}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    data.append({
        "unique_id": unique_id,
        "prompt": prompt_num,
        "country": country,
        "level": level,
        "content": text
    })

df = pd.DataFrame(data)
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
print("✅ DataFrame saved to output.csv")