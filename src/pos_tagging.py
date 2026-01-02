import pandas as pd

# POS 태깅된 CSV 불러오기
df = pd.read_csv("/Users/hyunjee/project/toefl/data/processed/output_pos_tagged.csv")

def extract_pos_only(tagged_text):
    if pd.isna(tagged_text):
        return ""
    tokens = tagged_text.split()
    pos_tags = [token.split("_")[-1] for token in tokens]
    return " ".join(pos_tags)

# POS 시퀀스만 생성
df["pos_sequence"] = df["content_pos"].apply(extract_pos_only)

# POS만 있는 CSV 저장
save_path = "/Users/hyunjee/project/toefl/data/processed/pos_sequence_only.csv"
df[["pos_sequence"]].to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"✅ POS 시퀀스만 저장 완료: {save_path}")
