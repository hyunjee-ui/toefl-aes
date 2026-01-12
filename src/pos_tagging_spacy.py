import pandas as pd
import spacy
from tqdm import tqdm
# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
# 데이터 로드 
df = pd.read_csv("/Users/hyunjee/project/toefl/data/processed/output.csv")
texts=df["content"].fillna("")

# POS 태깅 함수
def pos_tag_text(text):
    doc = nlp(text)
    tagged_tokens = [f"{token.text}_{token.tag_}" for token in doc]
    return " ".join(tagged_tokens)

def extract_pos_sequence(tagged_text):
    tokens=tagged_text.split()
    pos_tags=[token.split("_")[-1] for token in tokens]
    return " ".join(pos_tags)

# POS 태깅 적용
tqdm.pandas(desc="POS Tagging")
df["content_pos"] = texts.progress_apply(pos_tag_text)
df["pos_sequence"] = df["content_pos"].apply(extract_pos_sequence)

# (1) word_POS 형태
pos_tagged_path = "/Users/hyunjee/project/toefl/data/processed/output_pos_tagged.csv"
df.to_csv(pos_tagged_path, index=False, encoding="utf-8-sig")

# (2) POS-only 시퀀스
pos_only_path = "/Users/hyunjee/project/toefl/data/processed/pos_sequence_only.csv"
df[["pos_sequence"]].to_csv(pos_only_path, index=False, encoding="utf-8-sig")

print(" spaCy 기반 POS 태깅 완료")
print(f"✔ word_POS 저장: {pos_tagged_path}")
print(f"✔ POS-only 저장: {pos_only_path}")