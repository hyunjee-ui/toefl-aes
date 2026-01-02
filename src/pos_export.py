import pandas as pd

# POS 태깅된 CSV 불러오기
df = pd.read_csv("/Users/hyunjee/project/toefl/data/processed/output_pos_tagged.csv")

# content_pos 컬럼만 TXT로 저장
df["content_pos"].to_csv(
    "/Users/hyunjee/project/toefl/data/processed/output_pos_tagged.txt",
    index=False,
    header=False
)

print("✔ 전체 POS 태깅 텍스트가 output_pos_tagged.txt 로 저장되었습니다!")
