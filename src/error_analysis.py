import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#  1. 파일 불러오기
input_path = "/Users/hyunjee/project/toefl/output/reports/prediction_results.csv"
df = pd.read_csv(input_path)

save_dir = "/Users/hyunjee/project/toefl/output/reports/error_analysis"
os.makedirs(save_dir, exist_ok=True)

#  2. 잘못 예측된 케이스 추출
df_errors = df[df["y_true"] != df["y_pred"]].copy()

#  3. 오류 유형 정의
def classify_error(row):
    # 레벨 순서를 수치화해 비교
    level_order = {"low": 0, "medium": 1, "high": 2}
    
    true_val = level_order[row["y_true"]]
    pred_val = level_order[row["y_pred"]]

    if pred_val > true_val:
        return "over-prediction"   # 실제보다 높게 예측
    elif pred_val < true_val:
        return "under-prediction"  # 실제보다 낮게 예측
    else:
        return "other"

df_errors["error_type"] = df_errors.apply(classify_error, axis=1)

#  4. 잘못된 예측 저장
df_errors.to_csv(f"{save_dir}/error_cases.csv", index=False, encoding="utf-8-sig")
print(" error_cases.csv 생성 완료")

#  5. 오류 요약 생성
summary = df_errors["error_type"].value_counts().reset_index()
summary.columns = ["error_type", "count"]
summary.to_csv(f"{save_dir}/error_summary.csv", index=False, encoding="utf-8-sig")
print(" error_summary.csv 생성 완료")

# 6. 라벨별 오류 비율 분석
label_summary = df_errors.groupby("y_true").size().reset_index(name="error_count")
label_summary.to_csv(f"{save_dir}/label_error_count.csv", index=False, encoding="utf-8-sig")
print(" label_error_count.csv 생성 완료")

# 7. Confusion Matrix 재계산 및 저장
le = LabelEncoder()
y_true_enc = le.fit_transform(df["y_true"])
y_pred_enc = le.transform(df["y_pred"])

cm = confusion_matrix(y_true_enc, y_pred_enc)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Error Analysis)")

plt.tight_layout()
plt.savefig(f"{save_dir}/confusion_matrix_error.png", dpi=300)
plt.close()

print(" confusion_matrix_error.png 생성 완료")

# 8. 샘플 오류 문장 10개 추출
sample_errors = df_errors.sample(min(10, len(df_errors)))

with open(f"{save_dir}/sample_errors.txt", "w", encoding="utf-8") as f:
    for idx, row in sample_errors.iterrows():
        f.write(f"True: {row['y_true']} | Pred: {row['y_pred']}\n")
        f.write(f"Content: {row.get('content', '(content not included)')}\n")
        f.write("-" * 60 + "\n")

print("sample_errors.txt 생성 완료")

print("\n 에러 분석 전체 완료!")
print(f" 결과 폴더: {save_dir}")
