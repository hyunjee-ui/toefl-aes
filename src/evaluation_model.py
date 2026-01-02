import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 예측 결과 불러오기
folder = "/content/drive/MyDrive/toefl/output/reports/exp003_pos_epoch20/exp001_epoch10/prediction_results.csv"


csv_path = "/content/drive/MyDrive/toefl/output/reports/exp003_pos_epoch20/exp001_epoch10/prediction_results.csv"

print("불러오는 파일:", csv_path)
df = pd.read_csv(csv_path)

save_dir="/content/drive/MyDrive/toefl/output/reports/exp003_pos_epoch20/exp001_epoch10//metrics"
os.makedirs(save_dir, exist_ok=True)

y_true = df["y_true"]
y_pred = df["y_pred"]

label_order = ["low", "medium", "high"]

df["y_true"] = pd.Categorical(df["y_true"], categories=label_order, ordered=True)
df["y_pred"] = pd.Categorical(df["y_pred"], categories=label_order, ordered=True)



# y_true와 y_pred를 원래 레이블로 변환
y_true = df["y_true"]
y_pred = df["y_pred"]

# 정확도 계산
accuracy = accuracy_score(y_true, y_pred)

# 분류 리포트 (성능 지표 계산)
report = classification_report(y_true, y_pred, output_dict=True)
report_df=pd.DataFrame(report).transpose()
report_df.to_csv(f"{save_dir}/classification_report.csv", encoding="utf-8-sig")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=label_order)

# =========================
# 5. QWK Calculation
# =========================
# Convert to numeric labels
label_to_num = {"low": 0, "medium": 1, "high": 2}
y_true_num = df["y_true"].map(label_to_num)
y_pred_num = df["y_pred"].map(label_to_num)

qwk = cohen_kappa_score(y_true_num, y_pred_num, weights='quadratic')

# Save QWK to file
with open(f"{save_dir}/qwk_score.txt", "w") as f:
    f.write(f"QWK Score: {qwk:.4f}\n")

print(f"QWK Score: {qwk:.4f}")

# =========================
# 6. Save Confusion Matrix
# =========================
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_order,
    yticklabels=label_order
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.savefig(f"{save_dir}/confusion_matrix_qwk.png", dpi=300)
plt.close()

# =========================
# 7. Print summary
# =========================
print(f"✔ Accuracy: {accuracy:.4f}")
print(f"✔ Confusion matrix saved to confusion_matrix_qwk.png")
print(f"✔ Classification report saved")
print(f"✔ QWK Score saved to qwk_score.txt")
print("모든 평가 파일 생성 완료!")