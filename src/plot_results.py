import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 예측 결과 불러오기
df = pd.read_csv("/Users/hyunjee/project/toefl/output/reports/prediction_results.csv")

# 라벨 순서 정의
label_order = ["low", "medium", "high"]

# category 타입으로 변환 (중요)
df["y_true"] = pd.Categorical(df["y_true"], categories=label_order, ordered=True)
df["y_pred"] = pd.Categorical(df["y_pred"], categories=label_order, ordered=True)

# 혼동 행렬 계산  (★ 여기를 df[y_true]로 하면 절대 안 됨!)
cm = confusion_matrix(df["y_true"], df["y_pred"], labels=label_order)

# 시각화
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",   # 오타 수정
    xticklabels=label_order,
    yticklabels=label_order
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# 저장
plt.savefig("/Users/hyunjee/project/toefl/output/reports/confusion_matrix2.png", dpi=300)

plt.show()
