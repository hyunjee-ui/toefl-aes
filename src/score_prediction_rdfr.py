import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os

embeddings = np.load("/Users/hyunjee/Desktop/toefl/data/embeddings/embeddings_final.npy")
df = pd.read_csv("/Users/hyunjee/Desktop/toefl/data/processed/output.csv")

y = df["level"]  
X = embeddings

le=LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


os.makedirs("../output/reports", exist_ok=True)
pd.DataFrame({
    "y_true": le.inverse_transform(y_test),
    "y_pred": le.inverse_transform(y_pred)
}).to_csv("../output/reports/prediction_results.csv", index=False)

print("✅ 예측 결과가 /output/reports/prediction_results.csv 에 저장되었습니다.")
