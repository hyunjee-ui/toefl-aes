import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Load embeddings and scores
embed_path = "/Users/hyunjee/project/toefl/data/embeddings/pos_embeddings_final.npy"
data_path = "/Users/hyunjee/project/toefl/data/processed/output.csv"

X=np.load(embed_path)
df=pd.read_csv(data_path)

level_map={"low":0, "medium":1, "high":2}
y=df["level"].map(level_map).values


print(f"Embeddings shape: {X.shape}, Scores shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Define regression models to evaluate
models = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

# Train the model
models.fit(X_train, y_train)
# Make predictions
y_pred = models.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Ridge Regression - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# save results
results_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
save_path = "/Users/hyunjee/project/toefl/output/pos_regression_results.csv"
results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"Regression results saved to {save_path}")