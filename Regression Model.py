import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# LOAD DATASET
# =========================

housing = fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("Dataset Loaded:", len(X), "rows")

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL WITH HYPERPARAMETERS
# =========================

ridge_model = Ridge(
    alpha=10,          # Regularization strength
    fit_intercept=True,
    solver="auto",
    tol=0.0001,
    max_iter=None
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", ridge_model)
])

# =========================
# DISPLAY HYPERPARAMETERS
# =========================

print("\n===== HYPERPARAMETERS =====")

for param, value in ridge_model.get_params().items():
    print(f"{param}: {value}")

# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n===== MODEL PERFORMANCE =====")
print("R² Score:", round(r2, 4))
print("Accuracy (%):", round(r2 * 100, 2))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))

# =========================
# LEARNING CURVE
# =========================

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# =========================
# PLOT LEARNING CURVE
# =========================

plt.figure(figsize=(8,5))

plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Validation Score")

plt.title("Multivariable Regression Learning Curve")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.legend()
plt.grid()

plt.show()
