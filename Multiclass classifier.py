import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    learning_curve
)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("train.csv")

print("Dataset Loaded Successfully")
print("Total Samples:", len(data))

# =========================
# HANDLE MISSING VALUES
# =========================

data = data.fillna(0)

# =========================
# FEATURES & LABELS
# =========================

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# NORMALIZATION
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# NEURAL NETWORK MODEL
# =========================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,              # Regularization lambda
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=20,
    random_state=42
)

# =========================
# DISPLAY HYPERPARAMETERS
# =========================

print("\n===== HYPERPARAMETERS =====")

print("Hidden Layers:", 2)
print("Neurons:", (128, 64))
print("Activation Function:", "ReLU")
print("Optimizer:", "Adam")
print("Regularization Lambda:", 0.001)
print("Batch Size:", 64)
print("Learning Rate:", 0.001)
print("Epochs:", 20)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining Model...")

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# LEARNING CURVE
# =========================

print("\nGenerating Learning Curve...")

train_sizes, train_scores, validation_scores = learning_curve(
    MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=20,
        random_state=42
    ),
    X_train,
    y_train,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
validation_mean = np.mean(validation_scores, axis=1)

# =========================
# PLOT LEARNING CURVE
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    train_mean,
    marker='o',
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    validation_mean,
    marker='o',
    label='Validation Accuracy'
)

plt.title("MNIST Neural Network Learning Curve")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.show()
