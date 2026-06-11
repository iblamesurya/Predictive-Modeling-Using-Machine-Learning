import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def generate_mock_dataset():
    """Generates a synthetic binary classification dataset representing customer churn/retention."""
    print("--- Step 1: Generating Synthetic Dataset ---")
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_redundant=3,
        random_state=42,
        class_sep=1.2,
    )

    feature_names = [f"Feature_{i+1}" for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y

    df.to_csv("predictive_data.csv", index=False)
    print("✓ Success: 'predictive_data.csv' created.")
    return df


def train_and_evaluate_model(df):
    """Splits data, trains a Random Forest model, and runs evaluation metrics."""
    print("\n--- Step 2: Training Machine Learning Model ---")

    # Separate features and target
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

    # Initialize and train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✓ Model training complete using Random Forest.")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Print Text Evaluation Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    return y_test, y_pred, y_proba, model


def generate_evaluation_visualizations(y_test, y_pred, y_proba):
    """Generates and saves Confusion Matrix and ROC Curve plots."""
    print("\n--- Step 3: Generating Performance Visualizations ---")
    os.makedirs("evaluation_plots", exist_ok=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.title("Confusion Matrix Dashboard", fontsize=14, pad=15)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    plt.savefig("evaluation_plots/1_confusion_matrix.png")
    plt.close()
    print("✓ Saved: 'evaluation_plots/1_confusion_matrix.png'")

    # 2. ROC Curve Plot
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Calculate baseline diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="Random Forest Model")

    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=15)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("evaluation_plots/2_roc_curve.png")
    plt.close()
    print("✓ Saved: 'evaluation_plots/2_roc_curve.png'")


if __name__ == "__main__":
    # Run the complete machine learning pipeline
    data = generate_mock_dataset()
    y_test, y_pred, y_proba, model = train_and_evaluate_model(data)
    generate_evaluation_visualizations(y_test, y_pred, y_proba)
    print("\n[Pipeline Complete] Ready for submission!")
