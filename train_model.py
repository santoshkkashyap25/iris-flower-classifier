import os
import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_model():
    """Trains an Iris classification pipeline with StandardScaler and Logistic Regression,

    evaluates it, and saves the trained pipeline artifact.
    """
    print("=" * 50)
    print("Iris Model Training Pipeline")
    print("=" * 50)

    print("\n1. Loading Iris dataset...")
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    x = df.drop("target", axis="columns")
    y = df["target"]

    print("\n2. Splitting data (80% train, 20% test)...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n3. Building & training ML Pipeline (StandardScaler + LogisticRegression)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
    ])
    pipeline.fit(x_train, y_train)

    print("\n4. Evaluating model on test set...")
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {accuracy * 100:.2f}%\n")
    print("   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(os.path.dirname(__file__), 'iris.pkl')
    print(f"\n5. Saving pipeline model to: {model_path}")
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(pipeline, file)
        print("   Model pipeline saved successfully.")
    except Exception as e:
        print(f"   Error saving model: {e}")
        raise

if __name__ == "__main__":
    train_and_save_model()