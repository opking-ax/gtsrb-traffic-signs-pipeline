import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


def evaluate_model(model, test_data, class_names):
    """
    Evaluates the models and saves a classification report and confusion matrix to its directory

    Args:
        model:
        test_data:
        class_names:
    """
    X_test, y_test = test_data

    probs = model.predict(X_test)
    predictions = np.argmax(probs, axis=1)

    report = classification_report(
        y_test, predictions, target_names=class_names, output_dict=True
    )

    cm = confusion_matrix(y_test, predictions)

    return cm, report


def plot_confusion_matrix(cm, class_names, save_path="../reports"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm, annot=False, cmap="cyan", xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()


def save_classification_report(report, save_path="../reports"):
    with open(f"{save_path}/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
