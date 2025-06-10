from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"\n=== Evaluasi {model_name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))

    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
