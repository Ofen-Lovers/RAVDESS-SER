import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Extra layer for Jupyter/VS Code interactive
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import sounddevice as sd
import soundfile as sf
from feature_extraction_selected import extract_features
from model import SVM_modeling, LR_modeling


def load_data():
    df = pd.read_csv("features/extracted_features_selected.csv")
    return df

def feature_scaling(X_train, X_test):
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def svm_grid_search(X_train_scaled, y_train, X_test_scaled, y_test):
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10],                  # Regularization strength
        'gamma': ['scale', 0.01, 0.1],      # Only for RBF kernel
        'kernel': ['linear', 'rbf']         # Linear and RBF kernels
    }

    # Grid search with 5-fold cross-validation
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print("\nSVM Grid Search Results:")
    print("Best parameters:", grid.best_params_)
    print("Best cross-validated accuracy:", grid.best_score_)

    # Evaluate on the test set
    y_pred_best = grid.predict(X_test_scaled)
    
    # Evaluate and print results
    results_best = pd.DataFrame({
        "SVM": evaluate_model(y_test, y_pred_best)
    }).T
    print("Test set evaluation with best parameters:")
    print(results_best)
    return y_pred_best

def lr_grid_search(X_train_scaled, y_train, X_test_scaled, y_test):
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],           # Regularization strength
    'penalty': ['l2'],                       # 'l2' works with lbfgs and multinomial
    'solver': ['lbfgs', 'saga'],             # solvers that support multinomial
    'multi_class': ['multinomial'],          # multi-class strategy
    'class_weight': [None, 'balanced']       # handle class imbalance
    }

    # Grid search with 5-fold cross-validation
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    # Best hyperparameters and cross-validated accuracy
    print("\nLR Grid Search Results:")
    print("Best parameters:", grid.best_params_)
    print("Best cross-validated accuracy:", grid.best_score_)

    # Evaluate on test set
    y_pred_best = grid.predict(X_test_scaled)
    results_best = pd.DataFrame({
        "Logistic Regression": evaluate_model(y_test, y_pred_best)
    }).T
    print("Test set evaluation with best parameters:")
    print(results_best)
    return y_pred_best

def SVM_modeling(X_train_scaled, y_train, X_test_scaled, y_test):
    svm_rbf = SVC(kernel='rbf',  
                C=10,
                gamma=0.01,
                random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred_rbf = svm_rbf.predict(X_test_scaled)
    return svm_rbf, y_pred_rbf

def LR_modeling(X_train_scaled, y_train, X_test_scaled, y_test):
    logreg = LogisticRegression(
        C=1,
        class_weight=None,
        multi_class='multinomial',
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    logreg.fit(X_train_scaled, y_train)  
    y_pred_lr = logreg.predict(X_test_scaled)
    return logreg, y_pred_lr

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-score": f1_score(y_true, y_pred, average='weighted')
    }

def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    cm = confusion_matrix(y_test, y_pred)  
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def feature_importance(model, X_test_scaled, y_test, X_train):
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importance = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)

    plt.figure(figsize=(12,6))
    feature_importance.plot(kind='bar', color='salmon')
    plt.title("Permutation Feature Importance (RBF SVM)")
    plt.ylabel("Mean decrease in accuracy")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def record_audio(filename="my_voice.wav", fs=16000, duration=3):
    fs = 16000  # Sample rate
    duration = 3  # seconds

    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, fs)
    print(f"Saved as {filename}")
    return filename

def predict_emotion(audio_path, model, scaler, model_name="Model"):
    emotion_mapping = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    
    features = extract_features(audio_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred_id = model.predict(features_scaled)[0]
    label = emotion_mapping.get(pred_id, "Unknown")
    print(f"Predicted emotion ({model_name}): {label}")
    return label

def main():
    # Load data
    df = load_data()

    # Split features and labels
    X = df.drop("emotion", axis=1)
    y = df["emotion"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)
    
    # SVM and Logistic Regression hyperparameter tuning
    svm_grid_search(X_train_scaled, y_train, X_test_scaled, y_test)
    lr_grid_search(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # SVM and Logistic Regression Model Training 
    svm_rbf, y_pred_rbf = SVM_modeling(X_train_scaled, y_train, X_test_scaled, y_test)
    logreg, y_pred_lr = LR_modeling(X_train_scaled, y_train, X_test_scaled, y_test)

    # Model comparison
    results_logreg = pd.DataFrame({
    "Logistic Regression": evaluate_model(y_test, y_pred_lr),
    "RBF SVM": evaluate_model(y_test, y_pred_rbf)
    }).T
    print("\nModel Comparison:")
    print(results_logreg)
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred_rbf, model_name="RBF SVM")
    plot_confusion_matrix(y_test, y_pred_lr, model_name="Logistic Regression")

    # Feature Importance
    feature_importance(svm_rbf, X_test_scaled, y_test, X_train)
    feature_importance(logreg, X_test_scaled, y_test, X_train)

    # Record audio and predict emotion
    audio_path = record_audio()

    predict_emotion(audio_path, logreg, scaler, model_name="Logistic Regression")
    predict_emotion(audio_path, svm_rbf, scaler, model_name="RBF SVM")

if __name__ == "__main__":
    main()

    



    