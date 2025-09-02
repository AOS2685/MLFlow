import mlflow
import mlflow.sklearn
# Wine datasets
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import dagshub
dagshub.init(repo_owner='AOS2685', repo_name='MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/AOS2685/MLFlow.mlflow")


# Load Wine Dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.10, random_state=42)

# Define the Params for RF model
max_depth = 10
n_estimators = 5

# Mention your experiment below
mlflow.set_experiment('MLOPS-Exp2')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state= 42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    # Createa Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', xticklabels = wine.target_names, yticklabels=wine.target_names)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save plot
    plt.savefig("Confusion-matrix.png")

    # Log Artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # Save trained model locally
    joblib.dump(rf, "rf_model.pkl")

    # Log model as artifact (since DagsHub doesn’t support mlflow.sklearn.log_model fully)
    mlflow.log_artifact("rf_model.pkl")


    # Tags
    mlflow.set_tags({"Author": 'Aman', "Project": "Wine Classification"})
    # Log the model
    # mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    print(accuracy)