import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    mlflow.set_experiment("assignment-3-mlflow-2")

    with mlflow.start_run(run_name="Training Model"):
        run = mlflow.active_run()
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        train_dataset = pd.read_csv("data/train.csv")
        y = train_dataset["target"].values.astype("float32")
        X = train_dataset.drop("target", axis=1).values

        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
        clf.fit(X, y)

        joblib.dump(clf, "models/model.joblib")

        mlflow.log_artifact("models/model.joblib")

        signature = infer_signature(X, clf.predict(X))
        mlflow.sklearn.log_model(clf, "model", signature=signature)

        mlflow.log_metric("training_score", clf.score(X, y))
