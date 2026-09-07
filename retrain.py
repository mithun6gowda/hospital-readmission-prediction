import json

import mlflow
import xgboost as xgb
from sklearn.pipeline import Pipeline

import config as cfg
import data_prep as dp


PSI_THRESHOLD = 0.20
DRIFT_SHARE_THRESHOLD = 0.30


def should_retrain(
    prediction_psi,
    drifted_feature_share,
    dataset_drift
):
    psi_trigger = prediction_psi > PSI_THRESHOLD
    feature_drift_trigger = drifted_feature_share > DRIFT_SHARE_THRESHOLD
    dataset_drift_trigger = dataset_drift

    retrain = (
        psi_trigger
        or feature_drift_trigger
        or dataset_drift_trigger
    )

    return {
        "prediction_psi": float(prediction_psi),
        "psi_threshold": PSI_THRESHOLD,
        "psi_trigger": bool(psi_trigger),
        "drifted_feature_share": float(drifted_feature_share),
        "drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "feature_drift_trigger": bool(feature_drift_trigger),
        "dataset_drift": bool(dataset_drift),
        "dataset_drift_trigger": bool(dataset_drift_trigger),
        "retrain": bool(retrain)
    }


def retrain_model():
    (
        X,
        y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = dp.prepare(cfg.DATA_PATH)

    scale_pos_weight = (
        y_train.value_counts()[0]
        / y_train.value_counts()[1]
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", dp.build_preprocessor(X_train)),
        ("classifier", xgb_model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def execute_retraining(
    prediction_psi,
    drifted_feature_share,
    dataset_drift
):
    decision = should_retrain(
        prediction_psi,
        drifted_feature_share,
        dataset_drift
    )

    if not decision["retrain"]:
        decision["retraining_executed"] = False
        return decision

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = retrain_model()

    trusted_types = [
        "numpy.dtype",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]

    with mlflow.start_run(
        run_name="retraining_triggered"
    ) as run:

        mlflow.log_param(
            "trigger_reason",
            "monitoring_drift"
        )

        mlflow.log_param(
            "random_state",
            cfg.RANDOM_STATE
        )

        mlflow.log_param(
            "n_features",
            46
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=trusted_types
        )

        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name="hospital_readmission_xgboost"
    )

    version = registered.version

    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        "hospital_readmission_xgboost",
        "production",
        version
    )

    decision["retraining_executed"] = True
    decision["retrain_run_id"] = run_id
    decision["registered_model"] = (
        "hospital_readmission_xgboost"
    )
    decision["registered_version"] = int(version)
    decision["production_alias"] = "production"

    with open(
        "retraining_decision.json",
        "w"
    ) as f:
        json.dump(
            decision,
            f,
            indent=2
        )

    return decision


if __name__ == "__main__":

    result = execute_retraining(
        prediction_psi=0.25,
        drifted_feature_share=0.087,
        dataset_drift=False
    )

    print(json.dumps(result, indent=2))