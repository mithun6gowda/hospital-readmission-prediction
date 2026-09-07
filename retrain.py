def should_retrain(
    prediction_psi,
    drifted_feature_share,
    dataset_drift
):
    """
    Trigger retraining when any monitoring signal
    crosses its defined threshold.
    """

    psi_trigger = prediction_psi > 0.20
    feature_drift_trigger = drifted_feature_share > 0.30
    dataset_drift_trigger = dataset_drift

    retrain = (
        psi_trigger
        or feature_drift_trigger
        or dataset_drift_trigger
    )

    return {
        "prediction_psi": prediction_psi,
        "psi_threshold": 0.20,
        "psi_trigger": psi_trigger,
        "drifted_feature_share": drifted_feature_share,
        "drift_share_threshold": 0.30,
        "feature_drift_trigger": feature_drift_trigger,
        "dataset_drift": dataset_drift,
        "dataset_drift_trigger": dataset_drift_trigger,
        "retrain": retrain
    }