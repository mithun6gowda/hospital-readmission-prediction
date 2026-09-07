from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "hospital_readmission_xgboost"
    assert data["version"] == "production"


def test_predict_endpoint():
    payload = {
        "time_in_hospital": 5,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 6,
        "age_midpoint": 75,
        "total_prior_visits": 0,
        "medication_change_count": 2,

        "race": "Caucasian",
        "gender": "Female",
        "admission_type_id": "1",
        "discharge_disposition_id": "1",
        "admission_source_id": "7",
        "payer_code": "MC",
        "max_glu_serum": "None",
        "A1Cresult": "None",

        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "insulin": "Up",
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",

        "change": "Ch",
        "diabetesMed": "Yes",

        "diag_1_category": "Circulatory",
        "diag_2_category": "Diabetes",
        "diag_3_category": "Endocrine/Metabolic",
        "medical_specialty_topk": "Unknown"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["readmission_probability"] <= 1.0
    assert data["risk"] in ["low", "high"]
    assert data["model"] == "hospital_readmission_xgboost"
    assert data["version"] == "production"


def test_predict_validation():
    response = client.post(
        "/predict",
        json={
            "time_in_hospital": 5
        }
    )

    assert response.status_code == 422