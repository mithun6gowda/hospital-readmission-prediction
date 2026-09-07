from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import mlflow


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="API for predicting 30-day hospital readmission risk.",
    version="1.0.0"
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MODEL_PATH = "mlruns/1/models/m-d91399435d7443b1ab3a508644f43793/artifacts"
model = mlflow.sklearn.load_model(MODEL_PATH)


# ============================================================
# PREDICTION INPUT SCHEMA
# ============================================================

class PatientData(BaseModel):

    model_config = ConfigDict(
        populate_by_name=True
    )

    # --------------------------------------------------------
    # Numeric features — 11
    # --------------------------------------------------------

    time_in_hospital: float
    num_lab_procedures: float
    num_procedures: float
    num_medications: float

    number_outpatient: float
    number_emergency: float
    number_inpatient: float
    number_diagnoses: float

    age_midpoint: float
    total_prior_visits: float
    medication_change_count: float

    # --------------------------------------------------------
    # Categorical features — 35
    # --------------------------------------------------------

    race: str
    gender: str

    admission_type_id: str
    discharge_disposition_id: str
    admission_source_id: str

    payer_code: str
    max_glu_serum: str
    A1Cresult: str

    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    insulin: str

    glyburide_metformin: str = Field(
        alias="glyburide-metformin"
    )

    glipizide_metformin: str = Field(
        alias="glipizide-metformin"
    )

    glimepiride_pioglitazone: str = Field(
        alias="glimepiride-pioglitazone"
    )

    metformin_rosiglitazone: str = Field(
        alias="metformin-rosiglitazone"
    )

    metformin_pioglitazone: str = Field(
        alias="metformin-pioglitazone"
    )

    change: str
    diabetesMed: str

    diag_1_category: str
    diag_2_category: str
    diag_3_category: str

    medical_specialty_topk: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "hospital_readmission_xgboost",
        "version": "production"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    # --------------------------------------------------------
    # Convert validated Pydantic input to dictionary
    # using the exact ML model column names.
    # --------------------------------------------------------

    patient_data = patient.model_dump(
        by_alias=True
    )

    # --------------------------------------------------------
    # Create one-row DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [patient_data]
    )

    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # --------------------------------------------------------
    # Classification threshold
    # --------------------------------------------------------

    prediction = int(
        probability >= 0.5
    )

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "readmission_probability": round(
            probability,
            4
        ),
        "risk": (
            "high"
            if prediction == 1
            else "low"
        ),
        "model": "hospital_readmission_xgboost",
        "version": "production"
    }