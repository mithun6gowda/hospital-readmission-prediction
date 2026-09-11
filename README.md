# Hospital Readmission Prediction — End-to-End MLOps

An end-to-end **MLOps project for predicting 30-day hospital readmission** using the Diabetes 130-US Hospitals dataset.

The project goes beyond model development and demonstrates the complete machine-learning lifecycle:

**Data Preparation → Feature Engineering → Model Development → Experiment Tracking → Model Registry → REST API → Docker → CI/CD → Drift Monitoring → Automated Retraining**

---

## 📌 Project Overview

Hospital readmissions within 30 days are an important healthcare quality and operational concern.

This project builds a binary classification model to predict whether a patient will be **readmitted within 30 days of discharge**.

The target is defined as:

* `1` → Readmitted within 30 days (`<30`)
* `0` → Not readmitted within 30 days (`NO` or `>30`)

The project is implemented as a reproducible MLOps pipeline with model serving, testing, monitoring, and retraining capabilities.

---

## 🎯 Project Objectives

The main objectives are to:

* Build a leakage-safe machine learning pipeline.
* Engineer clinically meaningful features.
* Handle class imbalance.
* Compare multiple ML models.
* Track experiments using MLflow.
* Register and version the selected model.
* Serve predictions through FastAPI.
* Containerize the API using Docker.
* Automate testing using GitHub Actions.
* Detect feature drift using Evidently.
* Detect prediction drift using Population Stability Index (PSI).
* Automatically retrain the model when monitoring thresholds are exceeded.
* Maintain model-version and retraining decision evidence.

---

## 🏗️ MLOps Architecture

```text
                    ┌──────────────────────┐
                    │ Diabetes Hospital    │
                    │ Dataset              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preparation     │
                    │ & Feature Engineering│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Train / Validation   │
                    │ / Test Split         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Development    │
                    │ Logistic Regression  │
                    │ XGBoost              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ MLflow Tracking      │
                    │ & Model Registry     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ Production Model     │
                    │ XGBoost              │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
      ┌──────────────────┐          ┌──────────────────┐
      │ FastAPI          │          │ Monitoring       │
      │ /health          │          │ Evidently + PSI  │
      │ /predict         │          └────────┬─────────┘
      └────────┬─────────┘                   │
               │                             ▼
               ▼                    ┌──────────────────┐
      ┌──────────────────┐          │ Retraining       │
      │ Docker Container │          │ Trigger          │
      └──────────────────┘          └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ Retrain + Register│
                                   │ New Model Version │
                                   └──────────────────┘
```

---

# 📊 Dataset

The project uses the **Diabetes 130-US Hospitals** dataset.

### Dataset statistics

| Item                 |   Value |
| -------------------- | ------: |
| Original records     | 101,766 |
| Original features    |      50 |
| Unique patients      |  71,518 |
| Final modelling rows |  69,973 |
| Final model features |      46 |
| Positive class       |   8.97% |

The dataset contains patient demographics, admission information, diagnoses, procedures, medications, laboratory information, and prior utilization information.

---

# 🧹 Data Preparation

The data preparation pipeline is implemented in:

```text
data_prep.py
```

Major preprocessing steps include:

### Missing values

The dataset's `?` values are converted into missing values.

Highly incomplete fields such as `weight` are removed.

### Patient-level deduplication

Only the first encounter per patient is retained to avoid multiple encounters for the same patient appearing in the modelling population.

### Invalid discharge records

The following discharge disposition IDs are removed:

```text
11, 13, 14, 19, 20, 21
```

### Target creation

```python
target = 1 if readmitted == "<30" else 0
```

### Diagnosis engineering

ICD-9 diagnosis codes are grouped into broader clinical categories such as:

```text
Circulatory
Diabetes
Respiratory
Digestive
Genitourinary
Endocrine/Metabolic
Neoplasms
...
```

### Additional engineered features

The project creates:

* `age_midpoint`
* `total_prior_visits`
* `medication_change_count`
* `medical_specialty_topk`
* `diag_1_category`
* `diag_2_category`
* `diag_3_category`

---

# 🔐 Leakage Prevention

The project explicitly separates model features from identifiers and target information.

The final model uses exactly:

```text
11 numerical features
35 categorical features
-----------------------
46 total features
```

Patient identifiers, encounter identifiers, the target, and raw diagnosis/age/specialty fields are excluded from the final model feature matrix.

The dataset is split **before fitting preprocessing transformations**.

```text
70% Training
15% Validation
15% Test
```

All splits use:

```python
random_state = 42
```

The preprocessing pipeline is fitted only on the training data.

---

# 🧠 Feature Set

## Numerical features

```text
time_in_hospital
num_lab_procedures
num_procedures
num_medications
number_outpatient
number_emergency
number_inpatient
number_diagnoses
age_midpoint
total_prior_visits
medication_change_count
```

## Categorical features

The categorical feature set includes:

```text
race
gender
admission_type_id
discharge_disposition_id
admission_source_id
payer_code
max_glu_serum
A1Cresult
```

along with medication-related variables, diagnosis categories, diabetes medication indicators, medication change indicators, and:

```text
medical_specialty_topk
```

---

# 🤖 Model Development

Two models were evaluated:

1. Logistic Regression
2. XGBoost

Class imbalance was handled using balanced Logistic Regression and `scale_pos_weight` for XGBoost.

## Validation Results

| Metric    | Logistic Regression |    XGBoost |
| --------- | ------------------: | ---------: |
| ROC-AUC   |              0.6413 | **0.6502** |
| PR-AUC    |              0.1703 | **0.1814** |
| Recall    |          **0.5287** |     0.5191 |
| Precision |              0.1362 | **0.1413** |
| F1        |              0.2166 | **0.2221** |
| Accuracy  |              0.6567 | **0.6737** |

XGBoost was selected based on validation performance.

---

# 🏆 Final Model

The selected model is:

```text
XGBoost Classifier
+
Scikit-learn preprocessing pipeline
```

The final untouched test-set performance:

| Metric    | Test Result |
| --------- | ----------: |
| ROC-AUC   |  **0.6557** |
| PR-AUC    |  **0.1814** |
| Recall    |  **0.5197** |
| Precision |  **0.1418** |
| F1        |  **0.2228** |
| Accuracy  |  **0.6749** |

The test set was kept separate from model selection and tuning.

---

# 📈 MLflow Experiment Tracking

The project uses:

```text
MLflow 3.16.0
```

for:

* Experiment tracking
* Parameter logging
* Metric logging
* Model artifact logging
* Model registration
* Model versioning
* Production aliases

Registered model:

```text
hospital_readmission_xgboost
```

Production model lifecycle:

```text
Model Version
      │
      ▼
MLflow Registry
      │
      ▼
production alias
```

The retraining workflow successfully created a new registered model version and moved the `production` alias to it.

---

# 🚀 FastAPI Deployment

The API is implemented in:

```text
app.py
```

## Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "hospital_readmission_xgboost",
  "version": "production"
}
```

### Prediction

```http
POST /predict
```

The endpoint accepts the model's validated 46-feature patient input schema.

Example response:

```json
{
  "prediction": 1,
  "readmission_probability": 0.6143,
  "risk": "high",
  "model": "hospital_readmission_xgboost",
  "version": "production"
}
```

Input validation is implemented using Pydantic.

---

# 🐳 Docker

The API is containerized using:

```text
Dockerfile
```

Build the image:

```bash
docker build -t hospital-readmission-api:latest .
```

Run the container:

```bash
docker run -p 8000:8000 hospital-readmission-api:latest
```

The API is then available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

The Docker container was verified using both:

```text
GET /health
POST /predict
```

---

# 🧪 Testing

Tests are implemented using:

```text
pytest
```

Current API tests cover:

* Health endpoint
* Prediction endpoint
* Input validation

Run locally:

```bash
pytest -q
```

Expected result:

```text
3 passed
```

---

# 🔄 CI/CD

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The workflow:

1. Checks out the repository.
2. Installs Python 3.12.
3. Installs project dependencies.
4. Runs pytest.

Workflow:

```text
Git Push / Pull Request
          │
          ▼
   GitHub Actions
          │
          ▼
   Install Dependencies
          │
          ▼
       pytest
          │
      ┌───┴───┐
      ▼       ▼
    Pass     Fail
```

The GitHub Actions workflow has been successfully executed with passing tests.

---

# 📊 Monitoring and Drift Detection

Monitoring is implemented through:

```text
operations_monitoring_and_evidence.ipynb
monitoring.py
```

The project creates:

```text
reference_sample.csv
current_batch.csv
drift_report.html
drift_summary.json
```

## Reference Dataset

A 5,000-row reference sample is taken from the training data.

```text
5,000 rows × 46 features
```

## Current Batch

A simulated 5,000-row production batch is created with controlled distribution shifts.

The simulation increases:

* Hospital stay duration
* Number of medications
* Age distribution
* Circulatory diagnosis frequency

---

# 🔎 Feature Drift — Evidently

Evidently compares:

```text
Reference Training Data
          VS
Current Production Batch
```

Observed result:

```text
Total features:       46
Drifted features:      4
Drift share:        0.087
```

Drifted features:

```text
time_in_hospital
num_medications
age_midpoint
diag_1_category
```

Dataset drift was **not** triggered because the drift share remained below the retraining threshold.

---

# 📐 Prediction Drift — PSI

Prediction-score drift is measured using Population Stability Index (PSI).

Observed:

```text
Prediction PSI = 0.0262
```

Retraining threshold:

```text
PSI > 0.20
```

Therefore:

```text
Prediction drift = False
```

---

# 🔁 Automated Retraining

The retraining workflow is implemented in:

```text
retrain.py
```

Retraining occurs when **any one** of the following conditions is met:

```text
Prediction PSI > 0.20
OR
Drifted feature share > 0.30
OR
Dataset drift = True
```

The workflow:

```text
Monitoring
    │
    ▼
Evaluate Signals
    │
    ├── No trigger ──► Continue monitoring
    │
    └── Trigger
          │
          ▼
       Retrain
          │
          ▼
       MLflow Log
          │
          ▼
    Register New Version
          │
          ▼
    Update Production Alias
          │
          ▼
   Save Decision Record
```

---

# ✅ Retraining Verification

The workflow was tested with:

```text
Prediction PSI = 0.25
```

Since:

```text
0.25 > 0.20
```

the retraining trigger fired.

The workflow successfully:

* Retrained the XGBoost model
* Logged the model to MLflow
* Registered a new model version
* Updated the `production` alias
* Created `retraining_decision.json`

Latest verified retraining:

```text
MLflow Run:
d5a8010b6c75426a88e5c4dc33012328

Registered Model:
hospital_readmission_xgboost

Registered Version:
6

Alias:
production
```

---

# 📁 Repository Structure

```text
hospital-readmission-prediction/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── test/
│   └── test_api.py
│
├── app.py
├── config.py
├── data_prep.py
├── evaluate.py
├── monitoring.py
├── retrain.py
├── train.py
│
├── data_preparation.ipynb
├── model_development_and_tracking.ipynb
├── operations_monitoring_and_evidence.ipynb
│
├── diabetic_data.csv
├── reference_sample.csv
├── current_batch.csv
│
├── drift_report.html
├── drift_summary.json
├── retraining_decision.json
│
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── mlflow.db
└── mlruns/
```

---

# 🛠️ Technology Stack

| Area                | Technology            |
| ------------------- | --------------------- |
| Language            | Python                |
| Data Processing     | Pandas, NumPy         |
| ML                  | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow                |
| API                 | FastAPI               |
| Validation          | Pydantic              |
| Server              | Uvicorn               |
| Testing             | Pytest                |
| CI/CD               | GitHub Actions        |
| Containerisation    | Docker                |
| Monitoring          | Evidently             |
| Prediction Drift    | PSI                   |
| Notebook            | Jupyter               |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mithun6gowda/hospital-readmission-prediction.git
```

Navigate into the project:

```bash
cd hospital-readmission-prediction
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# ▶️ Run the API Locally

```powershell
uvicorn app:app --reload
```

Open Swagger:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

---

# 🧪 Run Tests

```powershell
pytest -q
```

---

# 📊 Run Monitoring

Open:

```text
operations_monitoring_and_evidence.ipynb
```

The notebook demonstrates:

1. Reference-data creation
2. Current-batch simulation
3. Evidently feature drift
4. Prediction-score generation
5. PSI calculation
6. Retraining decision logic

---

# 🔄 Run Retraining

The retraining workflow can be executed using:

```powershell
python retrain.py
```

The script evaluates the monitoring signals and, when a trigger is present:

```text
Retrain
→ Log to MLflow
→ Register model
→ Update production alias
→ Save decision record
```

---

# ⚠️ Important Implementation Note

The current API implementation uses a bundled model artifact path, while the MLflow registry maintains a `production` alias.

For a fully production-oriented deployment, the API model-loading strategy should be aligned with the registry alias so that promoting a new model version automatically changes the model served by the API without requiring application-code changes.

This is the next governance/hardening step after the implemented monitoring and retraining workflow.

---

# 📌 Key Project Outcomes

This project demonstrates:

* ✅ Leakage-aware ML pipeline
* ✅ Feature engineering
* ✅ Imbalanced classification
* ✅ Model comparison
* ✅ Reproducible train/validation/test split
* ✅ MLflow experiment tracking
* ✅ MLflow model registry
* ✅ Production model alias
* ✅ FastAPI prediction service
* ✅ Pydantic input validation
* ✅ Docker deployment
* ✅ Automated pytest tests
* ✅ GitHub Actions CI
* ✅ Evidently feature drift detection
* ✅ PSI prediction drift detection
* ✅ Multi-signal retraining
* ✅ Automated model registration
* ✅ Retraining decision records

---

# 👨‍💻 Author

**Mithun Gowda**

GitHub:

https://github.com/mithun6gowda

Project:

https://github.com/mithun6gowda/hospital-readmission-prediction

---

## 📄 Project Status

**Current status:** End-to-end MLOps implementation completed through automated retraining, with Logging & Governance hardening as the next stage.

**Model:** XGBoost
**Features:** 46
**Test ROC-AUC:** 0.6557
**Monitoring:** Evidently + PSI
**API:** FastAPI
**Container:** Docker
**CI:** GitHub Actions
**Registry:** MLflow
