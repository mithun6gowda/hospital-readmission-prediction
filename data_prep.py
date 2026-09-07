import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


RANDOM_STATE = 42


# ============================================================
# ICD-9 GROUPING
# ============================================================

def icd9_to_category(code):
    if pd.isna(code):
        return "Unknown"

    code = str(code).strip()

    if code.startswith(("V", "E")):
        return "Other"

    try:
        code_num = float(code)
    except ValueError:
        return "Unknown"

    if 140 <= code_num <= 239:
        return "Neoplasms"
    elif 250 <= code_num < 251:
        return "Diabetes"
    elif 240 <= code_num <= 279:
        return "Endocrine/Metabolic"
    elif 280 <= code_num <= 289:
        return "Blood"
    elif 290 <= code_num <= 319:
        return "Mental"
    elif 320 <= code_num <= 359:
        return "Nervous"
    elif 390 <= code_num <= 459:
        return "Circulatory"
    elif 460 <= code_num <= 519:
        return "Respiratory"
    elif 520 <= code_num <= 579:
        return "Digestive"
    elif 580 <= code_num <= 629:
        return "Genitourinary"
    elif 630 <= code_num <= 679:
        return "Pregnancy"
    elif 680 <= code_num <= 709:
        return "Skin"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 740 <= code_num <= 759:
        return "Congenital"
    elif 760 <= code_num <= 779:
        return "Perinatal"
    elif 780 <= code_num <= 799:
        return "Symptoms"
    elif 800 <= code_num <= 999:
        return "Injury"

    return "Unknown"


# ============================================================
# LOAD + CLEAN + FEATURE ENGINEERING
# ============================================================

def load_and_prepare_data(data_path):

    df = pd.read_csv(data_path, na_values=["?"])

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df["target"] = (df["readmitted"] == "<30").astype(int)

    # --------------------------------------------------------
    # Remove high-missingness column
    # --------------------------------------------------------

    if "weight" in df.columns:
        df = df.drop(columns=["weight"])

    # --------------------------------------------------------
    # Remove constant / zero-variance columns
    # --------------------------------------------------------

    constant_columns = [
        col for col in df.columns
        if col != "target" and df[col].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        df = df.drop(columns=constant_columns)

    # --------------------------------------------------------
    # Keep first encounter per patient
    # --------------------------------------------------------

    df = (
        df.sort_values(["patient_nbr", "encounter_id"])
          .drop_duplicates(
              subset="patient_nbr",
              keep="first"
          )
          .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove invalid discharge dispositions
    # --------------------------------------------------------

    invalid_discharge_codes = [11, 13, 14, 19, 20, 21]

    df = df[
        ~df["discharge_disposition_id"].isin(
            invalid_discharge_codes
        )
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Recreate target after cleaning
    # --------------------------------------------------------

    df["target"] = (df["readmitted"] == "<30").astype(int)

    # --------------------------------------------------------
    # ICD-9 categories
    # --------------------------------------------------------

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_category"] = df[col].apply(
            icd9_to_category
        )

    # --------------------------------------------------------
    # Age midpoint
    # --------------------------------------------------------

    age_mapping = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95
    }

    df["age_midpoint"] = df["age"].map(age_mapping)

    # --------------------------------------------------------
    # Total prior visits
    # --------------------------------------------------------

    df["total_prior_visits"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    # --------------------------------------------------------
    # Medication change count
    # --------------------------------------------------------

    medication_columns = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]

    available_medication_columns = [
        col for col in medication_columns
        if col in df.columns
    ]

    df["medication_change_count"] = (
        df[available_medication_columns]
        .isin(["Up", "Down"])
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Top-K medical specialties
    # --------------------------------------------------------

    top_specialties = (
        df["medical_specialty"]
        .value_counts()
        .head(10)
        .index
    )

    df["medical_specialty_topk"] = (
        df["medical_specialty"]
        .where(
            df["medical_specialty"].isin(top_specialties),
            "Other"
        )
        .fillna("Unknown")
    )

    return df


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

def get_feature_lists(df):

    numeric_features = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "age_midpoint",
        "total_prior_visits",
        "medication_change_count"
    ]

    excluded_raw_features = [
        "age",
        "diag_1",
        "diag_2",
        "diag_3",
        "medical_specialty"
    ]

    excluded_features = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "target"
    ] + excluded_raw_features

    categorical_features = [
        col
        for col in df.columns
        if col not in numeric_features
        and col not in excluded_features
    ]

    return numeric_features, categorical_features


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X_train):

    numeric_features, categorical_features = get_feature_lists(
        X_train
    )

    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ], remainder="drop")

    return preprocessor


# ============================================================
# PREPARE DATA FOR MODEL DEVELOPMENT
# ============================================================

def prepare(data_path):

    df = load_and_prepare_data(data_path)

    numeric_features, categorical_features = get_feature_lists(df)

    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()

    y = df["target"].copy()

    # --------------------------------------------------------
    # 70 / 15 / 15 stratified split
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return (
        X,
        y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    df = load_and_prepare_data(data_path)

    X = df.drop(
        columns=["target"]
    ).copy()

    y = df["target"].copy()

    # --------------------------------------------------------
    # 70 / 15 / 15 stratified split
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return (
        X,
        y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )