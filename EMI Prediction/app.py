import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# =========================== CONFIG ==========================================
DATA_PATH = "emi_prediction_dataset.csv"
MODEL_DIR = "saved_models_v2"
os.makedirs(MODEL_DIR, exist_ok=True)

CLS_PATH = f"{MODEL_DIR}/classifier.pkl"
REG_PATH = f"{MODEL_DIR}/regressor.pkl"
DT_PATH = f"{MODEL_DIR}/dt_regressor.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
ENCODERS_PATH = f"{MODEL_DIR}/encoders.pkl"
TARGETENC_PATH = f"{MODEL_DIR}/target_encoder.pkl"
FEATURELIST_PATH = f"{MODEL_DIR}/feature_list.pkl"

TARGET_CLS = "emi_eligibility"
TARGET_REG = "max_monthly_emi"

# ======================= LOAD & CLEAN DATA ===================================
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert numeric-looking columns
    numeric_cols = [
        "age","monthly_salary","current_emi_amount","requested_amount","requested_tenure",
        "bank_balance","emergency_fund","max_monthly_emi",
        "groceries_utilities","other_monthly_expenses",
        "school_fees","travel_expenses","college_fees"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    if "existing_loans" in df.columns:
        df["existing_loans"] = df["existing_loans"].map({"Yes": 1, "No": 0}).astype(float)

    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df = load_data()

# ======================= SAVE / LOAD HELPERS =================================
def save(obj, path):
    joblib.dump(obj, path, compress=("xz", 3))

def load(path):
    return joblib.load(path)

# ======================= TRAIN OR LOAD MODELS ================================
def train_or_load(df):

    if all(os.path.exists(p) for p in [
        CLS_PATH, REG_PATH, DT_PATH, SCALER_PATH, TARGETENC_PATH, FEATURELIST_PATH, ENCODERS_PATH
    ]):
        return (
            load(CLS_PATH),
            load(REG_PATH),
            load(DT_PATH),
            load(SCALER_PATH),
            load(TARGETENC_PATH),
            load(FEATURELIST_PATH),
            load(ENCODERS_PATH)
        )

    st.warning("⚠ Training new EMI prediction models...")

    data = df.copy()

    le_target = LabelEncoder()
    data[TARGET_CLS] = le_target.fit_transform(data[TARGET_CLS].astype(str))

    feature_list = [c for c in data.columns if c not in [TARGET_CLS, TARGET_REG]]

    X = data[feature_list].copy()
    y_cls = data[TARGET_CLS]
    y_reg = data[TARGET_REG]

    # Encode categorical features
    encoders = {}
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, ycls_train, ycls_test = train_test_split(X, y_cls, test_size=0.25, random_state=42)
    _, _, yreg_train, yreg_test = train_test_split(X, y_reg, test_size=0.25, random_state=42)

    # Models
    cls_model = LogisticRegression(max_iter=400)
    reg_model = LinearRegression()
    dt_model = DecisionTreeRegressor(max_depth=6, random_state=42)

    # Train
    cls_model.fit(X_train, ycls_train)
    reg_model.fit(X_train, yreg_train)
    dt_model.fit(X_train, yreg_train)

    # Save all
    save(cls_model, CLS_PATH)
    save(reg_model, REG_PATH)
    save(dt_model, DT_PATH)
    save(scaler, SCALER_PATH)
    save(encoders, ENCODERS_PATH)
    save(le_target, TARGETENC_PATH)
    save(feature_list, FEATURELIST_PATH)

    return cls_model, reg_model, dt_model, scaler, le_target, feature_list, encoders


# ✅ LOAD MODELS FIRST
cls_model, reg_model, dt_model, scaler, le_target, feature_list, encoders = train_or_load(df)

# ======================= UI CONFIG ===========================================
st.set_page_config(page_title="EMI Dashboard", layout="wide")
st.title("💳 EMI Prediction Dashboard")
st.write("AI-powered EMI Eligibility & EMI Capacity Estimator")

# ======================= SIDEBAR INPUTS ======================================
st.sidebar.header("📌 Enter Applicant Information")

user_inputs = {}
for feat in feature_list:
    if feat not in df.columns:
        continue

    col = df[feat]
    if pd.api.types.is_numeric_dtype(col):
        user_inputs[feat] = st.sidebar.number_input(
            feat.replace("_", " ").title(),
            value=float(col.median())
        )
    else:
        options = sorted(col.astype(str).unique())
        default = str(col.mode()[0])
        user_inputs[feat] = st.sidebar.selectbox(
            feat.replace("_", " ").title(),
            options=options,
            index=options.index(default) if default in options else 0
        )

predict = st.sidebar.button("🔮 Predict EMI Eligibility")

# ======================= PREDICT =============================================
if predict:
    input_df = pd.DataFrame([user_inputs])

    # Encode categorical
    for col in input_df.columns:
        if col in encoders:
            le = encoders[col]
            try:
                input_df[col] = le.transform(input_df[col])
            except:
                input_df[col] = le.transform([le.classes_[0]])[0]

    # Scale numeric
    num_cols = input_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    cls_pred = cls_model.predict(input_df)[0]
    cls_label = le_target.inverse_transform([cls_pred])[0]
    reg_pred_linear = reg_model.predict(input_df)[0]
    reg_pred_tree = dt_model.predict(input_df)[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.success(f"✅ EMI Eligibility: **{cls_label}**")
    with c2:
        st.info(f"💰 EMI Capacity (Linear Regression): **₹ {reg_pred_linear:,.2f}**")
    with c3:
        st.warning(f"🌳 EMI Capacity (Decision Tree): **₹ {reg_pred_tree:,.2f}**")

# ======================= ANALYTICS SECTION =====================================
st.markdown("## 📊 Data Insights")

g1, g2 = st.columns(2)
with g1:
    fig = px.histogram(df, x="monthly_salary", nbins=40, title="Salary Distribution")
    st.plotly_chart(fig, use_container_width=True)

with g2:
    if "credit_score" in df.columns:
        fig = px.box(df, y="credit_score", title="Credit Score Variation")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("## 📈 Salary vs EMI Capacity")
if "monthly_salary" in df.columns and "max_monthly_emi" in df.columns:
    fig = px.scatter(df, x="monthly_salary", y="max_monthly_emi",
                     opacity=0.5, title="Salary vs EMI Capacity")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("✅ EMI Dashboard — Fast, Lightweight, Accurate")
