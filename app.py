import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from catboost import CatBoostClassifier, Pool

from src.feature_engineering import create_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .hero {
        padding: 25px 0px 10px 0px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #667085;
    }

    .card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e4e7ec;
        margin-bottom: 20px;
    }

    .big-number {
        font-size: 48px;
        font-weight: 800;
    }

    .risk-high {
        font-size: 25px;
        font-weight: 800;
    }

    .risk-medium {
        font-size: 25px;
        font-weight: 800;
    }

    .risk-low {
        font-size: 25px;
        font-weight: 800;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .small-text {
        color: #667085;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #98a2b3;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "smartphone_addiction_model.cbm"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = CatBoostClassifier()

    model.load_model(str(MODEL_PATH))

    return model


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <div class="hero-title">
    📱 Smartphone Addiction Predictor
    </div>

    <div class="hero-subtitle">
    Machine-learning dashboard for predicting smartphone
    addiction risk from behavioral and lifestyle patterns.
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR INPUTS
# ============================================================

with st.sidebar:

    st.header("👤 User Profile")

    st.write(
        "Enter behavioral information to generate "
        "a prediction."
    )

    st.divider()

    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=21.0
    )

    daily_screen_time_hours = st.number_input(
        "Daily Screen Time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=6.0
    )

    social_media_hours = st.number_input(
        "Social Media Hours",
        min_value=0.0,
        max_value=24.0,
        value=3.0
    )

    gaming_hours = st.number_input(
        "Gaming Hours",
        min_value=0.0,
        max_value=24.0,
        value=2.0
    )

    work_study_hours = st.number_input(
        "Work / Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=6.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    notifications_per_day = st.number_input(
        "Notifications Per Day",
        min_value=0,
        max_value=1000,
        value=100
    )

    app_opens_per_day = st.number_input(
        "App Opens Per Day",
        min_value=0,
        max_value=1000,
        value=70
    )

    weekend_screen_time = st.number_input(
        "Weekend Screen Time",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High"]
    )

    academic_work_impact = st.selectbox(
        "Academic / Work Impact",
        ["Yes", "No"]
    )

    predict = st.button(
        "🔮 Predict Addiction",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# INPUT DATA
# ============================================================

input_data = pd.DataFrame(
    {
        "age": [age],

        "daily_screen_time_hours":
            [daily_screen_time_hours],

        "social_media_hours":
            [social_media_hours],

        "gaming_hours":
            [gaming_hours],

        "work_study_hours":
            [work_study_hours],

        "sleep_hours":
            [sleep_hours],

        "notifications_per_day":
            [notifications_per_day],

        "app_opens_per_day":
            [app_opens_per_day],

        "weekend_screen_time":
            [weekend_screen_time],

        "gender":
            [gender],

        "stress_level":
            [stress_level],

        "academic_work_impact":
            [academic_work_impact]
    }
)


# ============================================================
# BEFORE PREDICTION
# ============================================================

if not predict:

    st.info(
        "👈 Enter user information from the sidebar "
        "and click **Predict Addiction**."
    )

    st.markdown(
        '<div class="section-title">🤖 Project Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Model",
            "CatBoost"
        )

    with c2:
        st.metric(
            "Validation ROC-AUC",
            "0.959"
        )

    with c3:
        st.metric(
            "Features",
            "17"
        )

    with c4:
        st.metric(
            "Dataset",
            "987K+ rows"
        )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

input_features = create_features(
    input_data.copy()
)


categorical_cols = [
    "gender",
    "stress_level",
    "academic_work_impact"
]

input_features[categorical_cols] = (
    input_features[categorical_cols]
    .fillna("Missing")
    .astype(str)
)


# ============================================================
# PREDICTION
# ============================================================

probability = model.predict_proba(
    input_features
)[0][1]

percentage = probability * 100


# ============================================================
# RISK CLASSIFICATION
# ============================================================

if probability >= 0.70:

    risk = "High Risk"
    risk_icon = "🔴"

elif probability >= 0.40:

    risk = "Moderate Risk"
    risk_icon = "🟡"

else:

    risk = "Low Risk"
    risk_icon = "🟢"


# ============================================================
# RESULT
# ============================================================

st.markdown(
    '<div class="section-title">📊 Prediction Result</div>',
    unsafe_allow_html=True
)

result1, result2, result3 = st.columns(3)


with result1:

    st.metric(
        "Addiction Probability",
        f"{percentage:.2f}%"
    )


with result2:

    st.metric(
        "Risk Classification",
        risk
    )


with result3:

    st.metric(
        "Model",
        "CatBoost"
    )


# ============================================================
# PROBABILITY GAUGE
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Probability Gauge</div>',
    unsafe_allow_html=True
)

st.progress(
    float(probability)
)

st.markdown(
    f"""
    <div style="
        text-align:center;
        font-size:32px;
        font-weight:800;
        margin:10px;
    ">
        {risk_icon} {percentage:.2f}% — {risk}
    </div>
    """,
    unsafe_allow_html=True
)


st.caption(
    "The probability represents the model's estimated probability "
    "of the positive class: addicted_label = 1."
)


# ============================================================
# USER BEHAVIOR DASHBOARD
# ============================================================

st.markdown(
    '<div class="section-title">📱 Behavioral Profile</div>',
    unsafe_allow_html=True
)


b1, b2, b3, b4 = st.columns(4)


with b1:

    st.metric(
        "Daily Screen Time",
        f"{daily_screen_time_hours:.1f} hrs"
    )

    st.metric(
        "Social Media",
        f"{social_media_hours:.1f} hrs"
    )


with b2:

    st.metric(
        "Gaming",
        f"{gaming_hours:.1f} hrs"
    )

    st.metric(
        "Weekend Screen",
        f"{weekend_screen_time:.1f} hrs"
    )


with b3:

    st.metric(
        "Notifications",
        notifications_per_day
    )

    st.metric(
        "App Opens",
        app_opens_per_day
    )


with b4:

    st.metric(
        "Sleep",
        f"{sleep_hours:.1f} hrs"
    )

    st.metric(
        "Work / Study",
        f"{work_study_hours:.1f} hrs"
    )


# ============================================================
# BEHAVIOR VISUALIZATION
# ============================================================

st.markdown(
    '<div class="section-title">📈 Digital Behavior Visualization</div>',
    unsafe_allow_html=True
)

behavior_df = pd.DataFrame(
    {
        "Metric": [
            "Daily Screen Time",
            "Social Media",
            "Gaming",
            "Weekend Screen"
        ],

        "Hours": [
            daily_screen_time_hours,
            social_media_hours,
            gaming_hours,
            weekend_screen_time
        ]
    }
)

st.bar_chart(
    behavior_df.set_index("Metric")
)


# ============================================================
# DERIVED FEATURES
# ============================================================

st.markdown(
    '<div class="section-title">🧮 Feature Engineering</div>',
    unsafe_allow_html=True
)

f1, f2, f3, f4 = st.columns(4)


with f1:

    st.metric(
        "Entertainment Hours",
        f"{input_features['entertainment_hours'].iloc[0]:.2f}"
    )


with f2:

    st.metric(
        "Screen / Sleep Ratio",
        f"{input_features['screen_sleep_ratio'].iloc[0]:.2f}"
    )


with f3:

    st.metric(
        "Weekend / Daily Ratio",
        f"{input_features['weekend_daily_ratio'].iloc[0]:.2f}"
    )


with f4:

    st.metric(
        "Digital Activity",
        f"{input_features['digital_activity'].iloc[0]:.0f}"
    )


# ============================================================
# CATBOOST FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 CatBoost Feature Importance</div>',
    unsafe_allow_html=True
)

importance = model.get_feature_importance()

feature_names = model.feature_names_

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).head(10)

importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("_", " ")
    .str.title()
)

st.bar_chart(
    importance_df.set_index("Feature")
)


st.caption(
    "This chart shows global feature importance: which features "
    "the CatBoost model relied on most across the training data."
)


# ============================================================
# INDIVIDUAL PREDICTION EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Why did the model make this prediction?</div>',
    unsafe_allow_html=True
)

try:

    prediction_pool = Pool(
        input_features,
        cat_features=categorical_cols
    )

    shap_values = model.get_feature_importance(
        prediction_pool,
        type="ShapValues"
    )

    shap_values = shap_values[0][:-1]

    explanation_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Impact": shap_values
        }
    )

    explanation_df["Absolute Impact"] = (
        explanation_df["Impact"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    ).head(8)

    explanation_df["Feature"] = (
        explanation_df["Feature"]
        .str.replace("_", " ")
        .str.title()
    )

    st.bar_chart(
        explanation_df.set_index("Feature")["Impact"]
    )

    st.caption(
        "Positive values push the prediction toward the "
        "addicted class; negative values push it toward "
        "the non-addicted class."
    )

except Exception as e:

    st.warning(
        "Individual prediction explanation unavailable."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "Validation ROC-AUC",
        "0.9591"
    )


with m2:

    st.metric(
        "Algorithm",
        "CatBoostClassifier"
    )


with m3:

    st.metric(
        "Training Iterations",
        "1000"
    )


# ============================================================
# TECH STACK
# ============================================================

st.markdown(
    '<div class="section-title">🛠️ Technology Stack</div>',
    unsafe_allow_html=True
)

st.write(
    """
    **Python** · **Pandas** · **CatBoost** · **Scikit-learn** ·
    **Streamlit** · **Feature Engineering** ·
    """
)


# ============================================================
# DISCLAIMER
# ============================================================



# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    📱 Smartphone Addiction Predictor

    <br>

    Built with Python • CatBoost • Pandas • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)