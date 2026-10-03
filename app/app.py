"""
=============================================================================
app/app.py
Patient Health Risk Segmentation – Streamlit Dashboard
=============================================================================

This is the deployment application.
It loads the pre-trained model bundle (models/model_bundle.joblib) and:
  • Accepts new policyholder inputs.
  • Applies the same imputer + scaler used during training.
  • Predicts the K-Means cluster.
  • Maps the cluster to a descriptive profile name.
  • Displays cluster averages, comparison table, and a generic preventive programme.
  • Shows model analysis plots (elbow, silhouette, PCA, dendrogram).
  • Shows all cluster profiles in a tabular view.

Run:
    streamlit run app/app.py
=============================================================================
"""

import inspect
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─── Page config MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="Patient Health Risk Segmentation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Resolve paths relative to this file ─────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "models", "model_bundle.joblib")
PLOTS_DIR   = os.path.join(BASE_DIR, "reports", "plots")


# =============================================================================
# CUSTOM CSS – Premium Healthcare Theme
# =============================================================================
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ─────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8eaf6;
}

/* ── Main header ─────────────────────────────────────────── */
.main-header {
    background: linear-gradient(90deg, #1565C0 0%, #0D47A1 50%, #283593 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(21,101,192,0.4);
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
}

/* ── Cards ───────────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    transition: box-shadow 0.3s ease;
}
.card:hover {
    box-shadow: 0 4px 24px rgba(21,101,192,0.2);
}
.card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #90CAF9;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Metric boxes ────────────────────────────────────────── */
.metric-box {
    background: linear-gradient(135deg, rgba(21,101,192,0.3), rgba(13,71,161,0.2));
    border: 1px solid rgba(21,101,192,0.5);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-box:hover { transform: translateY(-2px); }
.metric-label {
    font-size: 0.78rem;
    color: #90CAF9;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #ffffff;
}

/* ── Cluster result banner ───────────────────────────────── */
.result-banner {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
    border: 1px solid rgba(76,175,80,0.4);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-banner.moderate {
    background: linear-gradient(135deg, #E65100 0%, #F57C00 100%);
    border-color: rgba(255,152,0,0.4);
}
.result-banner.higher {
    background: linear-gradient(135deg, #B71C1C 0%, #C62828 100%);
    border-color: rgba(244,67,54,0.4);
}
.result-cluster {
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin-bottom: 0.3rem;
}
.result-profile {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
}

/* ── Programme bullets ───────────────────────────────────── */
.prog-item {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.95rem;
}
.prog-icon { font-size: 1.1rem; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    color: rgba(255,255,255,0.6) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(21,101,192,0.5) !important;
    color: #ffffff !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 0.65rem 1.8rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(21,101,192,0.6);
    background: linear-gradient(135deg, #1976D2, #1565C0);
}

/* ── Inputs ──────────────────────────────────────────────── */
input, .stNumberInput input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* ── DataFrames / tables ─────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Disclaimer box ──────────────────────────────────────── */
.disclaimer {
    background: rgba(255,193,7,0.12);
    border: 1px solid rgba(255,193,7,0.35);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #FFE082;
    line-height: 1.5;
}

/* ── Section separator ───────────────────────────────────── */
.section-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD MODEL BUNDLE
# =============================================================================

@st.cache_resource(show_spinner="Loading model bundle …")
def load_bundle(path: str) -> dict:
    """Load and cache the Joblib model bundle."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


bundle = load_bundle(BUNDLE_PATH)

if bundle is None:
    st.error(
        "⚠️ **Model bundle not found.**  "
        "Please run `python train_model.py` first, then restart the app."
    )
    st.stop()

# Unpack bundle components
km_model        = bundle["model"]
imputer         = bundle["imputer"]
scaler          = bundle["scaler"]
pca_model       = bundle["pca"]
FEATURES        = bundle["features"]
selected_k      = bundle["selected_k"]
cluster_profiles: pd.DataFrame = bundle.get("cluster_profiles") if "cluster_profiles" in bundle else bundle.get("cluster_profile")
if isinstance(cluster_profiles, pd.DataFrame) and "cluster" not in cluster_profiles.columns and cluster_profiles.index.name == "cluster":
    cluster_profiles = cluster_profiles.reset_index()
profile_name_map: dict         = bundle.get("profile_name_map") if "profile_name_map" in bundle else bundle.get("cluster_names")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def predict_cluster(input_dict: dict) -> tuple[int, str]:
    """
    Apply the saved preprocessing pipeline and predict K-Means cluster.
    Returns: (cluster_id, descriptive_profile_name)
    """
    row = pd.DataFrame([input_dict])[FEATURES]
    X_imputed = pd.DataFrame(imputer.transform(row), columns=FEATURES)
    X_scaled  = scaler.transform(X_imputed)
    cluster   = int(km_model.predict(X_scaled)[0])
    profile   = profile_name_map.get(cluster, f"Cluster {cluster}")
    return cluster, profile


def get_cluster_row(cluster_id: int) -> pd.Series:
    """Return the profile row for a given cluster id."""
    return cluster_profiles[cluster_profiles["cluster"] == cluster_id].iloc[0]


def get_banner_class(profile_name: str) -> str:
    """Return CSS class based on profile name."""
    name_lower = profile_name.lower()
    if "higher" in name_lower:
        return "higher"
    if "moderate" in name_lower or "upper-moderate" in name_lower:
        return "moderate"
    return ""  # lower → green (default)


def get_programme(profile_name: str) -> list:
    """Return generic preventive programme items for the given profile."""
    name_lower = profile_name.lower()

    if "lower" in name_lower and "moderate" not in name_lower:
        return [
            ("🏃", "Maintain regular physical activity"),
            ("🥗", "Balanced nutrition"),
            ("🩺", "Routine preventive screening"),
            ("✨", "Healthy lifestyle maintenance"),
        ]
    elif "higher" in name_lower:
        return [
            ("🩺", "Regular health review"),
            ("🚶", "Appropriate physical activity"),
            ("🥗", "Nutrition and healthy weight management"),
            ("📋", "Routine monitoring and preventive screening"),
        ]
    else:  # moderate or mixed
        return [
            ("🏃", "Increase physical activity"),
            ("🥗", "Balanced nutrition"),
            ("⚖️", "Healthy weight management"),
            ("🩺", "Routine health screening"),
        ]


# Streamlit <= 1.37 uses `use_column_width`, while >= 1.39 uses `use_container_width`.
# Dynamically selecting the parameter via kwargs avoids static analyzer / linter errors.
_IMAGE_WIDTH_PARAM = (
    "use_container_width"
    if "use_container_width" in inspect.signature(st.image).parameters
    else "use_column_width"
)


def plot_image(path: str, caption: str = "", use_container_width: bool = True):
    """Display an image if it exists."""
    if os.path.exists(path):
        st.image(path, caption=caption, **{_IMAGE_WIDTH_PARAM: use_container_width})
    else:
        st.info(f"Plot not found: `{path}`. Run `train_model.py` to generate it.")


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🩺 Model Information")
    st.markdown("---")
    st.markdown(f"""
**Selected Model**  
`K-Means Clustering`

**Number of Clusters**  
`{selected_k}`

**Comparison Model**  
`Agglomerative Hierarchical Clustering`

**Dimensionality Reduction**  
`PCA (2 components)`

**Random Seed**  
`{bundle['seed']}`
""")

    st.markdown("---")
    st.markdown("### ⚠️ Educational Disclaimer")
    st.markdown("""
<div class="disclaimer">
This project uses <b>synthetic, student generated data</b>.
All results are for <b>academic and educational purposes only</b>.<br><br>
Cluster assignments are <b>NOT</b> medical diagnoses, clinical risk predictions,
or insurance premium decisions.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"""
- **Records**: 1 200 synthetic policyholders  
- **Features**: {len(FEATURES)} health indicators  
- **Missing values**: ~3 % (imputed with median)  
- **Scaling**: StandardScaler
""")


# =============================================================================
# PAGE HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🩺 Patient Health Risk Segmentation</h1>
    <p>Health profile segmentation using unsupervised machine learning</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TABS
# =============================================================================

tab_predict, tab_profiles, tab_analysis, tab_about = st.tabs([
    "🔎 Prediction",
    "📊 Cluster Profiles",
    "📈 Model Analysis",
    "ℹ️ About the Project",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ════════════════════════════════════════════════════════════════════════════

with tab_predict:

    st.markdown("### Enter Policyholder Health Indicators")
    st.markdown(
        "Fill in the values below and click **🔎 Assign Risk Group** "
        "to receive the cluster assignment and preventive programme."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="card"><div class="card-title">👤 Personal</div>', unsafe_allow_html=True)
            age = st.number_input("Age (years)", min_value=18, max_value=85,
                                  value=45, step=1, key="age_input")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">💉 Blood Pressure</div>', unsafe_allow_html=True)
            systolic_bp = st.slider("Systolic BP (mmHg)", 90, 190, 125, key="sys_input")
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 55, 120, 82, key="dia_input")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><div class="card-title">📏 Body Measurement</div>', unsafe_allow_html=True)
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=45.0,
                                  value=25.0, step=0.1, format="%.1f", key="bmi_input")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">🔬 Laboratory</div>', unsafe_allow_html=True)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=110, max_value=330,
                                          value=200, step=1, key="chol_input")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="card"><div class="card-title">🏃 Lifestyle</div>', unsafe_allow_html=True)
            physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 12.0,
                                          5.0, step=0.5, key="pa_input")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">📋 Claims History</div>', unsafe_allow_html=True)
            previous_claims = st.number_input("Previous Claims", min_value=0, max_value=12,
                                              value=2, step=1, key="claims_input")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔎 Assign Risk Group", use_container_width=True)

    # ── Handle prediction ──────────────────────────────────────────────────
    if submit:
        user_input = {
            "age":               age,
            "bmi":               bmi,
            "systolic_bp":       systolic_bp,
            "diastolic_bp":      diastolic_bp,
            "cholesterol":       cholesterol,
            "physical_activity": physical_activity,
            "previous_claims":   previous_claims,
        }

        cluster_id, profile_name = predict_cluster(user_input)
        cluster_row = get_cluster_row(cluster_id)
        banner_cls  = get_banner_class(profile_name)

        # ── Result banner ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Assigned Group")
        st.markdown(f"""
<div class="result-banner {banner_cls}">
    <div class="result-cluster">Cluster {cluster_id}</div>
    <div class="result-profile">{profile_name}</div>
</div>
""", unsafe_allow_html=True)

        # ── Entered values + Cluster averages comparison ───────────────────
        st.markdown("---")
        st.markdown("### 📋 Prediction Details")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Your Values")
            user_df = pd.DataFrame({
                "Indicator": [
                    "Age", "BMI", "Systolic BP", "Diastolic BP",
                    "Cholesterol", "Physical Activity", "Previous Claims"
                ],
                "Value": [
                    f"{age} yrs", f"{bmi:.1f} kg/m²",
                    f"{systolic_bp} mmHg", f"{diastolic_bp} mmHg",
                    f"{cholesterol} mg/dL",
                    f"{physical_activity:.1f} hrs/wk",
                    str(previous_claims),
                ],
            })
            st.dataframe(user_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### Cluster Averages")
            avg_df = pd.DataFrame({
                "Indicator": [
                    "Age", "BMI", "Systolic BP", "Diastolic BP",
                    "Cholesterol", "Physical Activity", "Previous Claims"
                ],
                "Cluster Average": [
                    f"{cluster_row['age']:.1f} yrs",
                    f"{cluster_row['bmi']:.1f} kg/m²",
                    f"{cluster_row['systolic_bp']:.1f} mmHg",
                    f"{cluster_row['diastolic_bp']:.1f} mmHg",
                    f"{cluster_row['cholesterol']:.1f} mg/dL",
                    f"{cluster_row['physical_activity']:.1f} hrs/wk",
                    f"{cluster_row['previous_claims']:.1f}",
                ],
            })
            st.dataframe(avg_df, use_container_width=True, hide_index=True)

        # ── Comparison table ───────────────────────────────────────────────
        st.markdown("#### Comparison Table")
        user_vals = [age, bmi, systolic_bp, diastolic_bp,
                     cholesterol, physical_activity, previous_claims]
        avg_vals  = [cluster_row["age"], cluster_row["bmi"],
                     cluster_row["systolic_bp"], cluster_row["diastolic_bp"],
                     cluster_row["cholesterol"], cluster_row["physical_activity"],
                     cluster_row["previous_claims"]]

        indicators = ["Age (yrs)", "BMI (kg/m²)", "Systolic BP (mmHg)",
                      "Diastolic BP (mmHg)", "Cholesterol (mg/dL)",
                      "Physical Activity (hrs/wk)", "Previous Claims"]

        comp_df = pd.DataFrame({
            "Indicator":       indicators,
            "User Value":      [f"{v:.1f}" for v in user_vals],
            "Cluster Average": [f"{v:.1f}" for v in avg_vals],
            "Difference":      [
                f"{(u - a):+.1f}" for u, a in zip(user_vals, avg_vals)
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── Metric summary boxes ───────────────────────────────────────────
        st.markdown("---")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
<div class="metric-box">
    <div class="metric-label">Assigned Cluster</div>
    <div class="metric-value">#{cluster_id}</div>
</div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
<div class="metric-box">
    <div class="metric-label">Profile</div>
    <div class="metric-value" style="font-size:1rem">{profile_name}</div>
</div>""", unsafe_allow_html=True)
        with mc3:
            cluster_size = int(cluster_row["size"])
            st.markdown(f"""
<div class="metric-box">
    <div class="metric-label">Cluster Size</div>
    <div class="metric-value">{cluster_size:,}</div>
</div>""", unsafe_allow_html=True)

        # ── Preventive programme ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏥 Suggested Preventive Programme")
        st.caption(
            "⚠️ **This recommendation is a generic educational example and "
            "is NOT medical advice.** Please consult a qualified healthcare professional."
        )

        programme = get_programme(profile_name)
        for icon, text in programme:
            st.markdown(
                f'<div class="prog-item">'
                f'<span class="prog-icon">{icon}</span>'
                f'<span>{text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – CLUSTER PROFILES
# ════════════════════════════════════════════════════════════════════════════

with tab_profiles:
    st.markdown("### 📊 Cluster Profile Summary")
    st.markdown(
        "The table below shows the mean characteristics of each cluster "
        "computed from **original (unscaled) feature values**."
    )

    # Build display table
    display_profiles = cluster_profiles.copy()
    display_profiles["profile_name"] = display_profiles["cluster"].map(profile_name_map)
    display_profiles["Average BP"] = display_profiles.apply(
        lambda r: f"{r['systolic_bp']:.0f} / {r['diastolic_bp']:.0f} mmHg", axis=1
    )

    display_profiles.rename(columns={
        "cluster":           "Cluster",
        "profile_name":      "Profile",
        "size":              "Size",
        "age":               "Average Age",
        "bmi":               "Average BMI",
        "cholesterol":       "Average Cholesterol",
        "physical_activity": "Average Activity",
        "previous_claims":   "Average Claims",
    }, inplace=True)

    col_order = ["Cluster", "Profile", "Size", "Average Age", "Average BMI",
                 "Average BP", "Average Cholesterol", "Average Activity", "Average Claims"]
    display_profiles = display_profiles[[c for c in col_order if c in display_profiles.columns]]

    st.dataframe(display_profiles, use_container_width=True, hide_index=True)

    # ── Per-cluster interpretation cards ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Cluster Interpretations")

    sorted_clusters = cluster_profiles.sort_values("cluster")

    interpretations = {
        "lower": (
            "Policyholders in this group reflect the healthiest baseline indicators. "
            "They are typically younger adults with normal BMI, optimal blood pressure (<120/80 mmHg), "
            "favourable cholesterol levels, high weekly physical activity, and a minimal history of insurance claims. "
            "Preventive programmes for this cohort focus on sustaining healthy habits and routine wellness screenings."
        ),
        "moderate": (
            "This cohort represents an intermediate profile characterized by middle age, slightly elevated BMI (overweight range), "
            "pre-hypertensive blood pressure, borderline cholesterol, moderate physical activity, and moderate claims. "
            "Preventive recommendations emphasize lifestyle modifications, cardiovascular conditioning, and nutritional guidance."
        ),
        "higher": (
            "This segment presents the highest cluster indicators, characterized by older policyholders with higher BMI (obesity range), "
            "stage-1 or stage-2 hypertension, elevated cholesterol, low/sedentary physical activity, and higher claims frequency. "
            "Preventive strategies focus on regular physician reviews, structured gentle activity, and ongoing health monitoring."
        ),
    }

    for _, row in sorted_clusters.iterrows():
        cid  = int(row["cluster"])
        name = profile_name_map.get(cid, f"Cluster {cid}")
        banner_cls = get_banner_class(name)

        colour_map = {"": "#1B5E20", "moderate": "#E65100", "higher": "#B71C1C"}
        border_colour = colour_map.get(banner_cls, "#1B5E20")

        # Determine interpretation text
        interp_key = "lower"
        if "higher" in name.lower():
            interp_key = "higher"
        elif "moderate" in name.lower():
            interp_key = "moderate"
        interp_text = interpretations[interp_key]

        st.markdown(f"""
<div class="card" style="border-left: 4px solid {border_colour};">
    <div class="card-title">Cluster {cid} — {name}</div>
    <p style="color:rgba(255,255,255,0.85); font-size:0.92rem; margin-bottom:1rem; line-height:1.5;">
        {interp_text}
    </p>
    <table style="width:100%; border-collapse:collapse; font-size:0.92rem;">
        <tr>
            <td style="padding:4px 8px; color:#90CAF9;">Cluster Size</td>
            <td style="padding:4px 8px; font-weight:600;">{int(row['size'])}</td>
            <td style="padding:4px 8px; color:#90CAF9;">Average Age</td>
            <td style="padding:4px 8px; font-weight:600;">{row['age']:.1f} yrs</td>
        </tr>
        <tr>
            <td style="padding:4px 8px; color:#90CAF9;">Average BMI</td>
            <td style="padding:4px 8px; font-weight:600;">{row['bmi']:.1f}</td>
            <td style="padding:4px 8px; color:#90CAF9;">Average Cholesterol</td>
            <td style="padding:4px 8px; font-weight:600;">{row['cholesterol']:.1f} mg/dL</td>
        </tr>
        <tr>
            <td style="padding:4px 8px; color:#90CAF9;">Average BP</td>
            <td style="padding:4px 8px; font-weight:600;">{row['systolic_bp']:.0f} / {row['diastolic_bp']:.0f} mmHg</td>
            <td style="padding:4px 8px; color:#90CAF9;">Average Activity</td>
            <td style="padding:4px 8px; font-weight:600;">{row['physical_activity']:.1f} hrs/wk</td>
        </tr>
        <tr>
            <td style="padding:4px 8px; color:#90CAF9;">Average Claims</td>
            <td style="padding:4px 8px; font-weight:600;">{row['previous_claims']:.1f}</td>
            <td style="padding:4px 8px; color:#90CAF9;">Risk Tier</td>
            <td style="padding:4px 8px; font-weight:600;">{name}</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    st.markdown("### 📈 Model Analysis & Visualizations")

    sub1, sub2 = st.tabs(["K-Means Evaluation", "Clustering Visualizations"])

    with sub1:
        col_e, col_s = st.columns(2)
        with col_e:
            st.markdown("#### Elbow Curve (WCSS)")
            plot_image(os.path.join(PLOTS_DIR, "elbow_curve.png"),
                       "Elbow curve – WCSS vs. number of clusters")
            st.caption(
                "The elbow point suggests where adding more clusters yields "
                "diminishing reduction in WCSS. The red dashed line marks the selected K."
            )
        with col_s:
            st.markdown("#### Silhouette Score")
            plot_image(os.path.join(PLOTS_DIR, "silhouette_curve.png"),
                       "Silhouette score vs. number of clusters")
            st.caption(
                "Higher silhouette scores indicate better-separated, well-defined clusters. "
                "The selected K maximises this score for K ≤ 5."
            )

    with sub2:
        col_p, col_d = st.columns(2)
        with col_p:
            st.markdown("#### PCA – 2D Cluster Visualization")
            plot_image(os.path.join(PLOTS_DIR, "pca_clusters.png"),
                       "PCA projection coloured by K-Means cluster label")
            st.caption(
                "PCA reduces the 7 scaled features to 2 principal components for visualization. "
                "It is **not** used as the clustering algorithm. ✕ marks cluster centroids."
            )
        with col_d:
            st.markdown("#### Dendrogram – Hierarchical Clustering")
            plot_image(os.path.join(PLOTS_DIR, "dendrogram.png"),
                       "Hierarchical clustering dendrogram (Ward linkage)")
            st.caption(
                "The red dashed line shows the cut height that yields the same number of "
                "clusters as selected by K-Means. Hierarchical clustering is used as a "
                "comparison model only and is **not** deployed for prediction."
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – ABOUT THE PROJECT
# ════════════════════════════════════════════════════════════════════════════

with tab_about:
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
## About This Project

**Patient Health Risk Segmentation Using Clustering** is a B.Tech academic machine-learning
project demonstrating unsupervised learning techniques applied to a synthetic healthcare scenario.

### Problem Statement
A health insurance provider wants to group policyholders according to health-related indicators
to support the design of **preventive health programmes**.

### Objectives
- Generate a realistic synthetic dataset of 1 200 policyholders.
- Handle missing clinical values through median imputation.
- Discover natural groupings using **K-Means** and **Hierarchical Clustering**.
- Evaluate clustering quality with WCSS and Silhouette Score.
- Visualise cluster structure in 2-D using **PCA**.
- Profile each cluster and assign descriptive (non-medical) labels.
- Deploy an interactive dashboard for new-record assignment.

### ML Pipeline
1. Synthetic data generation (NumPy)
2. ~3% missing-value injection → median imputation (SimpleImputer)
3. Feature scaling (StandardScaler)
4. K-Means (K = 2–8) with elbow + silhouette analysis
5. Agglomerative Hierarchical Clustering (Ward) + dendrogram
6. PCA (2 components) for visualisation
7. Cluster profiling on original values
8. Model bundle saved with Joblib

### Why K-Means for Deployment?
K-Means exposes a `.predict()` method that can assign any **new** record to an existing cluster
without retraining. Hierarchical clustering in scikit-learn does not provide this workflow,
making K-Means the natural choice for the deployment component.
""")

    with col_r:
        st.markdown("""
## ⚖️ Ethics & Limitations

### Synthetic Data
The dataset is entirely artificial and does not represent any real population.
All statistics, cluster boundaries, and assignments are educational artefacts.

### Not Medical Advice
Cluster assignments are statistical groupings — **not** medical diagnoses, clinical
risk scores, or insurance decisions.

### Potential Bias
Real-world health data may carry demographic, socioeconomic, measurement,
or reporting biases that could affect clustering results.

### Privacy
Real patient data would require strict privacy controls (HIPAA, GDPR, etc.).
No real data is used here.

### Insurance Fairness
Health-indicator clusters must **not** be used to increase premiums,
deny coverage, or restrict access to healthcare services.

### Stigmatisation
Labels such as "higher-indicator profile" can carry unintended social stigma
and must be handled with sensitivity in any real-world application.

### Self-Reported Data
In practice, many measurements are self-reported and may be incomplete or inaccurate.

### Model Limitations
- Clustering does not imply causality.
- Results depend on the choice of K and random seed.
- Synthetic data may not capture the complexity of real populations.
""")

    st.markdown("---")
    st.markdown("""
### 🛠️ Technology Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas / NumPy | Data manipulation and generation |
| Scikit-learn | KMeans, AgglomerativeClustering, SimpleImputer, StandardScaler, PCA |
| SciPy | Linkage matrix & dendrogram |
| Matplotlib / Seaborn | Visualisations |
| Joblib | Model serialisation |
| Streamlit | Interactive dashboard |

### 📁 Project Structure
```
Patient_Health_Risk_Segmentation/
├── app/app.py                   ← This dashboard
├── data/health_policyholders.csv
├── models/model_bundle.joblib   ← Saved model artefacts
├── reports/
│   ├── cluster_profiles.csv
│   ├── cluster_profiles_named.csv
│   ├── cluster_sizes.csv
│   ├── model_metrics.csv
│   ├── methodology.md
│   └── plots/  (elbow, silhouette, pca, dendrogram)
├── train_model.py               ← Full training pipeline
├── notebooks/analysis.py        ← EDA notebook equivalent
├── requirements.txt
└── README.md
```
""")

    st.markdown("---")
    st.caption(
        "🎓 Academic project · Synthetic data only · "
        "Not for clinical or commercial use · "
        f"Clusters discovered: {selected_k}"
    )
