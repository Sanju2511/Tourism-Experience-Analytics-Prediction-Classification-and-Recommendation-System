from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_tourism.csv"
MODELS_DIR = ROOT / "models"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --bg: #f4efe7;
      --paper: #fffaf3;
      --ink: #1f2a30;
      --accent: #0d6b6b;
      --accent-2: #d96c3c;
      --muted: #62737d;
      --card: #fff;
      --border: #eadfcd;
    }
    .stApp {
      background: radial-gradient(circle at 85% 10%, #ffe8d2 0%, #f4efe7 45%, #eef5f1 100%);
      color: var(--ink);
      font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;
    }
    .block-container {
      padding-top: 1.2rem;
      padding-bottom: 1.4rem;
      max-width: 1200px;
    }
    h1, h2, h3 {
      color: var(--ink);
      letter-spacing: 0.2px;
    }
    .hero {
      background: linear-gradient(130deg, #083d3d, #0d6b6b 45%, #d96c3c 140%);
      border-radius: 18px;
      padding: 20px 24px;
      color: #fff;
      box-shadow: 0 10px 24px rgba(13, 38, 45, 0.2);
      margin-bottom: 16px;
      border: 1px solid rgba(255,255,255,0.22);
    }
    .hero p {
      margin: 0;
      opacity: 0.95;
    }
    .metric-card {
      background: var(--card);
      border-radius: 14px;
      border: 1px solid var(--border);
      padding: 14px 16px;
      box-shadow: 0 6px 18px rgba(31, 42, 48, 0.08);
      min-height: 92px;
    }
    .metric-label { color: var(--muted); font-size: 0.85rem; }
    .metric-value { color: var(--ink); font-size: 1.6rem; font-weight: 700; }
    .section-card {
      background: var(--paper);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 5px 14px rgba(31, 42, 48, 0.07);
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #173b43 0%, #0f5e61 60%, #3b3530 140%);
      border-right: 1px solid rgba(255, 255, 255, 0.12);
    }
    [data-testid="stSidebar"] * {
      color: #f3f8f7 !important;
    }
    .stButton>button {
      background: linear-gradient(120deg, #0d6b6b, #1d837f);
      color: #fff;
      border: 0;
      border-radius: 10px;
      font-weight: 600;
      padding: 0.45rem 1rem;
    }
    .stButton>button:hover { filter: brightness(1.05); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    import json

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_artifacts():
    reg_model = joblib.load(MODELS_DIR / "best_regression_model.pkl")
    clf_model = joblib.load(MODELS_DIR / "best_classification_model.pkl")
    knn = joblib.load(MODELS_DIR / "recommendation_knn.pkl")
    user_item = joblib.load(MODELS_DIR / "user_item_matrix.pkl")
    feature_map = joblib.load(MODELS_DIR / "model_features.pkl")
    return reg_model, clf_model, knn, user_item, feature_map


def mode_lookup_map(df: pd.DataFrame) -> dict:
    return df[["VisitModeId", "VisitMode"]].drop_duplicates().set_index("VisitModeId")["VisitMode"].to_dict()


def recommend_for_user(df: pd.DataFrame, knn, user_item: pd.DataFrame, user_id: int, k: int = 5):
    if user_id not in user_item.index:
        return df["Attraction"].value_counts().head(k).index.tolist()

    _, indices = knn.kneighbors(user_item.loc[user_id].values.reshape(1, -1), n_neighbors=min(6, len(user_item)))
    neighbor_users = user_item.index[indices[0][1:]]

    seen = set(df.loc[df["UserId"] == user_id, "AttractionId"])
    neighbor_scores = (
        df[df["UserId"].isin(neighbor_users)]
        .groupby(["AttractionId", "Attraction"])["Rating"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    recs = neighbor_scores[~neighbor_scores["AttractionId"].isin(seen)].head(k)
    return recs["Attraction"].tolist()


def render_metric(label: str, value: str):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(df: pd.DataFrame):
    st.markdown("""
    <div class='hero'>
      <h2 style='margin:0;'>Tourism Experience Analytics Dashboard</h2>
      <p>Decision-ready analytics combining behavioral segmentation, rating prediction, and attraction recommendation.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Transactions", f"{len(df):,}")
    with c2:
        render_metric("Unique Users", f"{df['UserId'].nunique():,}")
    with c3:
        render_metric("Unique Attractions", f"{df['AttractionId'].nunique():,}")
    with c4:
        render_metric("Average Rating", f"{df['Rating'].mean():.2f}")

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Visit Mode Split")
        mode_counts = df["VisitMode"].value_counts().reset_index()
        mode_counts.columns = ["VisitMode", "Count"]
        if px:
            fig = px.bar(mode_counts, x="VisitMode", y="Count", color="VisitMode", color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(mode_counts)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Top 10 Source Countries")
        top_country = df["Country"].value_counts().head(10).reset_index()
        top_country.columns = ["Country", "Transactions"]
        if px:
            fig = px.bar(top_country, x="Transactions", y="Country", orientation="h", color="Transactions", color_continuous_scale="Tealgrn")
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(top_country)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Highest Traffic Attractions")
    top_attr = df["Attraction"].value_counts().head(12).reset_index()
    top_attr.columns = ["Attraction", "Visits"]
    st.dataframe(top_attr, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_prediction(df: pd.DataFrame, reg_model, clf_model, feature_map: dict):
    st.markdown("<div class='hero'><h2 style='margin:0;'>Prediction Console</h2><p>Use historical and location features to estimate rating and visitor segment.</p></div>", unsafe_allow_html=True)
    mode_map = mode_lookup_map(df)

    left, right = st.columns(2)

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Rating Prediction")
        reg_features = feature_map["regression"]
        reg_input = {}
        for col in reg_features:
            reg_input[col] = st.number_input(
                f"{col}",
                min_value=int(df[col].min()),
                max_value=int(df[col].max()),
                value=int(df[col].median()),
                key=f"reg_{col}",
            )
        if st.button("Predict Rating", key="btn_rating"):
            pred = reg_model.predict(pd.DataFrame([reg_input])[reg_features])[0]
            st.success(f"Predicted attraction rating: {pred:.2f} / 5")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Visit Mode Classification")
        clf_features = feature_map["classification"]
        clf_input = {}
        for col in clf_features:
            clf_input[col] = st.number_input(
                f"{col}",
                min_value=int(df[col].min()),
                max_value=int(df[col].max()),
                value=int(df[col].median()),
                key=f"clf_{col}",
            )
        if st.button("Predict Visit Mode", key="btn_mode"):
            pred = int(clf_model.predict(pd.DataFrame([clf_input])[clf_features])[0])
            st.success(f"Predicted segment: {mode_map.get(pred, str(pred))} (ID {pred})")
        st.markdown("</div>", unsafe_allow_html=True)


def page_recommendations(df: pd.DataFrame, knn, user_item: pd.DataFrame):
    st.markdown("<div class='hero'><h2 style='margin:0;'>Recommendation Studio</h2><p>Collaborative filtering suggests attractions from similar user behavior.</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        user_id = st.number_input(
            "Select User ID",
            min_value=int(df["UserId"].min()),
            max_value=int(df["UserId"].max()),
            value=int(df["UserId"].iloc[0]),
        )
        n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
        generate = st.button("Generate Recommendations")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        if generate:
            recs = recommend_for_user(df, knn, user_item, int(user_id), k=int(n))
            st.subheader("Recommended Attractions")
            for i, item in enumerate(recs, start=1):
                st.write(f"{i}. {item}")
        else:
            st.info("Select a user and click Generate Recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)


def page_model_performance(metrics: dict):
    st.markdown("<div class='hero'><h2 style='margin:0;'>Model Evaluation</h2><p>Comparison of regression, classification, and recommendation performance.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Regression")
    reg_df = pd.DataFrame(metrics["regression"]["all_models"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(reg_df, use_container_width=True, hide_index=True)
    st.caption(f"Best model: {metrics['regression']['best_model']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Classification")
    clf_df = pd.DataFrame(metrics["classification"]["all_models"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(clf_df, use_container_width=True, hide_index=True)
    st.caption(f"Best model: {metrics['classification']['best_model']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Recommendation")
    rec = metrics["recommendation"]
    st.write(f"Users in user-item matrix: {rec['users_in_matrix']:,}")
    st.write(f"Attractions in matrix: {rec['items_in_matrix']:,}")
    st.write(f"HitRate@5: {rec['hitrate_at_5']:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    df = load_data()
    metrics = load_metrics()
    reg_model, clf_model, knn, user_item, feature_map = load_artifacts()

    st.sidebar.title("Tourism Analytics")
    st.sidebar.caption("Assignment Demo Application")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Predict & Segment", "Recommend", "Model Performance"],
    )

    if page == "Overview":
        page_overview(df)
    elif page == "Predict & Segment":
        page_prediction(df, reg_model, clf_model, feature_map)
    elif page == "Recommend":
        page_recommendations(df, knn, user_item)
    else:
        page_model_performance(metrics)


if __name__ == "__main__":
    main()
