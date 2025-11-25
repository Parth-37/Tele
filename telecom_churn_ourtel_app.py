"""
Telecom Churn – OurTel vs Competitors
End-to-End Script

Usage:
1) Run pipeline (EDA + model + scoring):
   python telecom_churn_ourtel_app.py

2) Run dashboard:
   streamlit run telecom_churn_ourtel_app.py
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
DATA_CSV = "telecom_churn_dataset_current_operator.csv"
OURTEL_SCORED_CSV = "telecom_churn_with_scores_ourtel.csv"
BEST_MODEL_PKL = "best_churn_model.pkl"

OUR_OPERATOR_NAME = "OurTel"
RANDOM_SEED = 42


# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_raw_data(path: str = DATA_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Place telecom_churn_dataset_current_operator.csv in this folder."
        )
    df = pd.read_csv(path)
    return df


# -----------------------------
# 2. EDA
# -----------------------------
def run_eda(df: pd.DataFrame) -> None:
    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)
    print("\nHead:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe())

    print("\n===== OVERALL CHURN =====")
    churn_rate = df["churn"].mean()
    print(f"Overall churn rate: {churn_rate:.2%}")

    print("\n===== CHURN BY REGION =====")
    print(
        df.groupby("region")["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n===== CHURN BY CURRENT OPERATOR =====")
    print(
        df.groupby("current_operator")["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n===== CHURN BY PLAN TYPE =====")
    print(
        df.groupby("plan_type")["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n===== CHURN BY PLAN CATEGORY =====")
    print(
        df.groupby("plan_category")["churn"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n===== REGION × OPERATOR CHURN (BENCHMARK) =====")
    region_operator_churn = (
        df.groupby(["region", "current_operator"])["churn"]
        .mean()
        .reset_index()
        .sort_values(["region", "churn"], ascending=[True, False])
    )
    print(region_operator_churn)

    # Numeric columns for plots and correlation
    numeric_cols = [
        "age",
        "tenure_months",
        "monthly_charge",
        "call_drops",
        "network_issues",
        "data_speed_rating",
        "customer_service_calls",
        "ticket_count_last_6_months",
        "issue_resolution_time_avg",
        "customer_support_rating",
        "validity_days",
        "dissatisfaction_score",
        "total_charge",
    ]

    print("\n===== CORRELATION WITH CHURN =====")
    corr = df[numeric_cols + ["churn"]].corr()
    print(corr["churn"].sort_values(ascending=False))

    # Histograms
    for col in [
        "tenure_months",
        "monthly_charge",
        "call_drops",
        "network_issues",
        "dissatisfaction_score",
    ]:
        plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Correlation heatmap (numeric features + churn)")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3. MODEL TRAINING (OurTel only)
# -----------------------------
def train_and_compare_models(df: pd.DataFrame):
    # Filter to our operator
    df_our = df[df["current_operator"] == OUR_OPERATOR_NAME].copy()
    print(f"\nOurTel rows: {len(df_our)}")

    target = "churn"
    drop_cols = ["customer_id", "churn", "current_operator"]

    X = df_our.drop(columns=drop_cols)
    y = df_our[target]

    numeric_features = [
        "age",
        "tenure_months",
        "monthly_charge",
        "call_drops",
        "network_issues",
        "data_speed_rating",
        "customer_service_calls",
        "ticket_count_last_6_months",
        "issue_resolution_time_avg",
        "customer_support_rating",
        "validity_days",
        "dissatisfaction_score",
        "total_charge",
    ]

    categorical_features = [
        "region",
        "plan_type",
        "plan_category",
        "addons_subscribed",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=RANDOM_SEED,
        ),
    }

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
    except ImportError:
        print("\n[xgboost] not installed; skipping XGBoost model.")

    results = []
    best_model = None
    best_name = None
    best_roc_auc = -1.0

    for model_name, model in models.items():
        print("\n==============================")
        print(f"TRAINING MODEL: {model_name}")
        print("==============================")

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        results.append(
            [
                model_name,
                accuracy,
                precision,
                recall,
                f1,
                roc,
            ]
        )

        if roc > best_roc_auc:
            best_roc_auc = roc
            best_model = clf
            best_name = model_name

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ],
    )

    print("\n\n===== MODEL COMPARISON TABLE (OurTel) =====")
    print(results_df.sort_values("ROC-AUC", ascending=False))
    print(f"\nBest model (by ROC-AUC): {best_name}")

    # Save best model
    joblib.dump(best_model, BEST_MODEL_PKL)
    print(f"\nSaved best model pipeline as {BEST_MODEL_PKL}")

    return best_model, results_df


# -----------------------------
# 4. SCORING & RISK SEGMENTATION (OurTel)
# -----------------------------
def risk_segment(prob: float) -> str:
    if prob < 0.30:
        return "Low risk"
    elif prob <= 0.70:
        return "Grey area"
    else:
        return "High risk"


def score_and_segment_customers(df: pd.DataFrame, best_model) -> pd.DataFrame:
    df_our = df[df["current_operator"] == OUR_OPERATOR_NAME].copy()
    print(f"\nScoring OurTel customers: {len(df_our)}")

    X = df_our.drop(columns=["customer_id", "churn", "current_operator"])
    df_our["churn_probability"] = best_model.predict_proba(X)[:, 1]
    df_our["risk_segment"] = df_our["churn_probability"].apply(risk_segment)

    print("\n===== OurTel RISK SEGMENT COUNTS =====")
    print(df_our["risk_segment"].value_counts())
    print("\n===== OurTel RISK SEGMENT PROPORTIONS =====")
    print(df_our["risk_segment"].value_counts(normalize=True))

    print("\n===== OurTel CHURN RATE BY RISK SEGMENT =====")
    print(df_our.groupby("risk_segment")["churn"].mean())

    return df_our


# -----------------------------
# 5. RETENTION SUGGESTIONS (OurTel)
# -----------------------------
def retention_action(row) -> str:
    actions = []

    # 1. Network / quality issues
    if row["call_drops"] > 3 or row["network_issues"] > 5:
        actions.append(
            "Trigger network optimization and offer free data/top-up."
        )

    # 2. High dissatisfaction
    if row["dissatisfaction_score"] >= 8:
        actions.append(
            "Proactive support call + goodwill bill credit to address dissatisfaction."
        )

    # 3. High-value customer
    if row["monthly_charge"] > 900:
        actions.append(
            "Mark as high-value; offer limited-time discount or OTT bundle."
        )

    # 4. Long-tenure loyalty
    if row["tenure_months"] > 36:
        actions.append("Provide loyalty benefit (extra data or OTT subscription).")

    # 5. Region-based action (example)
    if row["region"] in ["North", "Metro"]:
        actions.append("Include in regional premium campaign for OurTel.")

    # 6. Risk-segment specific
    if row["risk_segment"] == "High risk":
        actions.append(
            "Assign relationship manager / senior agent for immediate retention intervention."
        )
    elif row["risk_segment"] == "Grey area":
        actions.append(
            "Send personalized SMS/App offer to nudge retention (mild discount or extra data)."
        )

    if not actions:
        actions.append(
            "No urgent action needed; include in generic engagement communication."
        )

    return " | ".join(actions)


def add_retention_suggestions(df_our_scored: pd.DataFrame) -> pd.DataFrame:
    df_our_scored["retention_actions"] = df_our_scored.apply(
        retention_action, axis=1
    )
    return df_our_scored


# -----------------------------
# 6. MAIN PIPELINE
# -----------------------------
def run_pipeline():
    print("\n=== STEP 1: LOAD DATA ===")
    df = load_raw_data(DATA_CSV)

    print("\n=== STEP 2: EDA ===")
    run_eda(df)

    print("\n=== STEP 3: TRAIN & COMPARE MODELS (OurTel only) ===")
    best_model, _ = train_and_compare_models(df)

    print("\n=== STEP 4: SCORE & SEGMENT OURTEL CUSTOMERS ===")
    df_our_scored = score_and_segment_customers(df, best_model)

    print("\n=== STEP 5: ADD RETENTION SUGGESTIONS ===")
    df_our_scored = add_retention_suggestions(df_our_scored)

    df_our_scored.to_csv(OURTEL_SCORED_CSV, index=False)
    print(f"\nSaved OurTel scored data to {OURTEL_SCORED_CSV}")
    print("\nPipeline complete.")


# -----------------------------
# 7. STREAMLIT DASHBOARD
# -----------------------------
if "streamlit" in sys.modules:
    import streamlit as st

    st.set_page_config(
        page_title="OurTel Churn Dashboard",
        layout="wide",
    )

    @st.cache_data
    def load_full_data():
        return load_raw_data(DATA_CSV)

    @st.cache_data
    def load_ourtel_scored():
        if not os.path.exists(OURTEL_SCORED_CSV):
            st.error(
                f"{OURTEL_SCORED_CSV} not found. "
                "Run `python telecom_churn_ourtel_app.py` first to generate it."
            )
            st.stop()
        return pd.read_csv(OURTEL_SCORED_CSV)

    @st.cache_resource
    def load_best_model():
        if not os.path.exists(BEST_MODEL_PKL):
            st.error(
                f"{BEST_MODEL_PKL} not found. "
                "Run `python telecom_churn_ourtel_app.py` first to train and save the model."
            )
            st.stop()
        return joblib.load(BEST_MODEL_PKL)

    df_full = load_full_data()
    df_ourtel = load_ourtel_scored()
    best_model = load_best_model()

    # KPIs for OurTel
    our_mask = df_full["current_operator"] == OUR_OPERATOR_NAME
    df_our_raw = df_full[our_mask].copy()

    our_total = len(df_our_raw)
    our_churn_rate = df_our_raw["churn"].mean()
    grey_share = (df_ourtel["risk_segment"] == "Grey area").mean()
    high_share = (df_ourtel["risk_segment"] == "High risk").mean()

    st.title("📊 OurTel Churn & Risk Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OurTel Customers", f"{our_total:,}")
    c2.metric("OurTel Churn Rate", f"{our_churn_rate:.1%}")
    c3.metric("Grey Area (OurTel)", f"{grey_share:.1%}")
    c4.metric("High Risk (OurTel)", f"{high_share:.1%}")

    st.markdown("---")

    # Sidebar filters
    st.sidebar.header("Filters")

    region_options = sorted(df_full["region"].unique())
    region_filter = st.sidebar.multiselect(
        "Region(s)",
        options=region_options,
        default=region_options,
    )

    risk_options = sorted(df_ourtel["risk_segment"].unique())
    risk_filter = st.sidebar.multiselect(
        "Risk segment(s) – OurTel",
        options=risk_options,
        default=risk_options,
    )

    plan_type_options = sorted(df_ourtel["plan_type"].unique())
    plan_type_filter = st.sidebar.multiselect(
        "Plan type (OurTel)",
        options=plan_type_options,
        default=plan_type_options,
    )

    # Filtered OurTel scored data
    df_our_filtered = df_ourtel[
        df_ourtel["region"].isin(region_filter)
        & df_ourtel["risk_segment"].isin(risk_filter)
        & df_ourtel["plan_type"].isin(plan_type_filter)
    ]

    st.subheader("Filtered OurTel Segment Overview")
    st.write(f"Customers in filter: **{len(df_our_filtered):,}**")
    st.write(
        f"Churn rate in filter: **{df_our_filtered['churn'].mean():.1%}**"
        if len(df_our_filtered) > 0
        else "Churn rate in filter: N/A"
    )

    tab_overview, tab_ourtel, tab_benchmark = st.tabs(
        ["📈 Overview (All Operators)", "🧍 OurTel Risk & Actions", "🏁 Region–Operator Benchmark"]
    )

    # ----- TAB 1: Overview (All operators) -----
    with tab_overview:
        st.markdown("### Churn Rate by Current Operator (All Regions)")

        churn_by_op = (
            df_full.groupby("current_operator")["churn"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        st.bar_chart(churn_by_op.set_index("current_operator"))

        st.markdown("### Churn Rate by Region (All Operators)")
        churn_by_region = (
            df_full.groupby("region")["churn"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        st.bar_chart(churn_by_region.set_index("region"))

    # ----- TAB 2: OurTel Risk & Actions -----
    with tab_ourtel:
        st.markdown("### Churn Probability Distribution – OurTel (Filtered)")
        if len(df_our_filtered) > 0:
            fig, ax = plt.subplots()
            df_our_filtered["churn_probability"].hist(bins=30, ax=ax)
            ax.set_xlabel("Churn probability")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.write("No customers in current filter.")

        st.markdown("### OurTel Customers – Grey & High Risk (Top 100 by probability)")
        top_risky = (
            df_our_filtered[
                df_our_filtered["risk_segment"].isin(["Grey area", "High risk"])
            ]
            .sort_values("churn_probability", ascending=False)
            .head(100)[
                [
                    "customer_id",
                    "region",
                    "plan_type",
                    "plan_category",
                    "monthly_charge",
                    "dissatisfaction_score",
                    "churn_probability",
                    "risk_segment",
                    "retention_actions",
                ]
            ]
        )
        st.dataframe(top_risky)

    # ----- TAB 3: Region–Operator Benchmark -----
    with tab_benchmark:
        st.markdown(
            "### Churn Rate by Region × Operator\n"
            "_Compare OurTel vs competitors in each region._"
        )

        df_bench = df_full[df_full["region"].isin(region_filter)].copy()
        bench = (
            df_bench.groupby(["region", "current_operator"])["churn"]
            .mean()
            .reset_index()
            .sort_values(["region", "churn"], ascending=[True, False])
        )
        st.dataframe(bench)

        st.markdown(
            "> Use this table to show the client where OurTel is doing better or worse than competitors in each region."
        )


# -----------------------------
# 8. ENTRY POINT FOR PIPELINE
# -----------------------------
if __name__ == "__main__" and "streamlit" not in sys.modules:
    run_pipeline()
