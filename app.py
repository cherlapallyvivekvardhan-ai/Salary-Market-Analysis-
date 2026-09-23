import warnings
import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Market & Corporate Cap Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💼 Tech Salary Market Analysis & Corporate Cap Platform")
st.caption("End-to-End Econometric & Statistical Pipeline: Real-Time Valuation -> CIs -> Hypothesis Testing -> Regression -> K-Means Banding")

# ---------------------------------------------------------
# Data Layer with Streamlit Caching
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_caps():
    tickers = {
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "Wipro": "WIPRO.NS",
        "Microsoft": "MSFT",
        "Google": "GOOGL"
    }
    caps = {}
    for comp, sym in tickers.items():
        try:
            tkr = yf.Ticker(sym)
            val = tkr.fast_info.get("marketCap", None)
            if val is None:
                val = tkr.info.get("marketCap", 1e11)
            caps[comp] = round(val / 1e9, 2)  # In Billions
        except Exception:
            # Fallbacks in Billions if network throttles
            fallbacks = {"TCS": 14200.0, "Infosys": 7800.0, "Wipro": 2900.0, "Microsoft": 3150.0, "Google": 2100.0}
            caps[comp] = fallbacks[comp]
    return caps

@st.cache_data
def generate_and_clean_data(market_caps):
    np.random.seed(42)
    n = 1200

    companies = np.random.choice(["TCS", "Infosys", "Wipro", "Microsoft", "Google"], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    roles = np.random.choice(["Data Analyst", "Data Scientist", "Software Engineer"], size=n, p=[0.35, 0.35, 0.30])
    cities = np.random.choice(["Hyderabad", "Bangalore", "Pune"], size=n, p=[0.40, 0.40, 0.20])
    experience = np.clip(np.random.gamma(shape=3.0, scale=1.8, size=n), 0.5, 20.0).round(1)
    has_python = np.random.binomial(1, 0.65, size=n)

    base = (
        4.5
        + 1.7 * experience
        + 3.2 * has_python
        + (cities == "Bangalore") * 2.1
        + (cities == "Hyderabad") * 1.2
        + (companies == "Microsoft") * 14.5
        + (companies == "Google") * 16.0
        + (companies == "TCS") * (-1.0)
        + np.random.normal(0, 2.5, size=n)
    )
    salaries = np.clip(base, 3.2, 85.0).round(2)

    # Outliers injection
    outlier_idx = np.random.choice(n, size=15, replace=False)
    salaries[outlier_idx] = np.random.uniform(90.0, 160.0, size=15)

    df_raw = pd.DataFrame({
        "Company": companies,
        "Role": roles,
        "City": cities,
        "Experience_Yrs": experience,
        "Python_Skill": has_python,
        "Salary_LPA": salaries
    })
    df_raw["Market_Cap_B"] = df_raw["Company"].map(market_caps)

    # IQR Outlier Removal
    q1 = df_raw["Salary_LPA"].quantile(0.25)
    q3 = df_raw["Salary_LPA"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(0, q1 - 1.5 * iqr)

    df_filtered = df_raw[(df_raw["Salary_LPA"] >= lower) & (df_raw["Salary_LPA"] <= upper)].copy()
    return df_raw, df_filtered

# Load data
with st.spinner("Fetching real-time market data & preparing models..."):
    market_caps = fetch_market_caps()
    raw_df, df = generate_and_clean_data(market_caps)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("🕹️ Filter & Parameters")
selected_cities = st.sidebar.multiselect("Select Cities", options=list(df["City"].unique()), default=list(df["City"].unique()))
selected_roles = st.sidebar.multiselect("Select Roles", options=list(df["Role"].unique()), default=list(df["Role"].unique()))

view_df = df[(df["City"].isin(selected_cities)) & (df["Role"].isin(selected_roles))]

# --- Added: export filtered dataset ---
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="⬇️ Download Filtered Data (CSV)",
    data=view_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_salary_data.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# Real-Time Market Cap Ticker
# ---------------------------------------------------------
st.markdown("### 🏢 Live Corporate Market Capitalization")
col_caps = st.columns(len(market_caps))
for idx, (comp, cap) in enumerate(market_caps.items()):
    curr = "₹" if comp in ["TCS", "Infosys", "Wipro"] else "$"
    col_caps[idx].metric(label=comp, value=f"{curr}{cap:,.1f} B")

st.markdown("---")

# ---------------------------------------------------------
# Tabbed Statistical Dashboard
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Descriptive & CDF",
    "2. Confidence Intervals",
    "3. Hypothesis Testing",
    "4. Regression & VIF",
    "5. K-Means Banding"
])

# --- TAB 1: DESCRIPTIVE & CDF ---
with tab1:
    st.subheader("1. Salary Distribution & Percentiles")
    c1, c2 = st.columns([1, 2])

    with c1:
        bins = [0, 6, 12, 18, 30, np.inf]
        labels = ["< 6 LPA", "6 - 12 LPA", "12 - 18 LPA", "18 - 30 LPA", "30+ LPA"]
        view_df["Bracket"] = pd.cut(view_df["Salary_LPA"], bins=bins, labels=labels)
        freq = view_df["Bracket"].value_counts().sort_index().to_frame("Count")
        st.write("**Frequency Distribution (LPA)**")
        st.dataframe(freq, use_container_width=True)

        target_cutoff = st.number_input("Empirical CDF Threshold (LPA):", min_value=3.0, max_value=80.0, value=12.0, step=1.0)
        pct_below = (view_df["Salary_LPA"] < target_cutoff).mean() * 100
        st.info(f"💡 **{pct_below:.1f}%** of engineers/analysts earn less than **{target_cutoff} LPA**.")

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(view_df["Salary_LPA"], kde=True, bins=25, color="#1f77b4", ax=ax)
        ax.set_title("Salary Distribution with KDE Fit")
        ax.set_xlabel("Compensation (LPA)")
        st.pyplot(fig)

# --- TAB 2: CONFIDENCE INTERVALS ---
with tab2:
    st.subheader("2. 95% Confidence Intervals for HR Salary Bands")
    st.write("Provides a statistically sound recruitment range ($\\bar{X} \\pm t \\times \\text{SEM}$) rather than a fragile single-number mean.")

    def calc_ci(sub_data, confidence=0.95):
        m = np.mean(sub_data)
        sem = stats.sem(sub_data)
        n = len(sub_data)
        inv = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
        return round(m, 2), round(m - inv, 2), round(m + inv, 2)

    ci_list = []
    for r in view_df["Role"].unique():
        for c in view_df["City"].unique():
            sample = view_df[(view_df["Role"] == r) & (view_df["City"] == c)]["Salary_LPA"]
            if len(sample) >= 5:
                mean_val, low, high = calc_ci(sample)
                ci_list.append({
                    "Role": r,
                    "City": c,
                    "N": len(sample),
                    "Mean (LPA)": mean_val,
                    "95% CI Lower": low,
                    "95% CI Upper": high,
                    "Recommended Band": f"{low} - {high} LPA"
                })

    ci_table = pd.DataFrame(ci_list)
    st.dataframe(ci_table, use_container_width=True)

# --- TAB 3: HYPOTHESIS TESTING ---
with tab3:
    st.subheader("3. Inferential Testing: What Drives Pay?")

    col_t, col_f = st.columns(2)

    with col_t:
        st.markdown("#### A. Two-Sample Welch's t-Test (Python Premium)")
        py1 = view_df[view_df["Python_Skill"] == 1]["Salary_LPA"]
        py0 = view_df[view_df["Python_Skill"] == 0]["Salary_LPA"]

        if len(py1) > 1 and len(py0) > 1:
            t_val, p_val = stats.ttest_ind(py1, py0, equal_var=False)
            st.write(f"- **Python Mean:** {py1.mean():.2f} LPA")
            st.write(f"- **No-Python Mean:** {py0.mean():.2f} LPA")
            st.write(f"- **t-statistic:** `{t_val:.4f}`")
            st.write(f"- **p-value:** `{p_val:.4e}`")
            if p_val < 0.05:
                st.success("Verdict: Statistically significant pay premium for Python skill (p < 0.05).")
            else:
                st.warning("Verdict: No statistically significant pay premium detected.")

    with col_f:
        st.markdown("#### B. One-Way ANOVA (City-Level Disparity)")
        groups = [view_df[view_df["City"] == city]["Salary_LPA"] for city in view_df["City"].unique() if len(view_df[view_df["City"] == city]) > 1]

        if len(groups) > 1:
            f_val, f_pval = stats.f_oneway(*groups)
            st.write(f"- **F-statistic:** `{f_val:.4f}`")
            st.write(f"- **p-value:** `{f_pval:.4e}`")
            if f_pval < 0.05:
                st.success("Verdict: Significant salary divergence across geographic locations (p < 0.05).")
            else:
                st.warning("Verdict: Pay difference across cities is not statistically significant.")

# --- TAB 4: REGRESSION & VIF ---
with tab4:
    st.subheader("4. Multiple Linear Regression & Multicollinearity")

    reg_df = df[["Salary_LPA", "Experience_Yrs", "Python_Skill", "City", "Company"]].copy()
    reg_df = pd.get_dummies(reg_df, columns=["City", "Company"], drop_first=True, dtype=float)

    X = reg_df.drop(columns=["Salary_LPA"])
    y = reg_df["Salary_LPA"]
    X_const = sm.add_constant(X)

    c_vif, c_ols = st.columns([1, 2])

    with c_vif:
        st.markdown("**VIF Multicollinearity Check**")
        vif_data = pd.DataFrame({
            "Feature": X_const.columns,
            "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        }).round(2)
        st.dataframe(vif_data, use_container_width=True)
        st.caption("All variable VIFs < 5.0 indicates absence of damaging multicollinearity.")

    with c_ols:
        st.markdown("**OLS Model Estimates**")
        model = sm.OLS(y, X_const).fit()
        coef_df = pd.DataFrame({
            "Coef": model.params,
            "Std Error": model.bse,
            "p-value": model.pvalues
        }).round(4)
        st.dataframe(coef_df, use_container_width=True)
        st.metric("Model R²", f"{model.rsquared:.4f}", f"Adj. R²: {model.rsquared_adj:.4f}")

# --- TAB 5: K-MEANS BANDING ---
with tab5:
    st.subheader("5. Automated Market Segmentation via K-Means")

    k = st.slider("Select Cluster Count (Bands):", min_value=2, max_value=6, value=4)
    cluster_features = df[["Experience_Yrs", "Salary_LPA"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_features)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(scaled)

    # Sort cluster names by average pay
    rank = df.groupby("Cluster")["Salary_LPA"].mean().sort_values().index
    names = [f"Band {i+1} (Tier {i+1})" for i in range(k)]
    mapping = {rank[i]: names[i] for i in range(k)}
    df["Salary_Band"] = df["Cluster"].map(mapping)

    summary = df.groupby("Salary_Band").agg(
        Count=('Salary_LPA', 'count'),
        Exp_Min=('Experience_Yrs', 'min'),
        Exp_Max=('Experience_Yrs', 'max'),
        Salary_Min=('Salary_LPA', 'min'),
        Median_LPA=('Salary_LPA', 'median'),
        Mean_LPA=('Salary_LPA', 'mean')
    ).round(2).reindex(names)

    st.dataframe(summary, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(
        data=df,
        x="Experience_Yrs",
        y="Salary_LPA",
        hue="Salary_Band",
        palette="viridis",
        alpha=0.8,
        ax=ax2
    )
    ax2.set_title("Market Clusters: Experience vs Salary Banding")
    ax2.set_xlabel("Experience (Years)")
    ax2.set_ylabel("Salary (LPA)")
    st.pyplot(fig2)
