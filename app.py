"""
=================================================================================
SALARY MARKET INTELLIGENCE ENGINE  (Streamlit app)
=================================================================================
What was fixed
  * SyntaxError: statsmodels formulas (patsy) cannot parse column names that
    contain spaces ("skill_Machine Learning"). The models now use the matrix API
    (sm.OLS(y, X)), so no formula string is ever parsed.
  * print() output never reaches a Streamlit page -> everything is rendered
    with st.* widgets, tables and charts.
  * Cook's distance is now computed on the FULL log-salary model (before, tier
    and region were missing, so legit high earners were flagged as outliers).
  * KS test with fitted parameters gives biased p-values -> Lilliefors test on
    log-salary is used instead.
  * Logistic regression now uses scaled features + a hold-out AUC.

What was added (real-life analysis)
  * Upload your own CSV (or use the synthetic simulator)
  * Raw vs. regression-ADJUSTED skill premium (the t-test alone is confounded
    by experience), effect sizes, Mann-Whitney, Tukey HSD, Kruskal-Wallis
  * Salary range (P10-P90 prediction interval), career trajectory curve,
    tier x region arbitrage heatmap
  * Skill ROI: uplift per skill, per week of effort, and a greedy learning path
  * K-Means with silhouette scan to choose k, gap skills vs. next band
  * Offer evaluator: percentile of an offer, negotiation ask, hike %, and an
    approximate in-hand salary (India new tax regime)
  * Model health: hold-out R2 / MAE / MAPE, 5-fold CV, interval coverage,
    VIF, Breusch-Pagan, robust (HC3) coefficients
  * Downloadable report + cleaned dataset
=================================================================================
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Salary Market Intelligence", page_icon="💼", layout="wide")

# ===============================================================================
# CONFIG
# ===============================================================================
SKILL_CATALOG = [
    "Python", "SQL", "Docker", "Kubernetes", "AWS",
    "Machine Learning", "System Design", "Distributed Systems", "GenAI",
]
HIGH_VALUE_SKILLS = ["GenAI", "Distributed Systems", "System Design"]
REGIONS = ["Bengaluru", "Hyderabad", "Pune", "Delhi-NCR"]
COMPANY_TIERS = ["Startup", "Global MNC", "Tier-1 Product"]

# Rough, EDITABLE assumptions for "weeks of part-time effort to become job-ready"
DEFAULT_WEEKS = {
    "Python": 8, "SQL": 4, "Docker": 3, "Kubernetes": 8, "AWS": 8,
    "Machine Learning": 24, "System Design": 16, "Distributed Systems": 20, "GenAI": 16,
}


def skill_col(name: str) -> str:
    """Spaces are fine now (no formulas), but keep names tidy anyway."""
    return "skill_" + name.strip().replace(" ", "_")


def sk_label(col: str) -> str:
    return col[len("skill_"):].replace("_", " ")


def pretty(c: str) -> str:
    if c == "const":
        return "Intercept"
    if c == "experience":
        return "Experience (per year)"
    if c == "age":
        return "Age"
    if c.startswith("skill_"):
        return "Skill: " + sk_label(c)
    if c.startswith("company_tier_"):
        return "Tier: " + c[len("company_tier_"):]
    if c.startswith("region_"):
        return "Region: " + c[len("region_"):]
    return c


# ---- small UI helpers (work across Streamlit versions) -------------------------
def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df: pd.DataFrame):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ===============================================================================
# DATA: SYNTHETIC + UPLOAD + VALIDATION
# ===============================================================================
@st.cache_data(show_spinner=False)
def generate_mock_salary_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    experience = np.clip(rng.gamma(shape=3.0, scale=2.0, size=n_samples), 0.5, 25.0)
    age = 22 + experience + rng.normal(0, 1.2, size=n_samples)  # collinear with experience

    tier = rng.choice(COMPANY_TIERS, size=n_samples, p=[0.35, 0.40, 0.25])
    region = rng.choice(REGIONS, size=n_samples, p=[0.40, 0.25, 0.20, 0.15])

    skill_data = {}
    for s in SKILL_CATALOG:
        prob = 0.55 if s in ("Python", "SQL") else (0.25 if s in HIGH_VALUE_SKILLS else 0.40)
        skill_data[skill_col(s)] = rng.binomial(1, prob, size=n_samples)
    skills_df = pd.DataFrame(skill_data)

    premium_cnt = skills_df[[skill_col(s) for s in HIGH_VALUE_SKILLS]].sum(axis=1).to_numpy()
    all_cnt = skills_df.sum(axis=1).to_numpy()

    tier_mult = {"Startup": 0.05, "Global MNC": 0.22, "Tier-1 Product": 0.58}
    region_mult = {"Bengaluru": 0.15, "Hyderabad": 0.08, "Pune": 0.02, "Delhi-NCR": 0.05}

    mu = (
        2.2 + 0.085 * experience + 0.04 * all_cnt + 0.12 * premium_cnt
        + pd.Series(tier).map(tier_mult).to_numpy()
        + pd.Series(region).map(region_mult).to_numpy()
    )
    salary = rng.lognormal(mean=mu, sigma=0.28)

    df = pd.DataFrame({
        "experience": experience, "age": age,
        "company_tier": tier, "region": region,
        "salary_lpa": np.round(salary, 2),
    })
    return pd.concat([df, skills_df], axis=1)


def parse_uploaded(file, min_skill_count: int = 20) -> pd.DataFrame:
    """CSV columns: experience, company_tier, region, salary_lpa, skills ("Python, SQL, AWS")."""
    raw = pd.read_csv(file)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    required = ["experience", "company_tier", "region", "salary_lpa", "skills"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    lists = raw["skills"].fillna("").astype(str).apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip()]
    )
    counts = pd.Series([x for l in lists for x in l]).value_counts()
    keep = sorted(counts[counts >= min_skill_count].index)
    if not keep:
        raise ValueError(f"No skill appears at least {min_skill_count} times.")

    out = raw[["experience", "company_tier", "region", "salary_lpa"]].copy()
    if "age" in raw.columns:
        out["age"] = raw["age"]
    for s in keep:
        out[skill_col(s)] = lists.apply(lambda l, s=s: int(s in l))
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["experience", "salary_lpa"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["experience", "salary_lpa", "company_tier", "region"])
    df = df[(df["salary_lpa"] > 0) & (df["experience"] >= 0)].reset_index(drop=True)
    if len(df) < 200:
        raise ValueError(f"Need at least 200 valid rows, found {len(df)}.")
    return df


# ===============================================================================
# CORE STATS / MODELS  (matrix API only -> immune to column-name problems)
# ===============================================================================
def design_matrix(df: pd.DataFrame, skill_cols: list):
    enc = pd.get_dummies(
        df[["experience", *skill_cols, "company_tier", "region"]],
        columns=["company_tier", "region"], drop_first=True, dtype=float,
    ).astype(float)
    feature_cols = list(enc.columns)
    X = sm.add_constant(enc, has_constant="add")
    return X, feature_cols


def remove_outliers(df: pd.DataFrame, skill_cols: list):
    """Cook's distance on the full log-salary model, threshold 4/n."""
    X, _ = design_matrix(df, skill_cols)
    y = np.log(df["salary_lpa"].to_numpy())
    cooks = sm.OLS(y, X).fit().get_influence().cooks_distance[0]
    thr = 4.0 / len(df)
    flag = cooks > thr
    return df.loc[~flag].reset_index(drop=True), df.loc[flag].reset_index(drop=True), thr


def fit_ols(df: pd.DataFrame, skill_cols: list) -> dict:
    X, feature_cols = design_matrix(df, skill_cols)
    y = np.log(df["salary_lpa"].to_numpy())
    ols = sm.OLS(y, X).fit()
    robust = sm.OLS(y, X).fit(cov_type="HC3")
    resid = np.asarray(ols.resid)
    return dict(
        X=X, y=y, feature_cols=feature_cols, ols=ols, robust=robust,
        smear=float(np.mean(np.exp(resid))),
        sigma=float(np.sqrt(ols.scale)),
    )


def user_row(profile: dict, feature_cols: list) -> dict:
    row = dict.fromkeys(feature_cols, 0.0)
    row["experience"] = float(profile["experience"])
    for s in profile["skills"]:
        c = skill_col(s)
        if c in row:
            row[c] = 1.0
    for c in (f"company_tier_{profile['tier']}", f"region_{profile['region']}"):
        if c in row:
            row[c] = 1.0
    return row


def frame_from(profiles: list, feature_cols: list) -> pd.DataFrame:
    return pd.DataFrame([user_row(p, feature_cols) for p in profiles])[feature_cols]


def predict_many(bundle: dict, profiles: list, alpha: float = 0.20) -> pd.DataFrame:
    """Median, mean (smearing-corrected) and (1-alpha) prediction interval, in LPA."""
    Xu = frame_from(profiles, bundle["feature_cols"])
    Xu.insert(0, "const", 1.0)
    sf = bundle["ols"].get_prediction(Xu).summary_frame(alpha=alpha)
    out = pd.DataFrame({
        "median": np.exp(sf["mean"].to_numpy()),
        "low": np.exp(sf["obs_ci_lower"].to_numpy()),
        "high": np.exp(sf["obs_ci_upper"].to_numpy()),
    })
    out["mean"] = out["median"] * bundle["smear"]
    return out


def point_median(bundle: dict, profile: dict) -> float:
    Xu = frame_from([profile], bundle["feature_cols"])
    vec = np.r_[1.0, Xu.iloc[0].to_numpy()]
    return float(np.exp(vec @ bundle["ols"].params.to_numpy()))


def coef_table(bundle: dict) -> pd.DataFrame:
    r = bundle["robust"]
    ci = np.asarray(r.conf_int())
    params = r.params.to_numpy()
    out = pd.DataFrame({
        "Feature": [pretty(c) for c in r.params.index],
        "Coef (log)": params,
        "Effect %": (np.exp(params) - 1) * 100,
        "CI low %": (np.exp(ci[:, 0]) - 1) * 100,
        "CI high %": (np.exp(ci[:, 1]) - 1) * 100,
        "p-value": r.pvalues.to_numpy(),
    })
    return out[out["Feature"] != "Intercept"].reset_index(drop=True)


def fit_logit(df: pd.DataFrame, bundle: dict, target_tier: str):
    Xf = bundle["X"].drop(columns="const")
    tier_median = float(df.loc[df["company_tier"] == target_tier, "salary_lpa"].median())
    y = (df["salary_lpa"].to_numpy() >= tier_median).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.25, random_state=42, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    clf.fit(Xf, y)
    return clf, tier_median, auc


def validate_ols(bundle: dict) -> dict:
    X, y, fc = bundle["X"], bundle["y"], bundle["feature_cols"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = sm.OLS(ytr, Xtr).fit()
    smear = float(np.mean(np.exp(np.asarray(m.resid))))
    pred = np.exp(np.asarray(m.predict(Xte))) * smear
    actual = np.exp(yte)
    sf = m.get_prediction(Xte).summary_frame(alpha=0.20)
    lo, hi = np.exp(sf["obs_ci_lower"].to_numpy()), np.exp(sf["obs_ci_upper"].to_numpy())
    cv = cross_val_score(LinearRegression(), X[fc], y, cv=5, scoring="r2")
    return dict(
        r2_log=float(r2_score(yte, np.log(pred / smear))),
        mae=float(mean_absolute_error(actual, pred)),
        mape=float(np.mean(np.abs(actual - pred) / actual) * 100),
        cv_r2=float(cv.mean()), cv_std=float(cv.std()),
        coverage=float(np.mean((actual >= lo) & (actual <= hi)) * 100),
    )


def vif_table(df: pd.DataFrame, skill_cols: list) -> pd.DataFrame:
    X, _ = design_matrix(df, skill_cols)
    X = X.copy()
    if "age" in df.columns and df["age"].notna().all():
        X["age"] = df["age"].to_numpy()
    rows = [
        {"Feature": pretty(c), "VIF": float(variance_inflation_factor(X.to_numpy(), i))}
        for i, c in enumerate(X.columns) if c != "const"
    ]
    out = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    out["Flag"] = np.where(out["VIF"] > 5, "⚠️ collinear", "ok")
    return out


def cohens_d(a, b) -> float:
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def effect_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


# ===============================================================================
# SEGMENTATION
# ===============================================================================
@st.cache_data(show_spinner=False)
def silhouette_scan(feat_df: pd.DataFrame, kmax: int = 7) -> pd.DataFrame:
    M = StandardScaler().fit_transform(feat_df)
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(M)
        rows.append({"k": k, "silhouette": float(silhouette_score(M, km.labels_)), "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)


def segment(df: pd.DataFrame, skill_cols: list, k: int, profile: dict, user_pred: float) -> dict:
    d = df.copy()
    d["log_salary"] = np.log(d["salary_lpa"])
    feats = ["experience", "log_salary", *skill_cols]
    scaler = StandardScaler()
    M = scaler.fit_transform(d[feats])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(M)
    d["cluster"] = km.labels_

    prof = (
        d.groupby("cluster")
        .agg(n=("salary_lpa", "size"), avg=("salary_lpa", "mean"), lo=("salary_lpa", "min"),
             hi=("salary_lpa", "max"), exp=("experience", "mean"))
        .reset_index().sort_values("avg").reset_index(drop=True)
    )
    prof["band"] = np.arange(1, k + 1)
    band_of = dict(zip(prof["cluster"], prof["band"]))
    d["band"] = d["cluster"].map(band_of)
    prevalence = d.groupby("band")[skill_cols].mean()

    urow = {"experience": profile["experience"], "log_salary": np.log(user_pred)}
    for c in skill_cols:
        urow[c] = 1.0 if sk_label(c) in profile["skills"] else 0.0
    cid = int(km.predict(scaler.transform(pd.DataFrame([urow])[feats]))[0])
    return dict(data=d, profiles=prof, prevalence=prevalence, user_band=int(band_of[cid]), k=k)


# ===============================================================================
# OFFER / TAX HELPERS  (approximation - India new regime, verify current rules)
# ===============================================================================
def new_regime_tax(taxable: float) -> float:
    slabs = [(400_000, 0.0), (800_000, 0.05), (1_200_000, 0.10), (1_600_000, 0.15),
             (2_000_000, 0.20), (2_400_000, 0.25), (float("inf"), 0.30)]
    tax, prev = 0.0, 0.0
    for limit, rate in slabs:
        if taxable > prev:
            tax += (min(taxable, limit) - prev) * rate
        prev = limit
    if taxable <= 1_200_000:          # 87A rebate
        tax = 0.0
    else:                             # marginal relief just above 12L
        tax = min(tax, taxable - 1_200_000)
    return tax * 1.04                 # 4% cess


def in_hand(ctc_lpa: float, basic_pct: float = 0.40, pf_pct: float = 0.12) -> dict:
    ctc = ctc_lpa * 1e5
    er_pf = ctc * basic_pct * pf_pct
    ee_pf = er_pf
    taxable = max(ctc - er_pf - 75_000, 0.0)          # employer PF not taxed, std deduction 75k
    tax = new_regime_tax(taxable)
    net = ctc - er_pf - ee_pf - tax
    return dict(monthly=net / 12, annual=net, tax=tax, pf=er_pf + ee_pf)


def inr(x: float) -> str:
    return f"₹{x:,.0f}"


# ===============================================================================
# APP
# ===============================================================================
def main():
    st.title("💼 Salary Market Intelligence Engine")
    st.caption("Distribution fitting · hypothesis tests · regression & classification · "
               "segmentation · skill ROI · offer evaluation")

    # ------------------------------ SIDEBAR: DATA ------------------------------
    st.sidebar.header("1 · Data")
    source = st.sidebar.radio("Dataset", ["Synthetic market simulator", "Upload my CSV"])
    if source == "Synthetic market simulator":
        n = st.sidebar.slider("Sample size", 500, 10000, 3000, 500)
        seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)
        raw = generate_mock_salary_data(int(n), int(seed))
    else:
        template = pd.DataFrame({
            "experience": [3.5, 8.0], "company_tier": ["Startup", "Tier-1 Product"],
            "region": ["Bengaluru", "Pune"], "salary_lpa": [14.5, 42.0],
            "skills": ["Python, SQL", "Python, AWS, System Design"],
        }).to_csv(index=False)
        st.sidebar.download_button("Download CSV template", template, "salary_template.csv", "text/csv")
        up = st.sidebar.file_uploader("Upload CSV", type="csv")
        if up is None:
            st.info("Upload a CSV (columns: experience, company_tier, region, salary_lpa, skills) "
                    "or switch to the synthetic simulator in the sidebar.")
            st.stop()
        try:
            raw = parse_uploaded(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")
            st.stop()

    try:
        raw = validate(raw)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

    skill_cols = [c for c in raw.columns if c.startswith("skill_")]
    skills = [sk_label(c) for c in skill_cols]
    tiers = sorted(raw["company_tier"].unique())
    regions = sorted(raw["region"].unique())

    filter_outliers = st.sidebar.checkbox("Remove outliers (Cook's distance, 4/n)", value=True)
    if filter_outliers:
        df, flagged, thr = remove_outliers(raw, skill_cols)
    else:
        df, flagged, thr = raw, raw.iloc[0:0], 4.0 / len(raw)

    # ------------------------------ SIDEBAR: PROFILE ---------------------------
    st.sidebar.header("2 · Your profile")
    exp = st.sidebar.slider("Years of experience", 0.5, 25.0, 4.5, 0.5)
    sel = st.sidebar.multiselect("Skills you have", skills,
                                 default=[s for s in ["Python", "SQL", "Docker"] if s in skills])
    tier = st.sidebar.selectbox("Target company tier", tiers,
                                index=tiers.index("Tier-1 Product") if "Tier-1 Product" in tiers else 0)
    region = st.sidebar.selectbox("Target region", regions,
                                  index=regions.index("Bengaluru") if "Bengaluru" in regions else 0)
    profile = dict(experience=exp, skills=sel, tier=tier, region=region)

    # ------------------------------ MODELS --------------------------------------
    bundle = fit_ols(df, skill_cols)
    pred = predict_many(bundle, [profile]).iloc[0]
    pred_median, pred_mean, pred_lo, pred_hi = pred["median"], pred["mean"], pred["low"], pred["high"]
    tier_sal = df.loc[df["company_tier"] == tier, "salary_lpa"]
    clf, tier_median, auc = fit_logit(df, bundle, tier)
    p_above = float(clf.predict_proba(frame_from([profile], bundle["feature_cols"]))[0, 1])
    pct_in_tier = float(stats.percentileofscore(tier_sal, pred_median))

    t_overview, t_tests, t_pred, t_skill, t_seg, t_offer, t_health = st.tabs([
        "📊 Market overview", "🧪 Hypothesis tests", "🎯 Your prediction",
        "🚀 Skill ROI & career path", "🧭 Segments", "🤝 Offer evaluator", "🔬 Model health",
    ])

    # =========================== TAB 1: OVERVIEW ================================
    with t_overview:
        s = df["salary_lpa"]
        c = st.columns(5)
        c[0].metric("Records used", f"{len(df):,}", f"-{len(flagged)} outliers" if len(flagged) else None)
        c[1].metric("Median", f"{s.median():.1f} LPA")
        c[2].metric("Mean", f"{s.mean():.1f} LPA")
        c[3].metric("P25 – P75", f"{s.quantile(.25):.0f} – {s.quantile(.75):.0f}")
        c[4].metric("P90", f"{s.quantile(.90):.1f} LPA")

        logs = np.log(s.to_numpy())
        mu, sigma = float(logs.mean()), float(logs.std(ddof=0))   # closed-form MLE for log-normal
        lil_stat, lil_p = lilliefors(logs, dist="norm")
        xs = np.linspace(s.min(), s.quantile(0.995), 300)
        fig = go.Figure()
        fig.add_histogram(x=s, nbinsx=60, histnorm="probability density", name="Observed", opacity=0.6)
        fig.add_scatter(x=xs, y=stats.lognorm.pdf(xs, s=sigma, scale=np.exp(mu)),
                        name=f"Log-normal MLE (μ={mu:.2f}, σ={sigma:.2f})", line=dict(width=3))
        fig.update_layout(title="Salary distribution & log-normal fit", xaxis_title="LPA", height=380)
        show_chart(fig)
        st.caption(f"Lilliefors normality test on log-salary: D = {lil_stat:.4f}, p = {lil_p:.3g} → "
                   + ("consistent with log-normal." if lil_p > 0.05 else
                      "deviates from a pure log-normal (mixtures of tiers/experience are normal in real data)."))

        st.subheader("95% confidence intervals by company tier")
        rows = []
        for t in tiers:
            x = df.loc[df["company_tier"] == t, "salary_lpa"]
            lo, hi = stats.t.interval(0.95, df=len(x) - 1, loc=x.mean(), scale=stats.sem(x))
            rows.append({"Tier": t, "n": len(x), "Median": x.median(), "Mean": x.mean(),
                         "CI low": lo, "CI high": hi, "P90": x.quantile(.9)})
        show_df(pd.DataFrame(rows).round(2))

        a, b = st.columns(2)
        with a:
            show_chart(px.box(df, x="company_tier", y="salary_lpa", color="region",
                              title="Salary by tier & region"))
        with b:
            samp = df.sample(min(len(df), 1500), random_state=1)
            show_chart(px.scatter(samp, x="experience", y="salary_lpa", color="company_tier",
                                  opacity=0.55, log_y=True, title="Experience vs salary (log axis)"))

        with st.expander(f"Outlier screening — {len(flagged)} rows flagged (Cook's D > {thr:.5f})"):
            st.write("Flags are computed on the full log-salary model (experience, skills, tier, region), "
                     "so a high salary that is *explained* by the model is not treated as an outlier.")
            if len(flagged):
                show_df(flagged.drop(columns=[c for c in ["age"] if c in flagged]).round(2).head(200))

    # =========================== TAB 2: TESTS ===================================
    with t_tests:
        st.subheader("Skill premium: raw vs. adjusted")
        skill = st.selectbox("Skill to test", skills, index=skills.index("GenAI") if "GenAI" in skills else 0)
        col = skill_col(skill)
        has, no = df.loc[df[col] == 1, "salary_lpa"], df.loc[df[col] == 0, "salary_lpa"]
        if len(has) < 5 or len(no) < 5:
            st.warning("Not enough rows on one side of this skill to test.")
        else:
            t_stat, t_p = stats.ttest_ind(has, no, equal_var=False)
            u_stat, u_p = stats.mannwhitneyu(has, no, alternative="two-sided")
            d_log = cohens_d(np.log(has), np.log(no))
            r = bundle["robust"]
            coef = float(r.params[col])
            ci = np.asarray(r.conf_int())[list(r.params.index).index(col)]
            c = st.columns(4)
            c[0].metric("Raw gap (Welch)", f"{has.mean() - no.mean():+.2f} LPA", f"p = {t_p:.2g}")
            c[1].metric("Mann-Whitney p", f"{u_p:.2g}")
            c[2].metric("Cohen's d (log)", f"{d_log:.2f}", effect_label(d_log))
            c[3].metric("Adjusted premium", f"{(np.exp(coef) - 1) * 100:+.1f}%",
                        f"CI {(np.exp(ci[0]) - 1) * 100:.1f}% to {(np.exp(ci[1]) - 1) * 100:.1f}%")
            st.caption(f"With {skill}: n={len(has)}, mean {has.mean():.2f} LPA · Without: n={len(no)}, "
                       f"mean {no.mean():.2f} LPA. The **raw** gap mixes in experience, tier and region "
                       f"(people with a skill are often more senior). The **adjusted** premium is the "
                       f"skill's effect holding those constant (robust HC3 standard errors).")

        st.divider()
        st.subheader("Salary differences across groups")
        factor = st.radio("Group by", ["company_tier", "region"], horizontal=True)
        groups = [np.log(g["salary_lpa"].to_numpy()) for _, g in df.groupby(factor)]
        f_stat, f_p = stats.f_oneway(*groups)
        h_stat, h_p = stats.kruskal(*groups)
        lev_stat, lev_p = stats.levene(*groups)
        allv = np.concatenate(groups)
        eta2 = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / np.sum((allv - allv.mean()) ** 2)
        c = st.columns(4)
        c[0].metric("ANOVA (log) F", f"{f_stat:.1f}", f"p = {f_p:.2g}")
        c[1].metric("Kruskal-Wallis p", f"{h_p:.2g}")
        c[2].metric("Levene (equal var) p", f"{lev_p:.2g}")
        c[3].metric("Effect size η²", f"{eta2:.3f}")
        tk = pairwise_tukeyhsd(np.log(df["salary_lpa"].to_numpy()), df[factor].to_numpy())
        tdf = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
        tdf["meandiff"] = tdf["meandiff"].astype(float)
        tdf["% difference"] = (np.exp(tdf["meandiff"]) - 1) * 100
        st.markdown("**Tukey HSD post-hoc** (which pairs differ, on the log scale; group2 vs group1):")
        show_df(tdf[["group1", "group2", "% difference", "p-adj", "reject"]].round(3))

    # =========================== TAB 3: PREDICTION ==============================
    with t_pred:
        st.subheader("Estimated market pay for your profile")
        c = st.columns(5)
        c[0].metric("Median estimate", f"{pred_median:.1f} LPA")
        c[1].metric("Mean estimate", f"{pred_mean:.1f} LPA")
        c[2].metric("80% range (P10–P90)", f"{pred_lo:.0f} – {pred_hi:.0f}")
        c[3].metric(f"Percentile in {tier}", f"{pct_in_tier:.0f}th")
        c[4].metric(f"P(≥ {tier} median)", f"{p_above * 100:.0f}%", f"median {tier_median:.1f} LPA")
        st.caption(("🟢 Competitive" if p_above >= 0.5 else "🟠 Under-indexed") +
                   f" versus the {tier} median. The range is a **prediction interval** for one person "
                   f"(individual pay varies a lot), not a confidence interval for the average.")

        yrs = np.arange(0.5, 15.01, 0.5)
        traj = predict_many(bundle, [dict(profile, experience=y) for y in yrs])
        f = go.Figure()
        f.add_scatter(x=yrs, y=traj["high"], line=dict(width=0), showlegend=False)
        f.add_scatter(x=yrs, y=traj["low"], fill="tonexty", line=dict(width=0), name="80% range")
        f.add_scatter(x=yrs, y=traj["median"], name="Median", line=dict(width=3))
        f.add_scatter(x=[exp], y=[pred_median], mode="markers", name="You",
                      marker=dict(size=13, symbol="star"))
        f.update_layout(title="Career trajectory (same skills, tier & region)",
                        xaxis_title="Years of experience", yaxis_title="LPA", height=380)
        show_chart(f)

        st.subheader("Where does the same profile earn most? (tier × region)")
        combos = [dict(profile, tier=t, region=r) for t in tiers for r in regions]
        cm = predict_many(bundle, combos)["median"].to_numpy().reshape(len(tiers), len(regions))
        hm = px.imshow(cm, x=regions, y=tiers, text_auto=".1f", aspect="auto",
                       color_continuous_scale="Blues", labels=dict(color="LPA"))
        show_chart(hm)

        st.subheader("What drives pay? (robust regression, effect on salary)")
        ct = coef_table(bundle)
        show_chart(px.bar(ct.sort_values("Effect %"), x="Effect %", y="Feature", orientation="h",
                          title="% change in salary, holding everything else equal"))
        show_df(ct.round(3))
        st.caption("Tier/region effects are relative to the baseline category (alphabetically first). "
                   "Age is excluded because it is almost perfectly collinear with experience (see Model health).")

    # =========================== TAB 4: SKILL ROI ===============================
    with t_skill:
        st.subheader("Which skill should you learn next?")
        missing = [s for s in skills if s not in sel]
        if not missing:
            st.success("You already have every skill in the dataset. Focus on scope, tier and negotiation.")
        else:
            base = point_median(bundle, profile)
            effort_df = pd.DataFrame({"Skill": missing,
                                      "Weeks to learn": [DEFAULT_WEEKS.get(s, 8) for s in missing]})
            st.markdown("Edit the **weeks to learn** to match your reality:")
            edited = st.data_editor(effort_df, key="effort_" + "|".join(missing), hide_index=True,
                                    disabled=["Skill"])
            rows = []
            for _, r in edited.iterrows():
                new = point_median(bundle, dict(profile, skills=sel + [r["Skill"]]))
                weeks = max(float(r["Weeks to learn"]), 1.0)
                rows.append({"Skill": r["Skill"], "Uplift (LPA)": new - base,
                             "Uplift %": (new / base - 1) * 100, "Weeks": weeks,
                             "LPA per 10 weeks": (new - base) / weeks * 10})
            roi = pd.DataFrame(rows).sort_values("LPA per 10 weeks", ascending=False).reset_index(drop=True)
            show_df(roi.round(2))
            show_chart(px.bar(roi, x="Skill", y="LPA per 10 weeks", title="Learning ROI (higher = better payoff per effort)"))

            # greedy learning path by ROI
            path, cur, weeks_cum = [dict(step="Today", skills=list(sel), lpa=base, weeks=0.0)], list(sel), 0.0
            for _, r in roi.iterrows():
                cur = cur + [r["Skill"]]
                weeks_cum += r["Weeks"]
                path.append(dict(step=f"+ {r['Skill']}", skills=list(cur),
                                 lpa=point_median(bundle, dict(profile, skills=cur)), weeks=weeks_cum))
            pdf = pd.DataFrame(path)
            fp = px.line(pdf, x="weeks", y="lpa", markers=True, text="step",
                         title="Greedy learning path: cumulative weeks vs. expected median pay")
            fp.update_traces(textposition="top center")
            fp.update_layout(xaxis_title="Cumulative weeks", yaxis_title="LPA (median)")
            show_chart(fp)
            st.caption("Uplifts come from the regression (associational, not guaranteed causal). "
                       "Effort weeks are assumptions you can edit above.")

    # =========================== TAB 5: SEGMENTS ================================
    with t_seg:
        st.subheader("Market segmentation (K-Means)")
        scan_df = df[["experience", *skill_cols]].assign(log_salary=np.log(df["salary_lpa"]))
        scan = silhouette_scan(scan_df)
        cc = st.columns([1, 2])
        with cc[0]:
            k = st.slider("Number of bands (k)", 2, 7, 4)
            best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
            st.caption(f"Best silhouette at k = {best_k}. Real markets are continuous, so use k as a "
                       f"storytelling device rather than a hard truth.")
        with cc[1]:
            show_chart(px.line(scan, x="k", y="silhouette", markers=True, title="Silhouette by k"))

        seg = segment(df, skill_cols, k, profile, pred_median)
        prof_tbl = seg["profiles"].rename(columns={"band": "Band", "n": "n", "avg": "Avg LPA",
                                                   "lo": "Min", "hi": "Max", "exp": "Avg exp (yrs)"})
        show_df(prof_tbl[["Band", "n", "Min", "Avg LPA", "Max", "Avg exp (yrs)"]].round(2))
        sd = seg["data"].sample(min(len(seg["data"]), 1500), random_state=1)
        sd["band"] = sd["band"].astype(str)
        show_chart(px.scatter(sd, x="experience", y="salary_lpa", color="band", opacity=0.6,
                              category_orders={"band": [str(i) for i in range(1, k + 1)]},
                              title="Bands: experience vs salary"))
        ub = seg["user_band"]
        st.success(f"Your profile maps to **Band {ub}** of {k}.")
        if ub < k:
            nxt = seg["prevalence"].loc[ub + 1]
            cur_prev = seg["prevalence"].loc[ub]
            gap = pd.DataFrame({
                "Skill": [sk_label(c) for c in skill_cols],
                f"Band {ub + 1} prevalence %": (nxt.to_numpy() * 100),
                f"Band {ub} prevalence %": (cur_prev.to_numpy() * 100),
            })
            gap["Lift (pp)"] = gap.iloc[:, 1] - gap.iloc[:, 2]
            gap = gap[~gap["Skill"].isin(sel)].sort_values("Lift (pp)", ascending=False)
            st.markdown(f"**Skills over-represented in Band {ub + 1} that you don't have yet** "
                        f"(lift vs. your current band):")
            show_df(gap.round(1))
        else:
            st.info("You are in the top band of this market. Growth now comes from scope, tier and equity.")

    # =========================== TAB 6: OFFER ===================================
    with t_offer:
        st.subheader("Evaluate an offer")
        c = st.columns(3)
        offer = c[0].number_input("Offer (LPA)", 1.0, 500.0, float(round(pred_median, 1)), 0.5)
        current = c[1].number_input("Current CTC (LPA, 0 = skip)", 0.0, 500.0, 0.0, 0.5)
        basic_pct = c[2].slider("Basic pay as % of CTC", 30, 60, 40) / 100

        sigma = bundle["sigma"]
        z = (np.log(offer) - np.log(pred_median)) / sigma
        pct = float(stats.norm.cdf(z) * 100)
        verdict = ("🔴 Below market" if pct < 25 else "🟡 Fair" if pct < 60 else
                   "🟢 Strong" if pct < 85 else "🟣 Exceptional")
        c = st.columns(3)
        c[0].metric("Market percentile for your profile", f"{pct:.0f}th", verdict)
        c[1].metric("Offer vs. median estimate", f"{(offer / pred_median - 1) * 100:+.1f}%")
        if current > 0:
            c[2].metric("Hike vs. current", f"{(offer / current - 1) * 100:+.1f}%")

        st.markdown("**Negotiation ladder** (what people with your profile typically land):")
        ladder = pd.DataFrame({
            "Anchor": ["Floor (P25)", "Fair (P50)", "Good ask (P70)", "Stretch (P85)"],
            "LPA": [pred_median * np.exp(sigma * stats.norm.ppf(q)) for q in (0.25, 0.50, 0.70, 0.85)],
        })
        show_df(ladder.round(1))

        st.markdown("**Approximate in-hand salary** (India new tax regime, simplified):")
        a, b = in_hand(offer, basic_pct), in_hand(pred_median, basic_pct)
        cmp_df = pd.DataFrame({
            "": ["CTC", "Income tax / yr", "PF (both sides) / yr", "Monthly in-hand"],
            "This offer": [inr(offer * 1e5), inr(a["tax"]), inr(a["pf"]), inr(a["monthly"])],
            "Market median": [inr(pred_median * 1e5), inr(b["tax"]), inr(b["pf"]), inr(b["monthly"])],
        })
        show_df(cmp_df)
        st.caption("Assumes FY2025-26 new-regime slabs, ₹75k standard deduction, 12% PF on basic, no bonus/"
                   "variable split, no professional tax. Tax rules change — verify before deciding.")

    # =========================== TAB 7: MODEL HEALTH ============================
    with t_health:
        st.subheader("Can you trust the model?")
        v = validate_ols(bundle)
        ols = bundle["ols"]
        c = st.columns(4)
        c[0].metric("In-sample R² (log)", f"{ols.rsquared:.3f}")
        c[1].metric("Hold-out R² (log)", f"{v['r2_log']:.3f}")
        c[2].metric("5-fold CV R²", f"{v['cv_r2']:.3f}", f"± {v['cv_std']:.3f}")
        c[3].metric("Classifier AUC (hold-out)", f"{auc:.3f}")
        c = st.columns(3)
        c[0].metric("Hold-out MAE", f"{v['mae']:.2f} LPA")
        c[1].metric("Hold-out MAPE", f"{v['mape']:.1f}%")
        c[2].metric("80% interval coverage", f"{v['coverage']:.0f}%", "target ≈ 80%")

        bp = het_breuschpagan(np.asarray(ols.resid), ols.model.exog)
        jb = stats.jarque_bera(np.asarray(ols.resid))
        st.caption(f"Breusch-Pagan heteroskedasticity p = {bp[1]:.3g} (small → error variance is uneven; "
                   f"we therefore report robust HC3 coefficients). Jarque-Bera normality of residuals "
                   f"p = {jb.pvalue:.3g}. Smearing factor for back-transforming to LPA: {bundle['smear']:.3f}.")
        rf = px.histogram(x=np.asarray(ols.resid), nbins=50, title="Residuals (log scale)")
        show_chart(rf)

        st.subheader("Multicollinearity (VIF)")
        vt = vif_table(df, skill_cols)
        show_df(vt.round(2))
        if "age" in df.columns and df["age"].notna().all():
            st.caption("Age and experience move together (VIF ≫ 5), so age is **dropped** from the regression "
                       "to keep coefficients stable and interpretable.")

        st.subheader("Full robust coefficient table")
        show_df(coef_table(bundle).round(4))

    # =========================== EXPORT =========================================
    st.sidebar.header("3 · Export")
    report = "\n".join([
        "# Salary Market Report",
        f"- Profile: {exp} yrs | skills: {', '.join(sel) or 'none'} | {tier} | {region}",
        f"- Median estimate: {pred_median:.2f} LPA  (mean {pred_mean:.2f})",
        f"- 80% range: {pred_lo:.1f} - {pred_hi:.1f} LPA",
        f"- Percentile within {tier}: {pct_in_tier:.0f}th",
        f"- P(>= {tier} median of {tier_median:.1f} LPA): {p_above * 100:.0f}%",
        f"- Records used: {len(df)} (outliers removed: {len(flagged)})",
    ])
    st.sidebar.download_button("Download report (.md)", report, "salary_report.md", "text/markdown")
    st.sidebar.download_button("Download cleaned data (.csv)", df.to_csv(index=False),
                               "salary_clean.csv", "text/csv")


main()
