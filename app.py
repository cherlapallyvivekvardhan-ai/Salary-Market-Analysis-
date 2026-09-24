"""
=================================================================================
PROD-ML: SALARY MARKET INTELLIGENCE & REAL-TIME PREDICTIVE ENGINE
Architecture:
  - Phase 1: Distribution Fitting (MLE Log-Normal), Cook's Distance Outlier Filtering,
             and 95% Confidence Interval Estimations.
  - Phase 2: Hypothesis Testing Engine (Independent Welch's t-test & One-Way ANOVA).
  - Phase 3: Dual-Model Predictive Engine:
             * OLS Regression with Variance Inflation Factor (VIF) multicollinearity checks.
             * Logistic Regression for Target Median Probability Classification.
  - Phase 4: Market Segmentation (K-Means Clustering) & Skill-Gap Career Action Plan.
=================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===============================================================================
# CONFIGURATION & SYNTHETIC DATA GENERATION
# ===============================================================================

SKILL_CATALOG = [
    "Python", "SQL", "Docker", "Kubernetes", "AWS", 
    "Machine Learning", "System Design", "Distributed Systems", "GenAI"
]

HIGH_VALUE_SKILLS = {"GenAI", "Distributed Systems", "System Design"}

REGIONS = ["Bengaluru", "Hyderabad", "Pune", "Delhi-NCR"]
COMPANY_TIERS = ["Startup", "Global MNC", "Tier-1 Product"]

def generate_mock_salary_data(n_samples: int = 2500) -> pd.DataFrame:
    """
    Generates a log-normally distributed compensation dataset with collinear 
    demographic features (Age & Experience) and skill vectors.
    """
    experience = np.clip(np.random.gamma(shape=3.0, scale=2.0, size=n_samples), 0.5, 25.0)
    # Inject collinearity: Age is directly correlated with Experience + noise
    age = 22 + experience + np.random.normal(0, 1.2, size=n_samples)
    
    tier = np.random.choice(COMPANY_TIERS, size=n_samples, p=[0.35, 0.40, 0.25])
    region = np.random.choice(REGIONS, size=n_samples, p=[0.40, 0.25, 0.20, 0.15])
    
    # Generate binary skill matrix
    skill_data = {}
    for skill in SKILL_CATALOG:
        prob = 0.55 if skill in ["Python", "SQL"] else (0.25 if skill in HIGH_VALUE_SKILLS else 0.40)
        skill_data[f"skill_{skill}"] = np.random.binomial(1, prob, size=n_samples)
    
    skills_df = pd.DataFrame(skill_data)
    premium_skill_count = sum(skills_df[f"skill_{s}"] for s in HIGH_VALUE_SKILLS)
    all_skill_count = skills_df.sum(axis=1)
    
    # Base compensation model on log scale
    tier_multipliers = {"Startup": 0.05, "Global MNC": 0.22, "Tier-1 Product": 0.58}
    region_multipliers = {"Bengaluru": 0.15, "Hyderabad": 0.08, "Pune": 0.02, "Delhi-NCR": 0.05}
    
    mu_log_salary = (
        2.2 
        + 0.085 * experience 
        + 0.04 * all_skill_count 
        + 0.12 * premium_skill_count 
        + pd.Series(tier).map(tier_multipliers).values
        + pd.Series(region).map(region_multipliers).values
    )
    
    # Log-normal distribution with random market variance
    sigma_log_salary = 0.28
    salary_lpa = np.random.lognormal(mean=mu_log_salary, sigma=sigma_log_salary)
    
    df = pd.DataFrame({
        "experience": experience,
        "age": age,
        "company_tier": tier,
        "region": region,
        "salary_lpa": np.round(salary_lpa, 2)
    })
    
    return pd.concat([df, skills_df], axis=1)


# ===============================================================================
# PHASE 1: DATA PREPARATION, DISTRIBUTION & ANOMALY FILTERING
# ===============================================================================

def run_phase_1(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 80)
    print("PHASE 1: DISTRIBUTION MODELING, CONFIDENCE INTERVALS & OUTLIER DETECTION")
    print("=" * 80)
    
    # 1. MLE Fit for Log-Normal Distribution
    shape, loc, scale = stats.lognorm.fit(df["salary_lpa"], floc=0)
    mu_mle = np.log(scale)
    sigma_mle = shape
    print(f"[*] MLE Log-Normal Fit Parameters: mu = {mu_mle:.4f}, sigma = {sigma_mle:.4f}")
    
    # Kolmogorov-Smirnov Test for Goodness of Fit
    ks_stat, ks_p_val = stats.kstest(df["salary_lpa"], "lognorm", args=(shape, loc, scale))
    print(f"[*] KS Test: Statistic = {ks_stat:.4f}, p-value = {ks_p_val:.4e} "
          f"({'Log-Normal supported' if ks_p_val > 0.05 else 'Empirical skew present'})")
    
    # 2. 95% Confidence Interval for Salary Across Corporate Tiers
    print("\n[*] 95% Parametric Confidence Intervals (LPA) by Company Tier:")
    for tier in COMPANY_TIERS:
        tier_data = df[df["company_tier"] == tier]["salary_lpa"]
        n = len(tier_data)
        mean = tier_data.mean()
        sem = stats.sem(tier_data)
        ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
        print(f"    - {tier:<16}: Mean = {mean:6.2f} LPA | 95% CI = [{ci_low:6.2f}, {ci_high:6.2f}]")
    
    # 3. Cook's Distance Outlier Filtering via OLS Baseline
    base_formula = "salary_lpa ~ experience + " + " + ".join([c for c in df.columns if c.startswith("skill_")])
    model = sm.OLS.from_formula(base_formula, data=df).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    
    # Standard threshold 4 / n
    threshold = 4.0 / len(df)
    outliers = np.where(cooks_d > threshold)[0]
    df_clean = df.drop(index=outliers).reset_index(drop=True)
    
    print(f"\n[*] Cook's Distance Anomaly Screening:")
    print(f"    - Threshold (4/n)   : {threshold:.6f}")
    print(f"    - Outliers Removed  : {len(outliers)} rows ({len(outliers) / len(df) * 100:.2f}%)")
    print(f"    - Retained Records  : {len(df_clean)}")
    
    return df_clean


# ===============================================================================
# PHASE 2: HYPOTHESIS TESTING ENGINE
# ===============================================================================

def run_phase_2(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("PHASE 2: STATISTICAL HYPOTHESIS TESTING (T-TEST & ANOVA)")
    print("=" * 80)
    
    # 1. Welch's t-test: Pay Gap for High-Value Skills (e.g., GenAI)
    has_genai = df[df["skill_GenAI"] == 1]["salary_lpa"]
    no_genai = df[df["skill_GenAI"] == 0]["salary_lpa"]
    
    t_stat, t_pval = stats.ttest_ind(has_genai, no_genai, equal_var=False)
    print(f"[*] Two-Sample Welch's t-test: Impact of 'GenAI' on Compensation:")
    print(f"    - Mean (With GenAI)    : {has_genai.mean():.2f} LPA (n={len(has_genai)})")
    print(f"    - Mean (Without GenAI) : {no_genai.mean():.2f} LPA (n={len(no_genai)})")
    print(f"    - Observed Delta       : +{has_genai.mean() - no_genai.mean():.2f} LPA")
    print(f"    - t-statistic = {t_stat:.4f}, p-value = {t_pval:.4e}")
    if t_pval < 0.05:
        print("    - Verdict: Statistically significant premium (p < 0.05). Reject Null Hypothesis H0.")
    else:
        print("    - Verdict: No statistically significant premium detected.")

    # 2. One-Way ANOVA: Variance Across Company Tiers
    groups = [df[df["company_tier"] == tier]["salary_lpa"].values for tier in COMPANY_TIERS]
    f_stat, f_pval = stats.f_oneway(*groups)
    print(f"\n[*] One-Way ANOVA: Salary Variance Across Company Tiers:")
    print(f"    - Tiers Compared: {COMPANY_TIERS}")
    print(f"    - F-statistic = {f_stat:.4f}, p-value = {f_pval:.4e}")
    if f_pval < 0.05:
        print("    - Verdict: Inter-tier compensation variances are statistically significant.")
    else:
        print("    - Verdict: Inconclusive variance difference across tiers.")


# ===============================================================================
# PHASE 3: REAL-TIME PREDICTIVE ENGINE (OLS & LOGISTIC CLASSIFICATION)
# ===============================================================================

def run_phase_3(df: pd.DataFrame, user_profile: dict):
    print("\n" + "=" * 80)
    print("PHASE 3: DUAL-MODEL PREDICTIVE ENGINE (REGRESSION & LOGISTIC)")
    print("=" * 80)
    
    # 1. Multicollinearity Assessment via VIF
    vif_features = df[["experience", "age"]].copy()
    vif_features = sm.add_constant(vif_features)
    vif_results = pd.DataFrame({
        "Feature": vif_features.columns,
        "VIF": [variance_inflation_factor(vif_features.values, i) for i in range(vif_features.shape[1])]
    })
    
    print("[*] Variance Inflation Factor (VIF) Diagnostics:")
    for _, row in vif_results.iterrows():
        print(f"    - {row['Feature']:<12}: {row['VIF']:.2f}")
    
    # Drop 'age' due to severe collinearity with 'experience' (VIF >> 5.0)
    print("    [!] Decision: 'age' exhibits severe collinearity with 'experience'. Dropping 'age' from regressor.")

    # 2. OLS Multiple Linear Regression (Trained on Log Scale for Skewed Target)
    df_encoded = pd.get_dummies(df.drop(columns=["age"]), columns=["company_tier", "region"], drop_first=True)
    feature_cols = [c for c in df_encoded.columns if c != "salary_lpa"]
    
    X = sm.add_constant(df_encoded[feature_cols].astype(float))
    y_log = np.log(df_encoded["salary_lpa"])
    ols_model = sm.OLS(y_log, X).fit()
    
    # Construct User Feature Vector
    user_row = {col: 0.0 for col in feature_cols}
    user_row["experience"] = user_profile["experience"]
    for s in user_profile["skills"]:
        if f"skill_{s}" in user_row:
            user_row[f"skill_{s}"] = 1.0
            
    tier_col = f"company_tier_{user_profile['target_tier']}"
    if tier_col in user_row:
        user_row[tier_col] = 1.0
        
    region_col = f"region_{user_profile['target_region']}"
    if region_col in user_row:
        user_row[region_col] = 1.0
        
    user_vector = pd.DataFrame([user_row])
    user_vector_const = sm.add_constant(user_vector, has_constant="add")
    # Ensure exact feature alignment
    user_vector_const = user_vector_const.reindex(columns=X.columns, fill_value=0.0)
    
    predicted_log_salary = ols_model.predict(user_vector_const)[0]
    # Smearing correction factor for log-transformed OLS
    residuals = y_log - ols_model.fittedvalues
    smearing_factor = np.mean(np.exp(residuals))
    predicted_lpa = np.exp(predicted_log_salary) * smearing_factor
    
    print(f"\n[*] OLS Prediction Result:")
    print(f"    - Model R-squared       : {ols_model.rsquared:.4f}")
    print(f"    - Predicted Market Rate : {predicted_lpa:.2f} LPA")

    # 3. Logistic Regression: Probability of Exceeding Target Tier's Median Salary
    tier_median = df[df["company_tier"] == user_profile["target_tier"]]["salary_lpa"].median()
    df_encoded["above_target_median"] = (df_encoded["salary_lpa"] >= tier_median).astype(int)
    
    X_clf = df_encoded[feature_cols].astype(float)
    y_clf = df_encoded["above_target_median"]
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_clf, y_clf)
    
    prob_above_median = clf.predict_proba(user_vector[feature_cols])[0][1]
    
    print(f"\n[*] Logistic Classification Engine:")
    print(f"    - Benchmark Target Tier : {user_profile['target_tier']} (Median = {tier_median:.2f} LPA)")
    print(f"    - Probability > Median  : {prob_above_median * 100:.2f}%")
    print(f"    - Classification Status : {'COMPETITIVE (Above Median)' if prob_above_median >= 0.50 else 'UNDER-INDEXED (Below Median)'}")
    
    return predicted_lpa, tier_median


# ===============================================================================
# PHASE 4: SEGMENTATION & SKILL GAP ACTION PLAN
# ===============================================================================

def run_phase_4(df: pd.DataFrame, user_profile: dict, user_predicted_lpa: float):
    print("\n" + "=" * 80)
    print("PHASE 4: K-MEANS SEGMENTATION & CAREER LADDER ACTION PLAN")
    print("=" * 80)
    
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    clustering_features = ["experience", "salary_lpa"] + skill_cols
    
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[clustering_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_matrix)
    
    # Compute cluster summary metrics
    cluster_profiles = []
    for c in range(4):
        c_df = df[df["cluster"] == c]
        cluster_profiles.append({
            "cluster_id": c,
            "mean_salary": c_df["salary_lpa"].mean(),
            "min_salary": c_df["salary_lpa"].min(),
            "max_salary": c_df["salary_lpa"].max(),
            "mean_exp": c_df["experience"].mean(),
            "skill_prevalence": c_df[skill_cols].mean()
        })
        
    # Sort clusters in ascending order of compensation
    cluster_profiles.sort(key=lambda x: x["mean_salary"])
    
    print("[*] Segmented Market Salary Bands (K-Means 4-Cluster Topology):")
    for rank, cp in enumerate(cluster_profiles):
        print(f"    Band {rank + 1} (Cluster #{cp['cluster_id']}): "
              f"Range: [{cp['min_salary']:5.2f} - {cp['max_salary']:5.2f}] LPA | "
              f"Avg: {cp['mean_salary']:5.2f} LPA | Exp: {cp['mean_exp']:.1f} yrs")
        
    # Project user profile into clustering space
    user_row = {"experience": user_profile["experience"], "salary_lpa": user_predicted_lpa}
    for col in skill_cols:
        skill_name = col.replace("skill_", "")
        user_row[col] = 1.0 if skill_name in user_profile["skills"] else 0.0
        
    user_vec_scaled = scaler.transform(pd.DataFrame([user_row])[clustering_features])
    user_cluster_id = kmeans.predict(user_vec_scaled)[0]
    
    # Locate current band rank
    current_band_rank = [i for i, cp in enumerate(cluster_profiles) if cp["cluster_id"] == user_cluster_id][0]
    print(f"\n[*] User Mapping:")
    print(f"    - Current Segment: Band {current_band_rank + 1} (Cluster #{user_cluster_id})")
    
    # Determine next cluster gap analysis
    if current_band_rank < len(cluster_profiles) - 1:
        target_band = cluster_profiles[current_band_rank + 1]
        print(f"    - Target Advancement: Band {current_band_rank + 2} (Avg: {target_band['mean_salary']:.2f} LPA)")
        
        target_skill_prev = target_band["skill_prevalence"]
        user_skills_set = set(user_profile["skills"])
        
        # Identify missing skills that have high prevalence in the target band
        gap_skills = []
        for col, rate in target_skill_prev.items():
            s_name = col.replace("skill_", "")
            if s_name not in user_skills_set and rate >= 0.40:
                gap_skills.append((s_name, rate))
                
        gap_skills.sort(key=lambda x: x[1], reverse=True)
        
        print("\n[*] Recommended Skill Acquisition Roadmap:")
        if gap_skills:
            for s_name, rate in gap_skills:
                print(f"    [+] Priority Skill: {s_name:<20} (Found in {rate * 100:.1f}% of profiles in next band)")
        else:
            print("    [+] Skills are aligned. Focus on continuous tenure and scale of impact.")
    else:
        print("    - Profile ranks within the highest compensation band in current market data.")


# ===============================================================================
# PIPELINE ORCHESTRATION & REAL-TIME INGESTION
# ===============================================================================

def main():
    # Ingest runtime profile
    live_user_profile = {
        "experience": 4.5,
        "skills": ["Python", "SQL", "Docker"],
        "target_region": "Bengaluru",
        "target_tier": "Tier-1 Product"
    }

    print("\n" + "#" * 80)
    print("RUNNING LIVE REAL-TIME SALARY PREDICTION PIPELINE")
    print(f"User Input: Exp={live_user_profile['experience']} yrs | "
          f"Skills={live_user_profile['skills']} | "
          f"Target Region={live_user_profile['target_region']} | "
          f"Target Company={live_user_profile['target_tier']}")
    print("#" * 80 + "\n")

    # Step 1: Synthesize and prepare data
    raw_df = generate_mock_salary_data(n_samples=3000)
    
    # Phase 1: Fit Distribution, Compute CIs, Filter Outliers
    clean_df = run_phase_1(raw_df)
    
    # Phase 2: Hypothesis Testing
    run_phase_2(clean_df)
    
    # Phase 3: Dual ML Inference
    predicted_lpa, target_median = run_phase_3(clean_df, live_user_profile)
    
    # Phase 4: Market Banding & Career Gap Plan
    run_phase_4(clean_df, live_user_profile, predicted_lpa)
    
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY REPORT")
    print(f"  * Estimated Fair Market Value : {predicted_lpa:.2f} LPA")
    print(f"  * Target Tier Benchmark       : {target_median:.2f} LPA ({live_user_profile['target_tier']})")
    print(f"  * Delta to Median Target      : {predicted_lpa - target_median:+.2f} LPA")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
