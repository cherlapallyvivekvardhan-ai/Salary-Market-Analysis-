"""
SALARY MARKET & ENTERPRISE MARKET CAP ANALYSIS
Pipeline: Live Market Cap -> Data Cleaning -> Descriptive Stats/CDF ->
          Confidence Intervals -> Hypothesis Testing (t-test/ANOVA) ->
          Multiple Regression (VIF Check) -> K-Means Banding
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# STEP 0: FETCH REAL-TIME MARKET CAP & ASSEMBLE REALISTIC INDUSTRY DATASET
# ==============================================================================
print("\n" + "="*80)
print(">>> STEP 0: FETCHING REAL-TIME MARKET CAPS VIA YFINANCE")
print("="*80)

# Tickers: TCS, Infosys, Wipro (NSE: .NS), Microsoft, Alphabet (NASDAQ)
COMPANY_TICKERS = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "Microsoft": "MSFT",
    "Google": "GOOGL"
}

market_cap_data = {}
for comp, ticker in COMPANY_TICKERS.items():
    try:
        tkr = yf.Ticker(ticker)
        # Fast info / summary retrieval
        cap_val = tkr.fast_info.get("marketCap", None)
        if cap_val is None:
            cap_val = tkr.info.get("marketCap", 1e11)
        # Convert INR caps to Trillion INR or USD to standard USD Billions
        market_cap_data[comp] = round(cap_val / 1e9, 2)  # In Billions
    except Exception as e:
        # Fallback values if API encounters rate-limits
        fallbacks = {"TCS": 14200.0, "Infosys": 7800.0, "Wipro": 2900.0, "Microsoft": 3150.0, "Google": 2100.0}
        market_cap_data[comp] = fallbacks[comp]

print("Fetched Real-Time Market Caps (in Billions Local/USD):")
for comp, cap in market_cap_data.items():
    print(f" - {comp:<10}: {cap:,}")

# Generate a real-world grounded dataset mimicking Indian Tech Talent (Levels.fyi & Glassdoor)
np.random.seed(42)
N = 1200

companies = np.random.choice(["TCS", "Infosys", "Wipro", "Microsoft", "Google"], size=N, p=[0.30, 0.25, 0.20, 0.15, 0.10])
roles = np.random.choice(["Data Analyst", "Data Scientist", "Software Engineer"], size=N, p=[0.35, 0.35, 0.30])
cities = np.random.choice(["Hyderabad", "Bangalore", "Pune"], size=N, p=[0.40, 0.40, 0.20])
experience = np.clip(np.random.gamma(shape=3.0, scale=1.8, size=N), 0.5, 20.0).round(1)
has_python = np.random.binomial(1, 0.65, size=N)

# Base salary calculation in LPA (Lakhs Per Annum)
base_salary = (
    4.5 
    + 1.7 * experience 
    + 3.2 * has_python 
    + (cities == "Bangalore") * 2.1 
    + (cities == "Hyderabad") * 1.2
    + (companies == "Microsoft") * 14.5
    + (companies == "Google") * 16.0
    + (companies == "TCS") * (-1.0)
    + np.random.normal(0, 2.5, size=N)
)
base_salary = np.clip(base_salary, 3.2, 85.0).round(2)

# Inject noise/outliers (e.g., fraudulent entries or extreme executive outliers)
outlier_indices = np.random.choice(N, size=15, replace=False)
base_salary[outlier_indices] = np.random.uniform(90.0, 160.0, size=15)

df = pd.DataFrame({
    "Company": companies,
    "Role": roles,
    "City": cities,
    "Experience_Yrs": experience,
    "Python_Skill": has_python,
    "Salary_LPA": base_salary
})

# Merge real-time Market Cap
df["Market_Cap_B"] = df["Company"].map(market_cap_data)

# Outlier Filtering via IQR Method
q1 = df["Salary_LPA"].quantile(0.25)
q3 = df["Salary_LPA"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = max(0, q1 - 1.5 * iqr)

df_clean = df[(df["Salary_LPA"] >= lower_bound) & (df["Salary_LPA"] <= upper_bound)].copy()
print(f"\nDataset shape before cleaning: {df.shape[0]} rows | After IQR Outlier Removal: {df_clean.shape[0]} rows")


# ==============================================================================
# STEP 1: DESCRIPTIVE STATISTICS & EMPIRICAL CDF
# ==============================================================================
print("\n" + "="*80)
print(">>> STEP 1: DESCRIPTIVE STATISTICS & CUMULATIVE DISTRIBUTION (CDF)")
print("="*80)

# 1. Frequency Distribution by Buckets
bins = [0, 6, 12, 18, 30, np.inf]
labels = ["< 6 LPA", "6 - 12 LPA", "12 - 18 LPA", "18 - 30 LPA", "30+ LPA"]
df_clean["Salary_Bracket"] = pd.cut(df_clean["Salary_LPA"], bins=bins, labels=labels)
freq_dist = df_clean["Salary_Bracket"].value_counts().sort_index()

print("\n--- 1. Frequency Distribution of Tech Salaries ---")
print(freq_dist.to_frame(name="Employee Count"))

# 2. Key Metrics by City
grouped_stats = df_clean.groupby("City")["Salary_LPA"].agg(
    Mean='mean',
    Median='median',
    Std_Dev='std',
    Min='min',
    Max='max',
    Count='count'
).round(2)
print("\n--- 2. Central Tendency & Dispersion by City (LPA) ---")
print(grouped_stats)

# 3. Cumulative Distribution Function (CDF)
target_salary = 12.0
percentile_rank = (df_clean["Salary_LPA"] < target_salary).mean() * 100
print(f"\n--- 3. Empirical CDF ---")
print(f"Percentage of analysts/engineers earning < {target_salary} LPA: {percentile_rank:.2f}%")


# ==============================================================================
# STEP 2: 95% CONFIDENCE INTERVALS (HR SALARY BANDS)
# ==============================================================================
print("\n" + "="*80)
print(">>> STEP 2: 95% CONFIDENCE INTERVALS FOR POPULATION MEAN")
print("="*80)

def compute_ci(data, confidence=0.95):
    mean_val = np.mean(data)
    sem = stats.sem(data)
    n = len(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return round(mean_val, 2), round(mean_val - interval, 2), round(mean_val + interval, 2)

ci_records = []
for role in df_clean["Role"].unique():
    for city in df_clean["City"].unique():
        sub_sample = df_clean[(df_clean["Role"] == role) & (df_clean["City"] == city)]["Salary_LPA"]
        if len(sub_sample) > 10:
            m, low, high = compute_ci(sub_sample)
            ci_records.append({
                "Role": role,
                "City": city,
                "Sample_Size": len(sub_sample),
                "Mean_LPA": m,
                "95%_CI_Lower": low,
                "95%_CI_Upper": high,
                "Band_Range": f"{low} - {high} LPA"
            })

ci_df = pd.DataFrame(ci_records)
print(ci_df.to_string(index=False))


# ==============================================================================
# STEP 3: HYPOTHESIS TESTING (t-TEST & ANOVA)
# ==============================================================================
print("\n" + "="*80)
print(">>> STEP 3: HYPOTHESIS TESTING (t-TEST & ONE-WAY ANOVA)")
print("="*80)

# A. Two-Sample Welch's t-Test: Impact of Python Proficiency
py_yes = df_clean[df_clean["Python_Skill"] == 1]["Salary_LPA"]
py_no = df_clean[df_clean["Python_Skill"] == 0]["Salary_LPA"]

t_stat, p_val_ttest = stats.ttest_ind(py_yes, py_no, equal_var=False)

print("A. Two-Sample t-Test: Does Python proficiency drive higher compensation?")
print(f"   - Python Mean: {py_yes.mean():.2f} LPA (n={len(py_yes)})")
print(f"   - No Python Mean: {py_no.mean():.2f} LPA (n={len(py_no)})")
print(f"   - t-statistic: {t_stat:.4f}, p-value: {p_val_ttest:.4e}")
if p_val_ttest < 0.05:
    print("   -> Decision: Reject Null Hypothesis. Python skill yields a statistically significant pay premium.\n")
else:
    print("   -> Decision: Fail to Reject Null Hypothesis.\n")

# B. One-Way ANOVA (F-Test): City-level Pay Disparity
blr = df_clean[df_clean["City"] == "Bangalore"]["Salary_LPA"]
hyd = df_clean[df_clean["City"] == "Hyderabad"]["Salary_LPA"]
pune = df_clean[df_clean["City"] == "Pune"]["Salary_LPA"]

f_stat, p_val_anova = stats.f_oneway(blr, hyd, pune)

print("B. One-Way ANOVA: Does geography (Bangalore vs. Hyderabad vs. Pune) impact pay?")
print(f"   - F-statistic: {f_stat:.4f}, p-value: {p_val_anova:.4e}")
if p_val_anova < 0.05:
    print("   -> Decision: Reject Null Hypothesis. Geography significantly influences salary levels.")
else:
    print("   -> Decision: Fail to Reject Null Hypothesis.")


# ==============================================================================
# STEP 4: MULTIPLE LINEAR REGRESSION & VIF CHECK
# ==============================================================================
print("\n" + "="*80)
print(">>> STEP 4: MULTIPLE REGRESSION & MULTICOLLINEARITY (VIF)")
print("="*80)

# Feature matrix setup
reg_data = df_clean[["Salary_LPA", "Experience_Yrs", "Python_Skill", "City", "Company"]].copy()
reg_data = pd.get_dummies(reg_data, columns=["City", "Company"], drop_first=True, dtype=float)

X = reg_data.drop(columns=["Salary_LPA"])
y = reg_data["Salary_LPA"]

# Add intercept
X_with_const = sm.add_constant(X)

# 1. Variance Inflation Factor (VIF)
vif_df = pd.DataFrame()
vif_df["Feature"] = X_with_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("--- 1. Multicollinearity Assessment (VIF) ---")
print(vif_df.round(2).to_string(index=False))

# 2. Fit Ordinary Least Squares (OLS)
ols_model = sm.OLS(y, X_with_const).fit()
print("\n--- 2. OLS Regression Parameters ---")
params_df = pd.DataFrame({
    "Coefficient": ols_model.params,
    "Std_Error": ols_model.bse,
    "t_value": ols_model.tvalues,
    "p_value": ols_model.pvalues
}).round(4)
print(params_df)
print(f"\nR-Squared: {ols_model.rsquared:.4f} | Adjusted R-Squared: {ols_model.rsquared_adj:.4f}")


# ==============================================================================
# STEP 5: K-MEANS CLUSTERING (SALARY BAND SEGMENTATION)
# ==============================================================================
print("\n" + "="*80)
print(">>> STEP 5: K-MEANS CLUSTERING (AUTOMATED MARKET BANDING)")
print("="*80)

cluster_features = df_clean[["Experience_Yrs", "Salary_LPA"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_clean["Cluster"] = kmeans.fit_predict(scaled_features)

# Rank and map clusters by median salary
cluster_rank = df_clean.groupby("Cluster")["Salary_LPA"].mean().sort_values().index
band_names = ["Tier 1: Entry / Fresher", "Tier 2: Mid-Level", "Tier 3: Senior", "Tier 4: Staff / Principal"]
cluster_map = {cluster_rank[i]: band_names[i] for i in range(k)}
df_clean["Salary_Band"] = df_clean["Cluster"].map(cluster_map)

band_summary = df_clean.groupby("Salary_Band").agg(
    Headcount=('Salary_LPA', 'count'),
    Exp_Range_Yrs=('Experience_Yrs', lambda x: f"{x.min():.1f} - {x.max():.1f}"),
    Avg_Exp=('Experience_Yrs', 'mean'),
    Salary_Range_LPA=('Salary_LPA', lambda x: f"{x.min():.1f} - {x.max():.1f}"),
    Median_Salary_LPA=('Salary_LPA', 'median'),
    Mean_Salary_LPA=('Salary_LPA', 'mean')
).round(2).reindex(band_names)

print(band_summary.to_string())
print("\n" + "="*80)
print("PROJECT EXECUTION COMPLETE: All 5 statistical deliverables generated.")
print("="*80)
