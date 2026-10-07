"""Data + economics + ML for the Labour-to-Owner Salary Ledger.
All market data here is SIMULATED but deterministic per day (same day -> same numbers).
Swap in real files (see README) to go live."""
from __future__ import annotations
import zlib
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TODAY = date.today()
MCA_ACTIVE_COMPANIES = 2_154_810  # Aug 2026, MCA

# sector: (value-added ratio, revenue per employee INR/yr, daily demand drift)
SECTORS = {
    "IT & Software": (0.42, 5.2e6, 0.00045),
    "Manufacturing": (0.28, 4.6e6, 0.00030),
    "Banking & Finance": (0.45, 4.0e6, 0.00040),
    "Retail & Trade": (0.18, 4.7e6, 0.00020),
    "Healthcare": (0.38, 3.4e6, 0.00050),
    "Logistics": (0.22, 6.4e6, 0.00030),
}
# role: (sector, base monthly salary INR: 5 yrs exp, Graduate, Tier-1 city)
ROLES = {
    "Software Engineer": ("IT & Software", 70000), "Data Scientist": ("IT & Software", 95000),
    "QA Engineer": ("IT & Software", 48000), "Systems Architect": ("IT & Software", 150000),
    "Machine Operator": ("Manufacturing", 24000), "Quality Inspector": ("Manufacturing", 32000),
    "Plant Manager": ("Manufacturing", 110000), "Maintenance Engineer": ("Manufacturing", 45000),
    "Credit Analyst": ("Banking & Finance", 65000), "Risk Officer": ("Banking & Finance", 90000),
    "Bank Clerk": ("Banking & Finance", 34000), "Finance Manager": ("Banking & Finance", 120000),
    "Store Associate": ("Retail & Trade", 20000), "Store Manager": ("Retail & Trade", 55000),
    "Supply Planner": ("Retail & Trade", 48000),
    "Staff Nurse": ("Healthcare", 32000), "Lab Technician": ("Healthcare", 27000),
    "Doctor (MBBS)": ("Healthcare", 95000), "Hospital Administrator": ("Healthcare", 60000),
    "Delivery Associate": ("Logistics", 19000), "Fleet Supervisor": ("Logistics", 40000),
    "Logistics Coordinator": ("Logistics", 34000), "Operations Director": ("Logistics", 150000),
}
QUALS = {"10th pass": 0.80, "12th pass": 0.88, "Diploma / ITI": 0.95, "Graduate": 1.00,
         "B.Tech / BE": 1.12, "Post-graduate / MBA": 1.30, "PhD": 1.50}
QUAL_LEVEL = {q: i for i, q in enumerate(QUALS)}
CITIES = {"Bengaluru": 1.00, "Mumbai": 1.02, "Delhi-NCR": 0.99, "Hyderabad": 0.96, "Pune": 0.94,
          "Chennai": 0.93, "Ahmedabad": 0.82, "Jaipur": 0.76, "Kochi": 0.78, "Indore": 0.74,
          "Coimbatore": 0.75, "Lucknow": 0.72}

def _z(key: str, d: date) -> float:
    """Deterministic standard-normal shock for (key, day)."""
    return float(np.random.default_rng(zlib.crc32(f"{key}|{d}".encode())).standard_normal())

def _dates(days: int) -> list[date]:
    return [TODAY - timedelta(days=days - 1 - i) for i in range(days)]

def sector_index(sector: str, days: int = 365) -> pd.Series:
    ds, drift = _dates(days), SECTORS[sector][2]
    r = np.array([drift + 0.003 * _z(sector, d) + (0.004 if (d.month, d.day) == (4, 1) else 0) for d in ds])
    s = pd.Series(np.cumprod(1 + r), index=pd.to_datetime(ds))
    return s / s.iloc[-1]  # today = 1.0

def role_index(role: str, days: int = 365) -> pd.Series:
    s = sector_index(ROLES[role][0], days)
    n = np.cumprod(1 + np.array([0.0015 * _z(role, d.date()) for d in s.index]))
    x = s * n
    return x / x.iloc[-1]

def fair_salary(role: str, exp: float, qual: str, city: str, premium: float = 1.0) -> float:
    base = ROLES[role][1] / (1 + 0.07 * 5)
    return base * (1 + 0.07 * min(exp, 15)) * QUALS[qual] * CITIES[city] * premium

def load_companies(n: int = 800) -> pd.DataFrame:
    p = Path(__file__).parent / "data" / "mca_companies.csv"
    if p.exists():  # real MCA master data: cin,name,sector,city,paid_up_cr,revenue_cr,headcount,premium
        return pd.read_csv(p)
    rng = np.random.default_rng(7)
    a = ["Bharat", "Surya", "Kaveri", "Indus", "Nava", "Aryan", "Tulsi", "Meridian", "Vikram", "Anvi", "Saras", "Orion"]
    suffix = {"IT & Software": "Technologies", "Manufacturing": "Industries", "Banking & Finance": "Finserv",
              "Retail & Trade": "Retail", "Healthcare": "Healthcare", "Logistics": "Logistics"}
    rows = []
    for i in range(n):
        sec = list(SECTORS)[rng.integers(len(SECTORS))]
        hc = int(np.clip(rng.lognormal(4.5, 1.2), 10, 50000))
        rev = hc * SECTORS[sec][1] * rng.lognormal(0, 0.25) / 1e7  # Rs crore
        paid = rev * rng.uniform(0.05, 0.4)
        rows.append({"cin": f"U{rng.integers(10000, 99999)}MH{rng.integers(2000, 2025)}PTC{i:06d}",
                     "name": f"{a[rng.integers(len(a))]} {suffix[sec]} Pvt Ltd", "sector": sec,
                     "city": list(CITIES)[rng.integers(len(CITIES))], "paid_up_cr": round(paid, 2),
                     "revenue_cr": round(rev, 2), "headcount": hc,
                     "premium": float(np.clip(0.85 + 0.1 * np.log10(1 + paid) + rng.normal(0, .05), .8, 1.4))})
    return pd.DataFrame(rows)

def company_economics(c: pd.Series, exp: float = 5, qual: str = "Graduate") -> dict:
    """Labour vs owner split of value added for one company (per employee, per year)."""
    roles = [r for r, v in ROLES.items() if v[0] == c["sector"]]
    ctc = np.mean([fair_salary(r, exp, qual, c["city"], c["premium"]) for r in roles]) * 12 * 1.12
    rev_pe = c["revenue_cr"] * 1e7 / c["headcount"]
    va = rev_pe * SECTORS[c["sector"]][0]
    return {"ctc": ctc, "value_added": va, "labour_share": ctc / va * 100,
            "owner_surplus": va - ctc, "rev_per_emp": rev_pe}

def company_daily(c: pd.Series, role: str, exp: float, qual: str, days: int = 90) -> pd.Series:
    base = fair_salary(role, exp, qual, c["city"], c["premium"])
    idx = role_index(role, days)
    noise = np.array([1 + 0.002 * _z(c["cin"] + role, d.date()) for d in idx.index])
    return base * idx * noise

# ---- ML: fair-salary model (trained on simulated panel; replace with real postings/payroll) ----
def _features(df: pd.DataFrame) -> np.ndarray:
    roles = pd.CategoricalDtype(list(ROLES))
    oh = pd.get_dummies(df["role"].astype(roles)).astype(float)
    num = pd.DataFrame({"exp": df["exp"], "qual": df["qual"].map(QUAL_LEVEL),
                        "city": df["city"].map(CITIES), "prem": df["premium"]})
    return pd.concat([oh, num], axis=1).values.astype(float)

def train_model() -> RandomForestRegressor:
    rng = np.random.default_rng(1)
    r, q, c = list(ROLES), list(QUALS), list(CITIES)
    df = pd.DataFrame({"role": rng.choice(r, 10000), "exp": rng.integers(0, 21, 10000),
                       "qual": rng.choice(q, 10000), "city": rng.choice(c, 10000),
                       "premium": np.clip(rng.normal(1, .12, 10000), .8, 1.4)})
    y = [fair_salary(a, b, d, e, f) * rng.lognormal(0, .08)
         for a, b, d, e, f in zip(df.role, df.exp, df.qual, df.city, df.premium)]
    m = RandomForestRegressor(n_estimators=60, min_samples_leaf=3, random_state=1, n_jobs=-1)
    return m.fit(_features(df), y)

def predict_band(m: RandomForestRegressor, role, exp, qual, city, premium) -> tuple[float, float, float]:
    X = _features(pd.DataFrame([{"role": role, "exp": exp, "qual": qual, "city": city, "premium": premium}]))
    preds = [t.predict(X)[0] for t in m.estimators_]
    p10, p50, p90 = np.percentile(preds, [10, 50, 90])
    return float(p10), float(p50), float(p90)

# ---- Government pay (7th CPC pay matrix, entry basic pay by level) ----
PAY_LEVEL = {1: 18000, 2: 19900, 3: 21700, 4: 25500, 5: 29200, 6: 35400, 7: 44900, 8: 47600, 9: 53100,
             10: 56100, 11: 67700, 12: 78800, 13: 123100, 14: 144200}
GOV_POSTS = {
    "Group D / MTS (RRB Group D, SSC MTS)": (1, "10th pass"),
    "LDC / Postal Assistant (SSC CHSL)": (2, "12th pass"),
    "Constable (CAPF / SSC GD)": (3, "12th pass"),
    "Tax Assistant / Station Master (SSC, RRB)": (4, "Graduate"),
    "Senior Clerk / Commercial Apprentice (RRB NTPC)": (5, "Graduate"),
    "Junior Engineer (SSC JE, RRB JE)": (6, "Diploma / ITI"),
    "Inspector / Assistant Section Officer (SSC CGL)": (7, "Graduate"),
    "PGT Teacher (KVS / NVS)": (8, "Post-graduate / MBA"),
    "IAS / IPS / IFS (UPSC CSE)": (10, "Graduate"),
    "Scientist B (DRDO / ISRO)": (10, "B.Tech / BE"),
}
HRA = {"X (metro)": 0.30, "Y": 0.20, "Z": 0.10}

def gov_pay(level: int, da: float, hra_class: str, years: int) -> dict:
    basic = np.ceil(PAY_LEVEL[level] * 1.03 ** years / 100) * 100
    d, h = basic * da, basic * HRA[hra_class]
    nps = 0.10 * (basic + d)
    return {"basic": basic, "da": d, "hra": h, "gross": basic + d + h, "nps": nps, "net": basic + d + h - nps}

def private_start(qual: str) -> float:
    return float(np.mean([v[1] for v in ROLES.values()])) / 1.35 * QUALS[qual] * 0.85
