"""Catalogue of jobs across India (government, private, trades, informal, gig) + daily pay model.
Government pay uses 7th CPC levels (+DA/HRA). Constitutional/judicial pay is statutory (approximate; verify).
Private, trade and informal series are SIMULATED; swap job_index() for a real feed."""
from __future__ import annotations
from datetime import date, datetime
import numpy as np, pandas as pd
import core as C

C.PAY_LEVEL.update({15: 182200, 16: 205400, 17: 225000, 18: 250000})
DA = 0.58
UG, SG, DP, JC, PS, PC, TR, IN, GG = ("Union government", "State government (Maharashtra)", "Defence & police",
    "Judiciary & constitution", "PSU & banks", "Private corporate", "Skilled trades", "Informal & daily wage", "Gig & self-employed")
FAMILIES = {UG: "#0F5C5E", SG: "#2E8B8B", DP: "#556B2F", JC: "#7A2E4F", PS: "#3B5BA9",
            PC: "#5B3FA6", TR: "#C46A1B", IN: "#B8860B", GG: "#A63D40"}
# family: (daily drift, daily vol, seasonal amplitude, seasonal peak day-of-year, DA/revision step on 1 Jan & 1 Jul)
MOVE = {UG: (0, 0, 0, 0, .020), SG: (0, 0, 0, 0, .012), DP: (0, 0, 0, 0, .020), JC: (0, 0, 0, 0, 0),
        PS: (.0002, .0006, 0, 0, 0), PC: (.0004, .0030, 0, 0, 0), TR: (.0003, .0060, .03, 340, 0),
        IN: (.0002, .0090, .04, 300, 0), GG: (.0002, .0120, .03, 305, 0)}
GROWTH = {UG: .05, SG: .045, DP: .05, JC: .03, PS: .06, PC: .08, TR: .05, IN: .04, GG: .04}
FIXED = {UG, SG, DP, JC}
CATALOG: list[dict] = []

def add(name, fam, typ, lo, hi, qual, route, days=26):
    CATALOG.append(dict(name=name, family=fam, typical=float(typ), low=float(lo), high=float(hi),
                        qualification=qual, route=route, days=days))

def gv(name, fam, level, qual, route):
    g = lambda y: C.gov_pay(level, DA, "Y", y)["gross"]
    add(name, fam, g(8), g(0), g(25), qual, route)

# --- Union government (7th CPC level, gross = basic + DA + HRA Y-class) ---
for n, lv, q, r in [("Peon / MTS (Union)", 1, "10th pass", "SSC MTS, department recruitment"),
    ("Railway Group D", 1, "10th pass / ITI", "RRB Group D"), ("Postal Assistant / LDC", 2, "12th pass", "SSC CHSL"),
    ("Tax Assistant", 4, "Graduate", "SSC CGL"), ("Railway Station Master / NTPC", 5, "Graduate", "RRB NTPC"),
    ("Junior Engineer", 6, "Diploma / B.Tech", "SSC JE, RRB JE"), ("Inspector / ASO (CGL)", 7, "Graduate", "SSC CGL"),
    ("PGT Teacher (KVS)", 8, "Post-graduate + B.Ed", "KVS / NVS exam"),
    ("Scientist B (DRDO / ISRO)", 10, "B.Tech / M.Sc", "GATE / DRDO RAC / ICRB"),
    ("IRS / IFS officer (entry)", 10, "Graduate", "UPSC Civil Services"),
    ("IAS: SDM / Asst. Secretary", 10, "Graduate", "UPSC Civil Services (3 stages)"),
    ("IAS: District Collector", 12, "Graduate", "Promotion after ~9-14 yrs"),
    ("IAS: Joint Secretary", 14, "Graduate", "Central deputation, empanelment"),
    ("IAS: Secretary to Govt of India", 17, "Graduate", "Senior-most selection")]:
    gv(n, UG, lv, q, r)
add("Cabinet Secretary", UG, 250000, 250000, 250000, "Graduate", "Highest civil-service post (fixed pay)")
# --- Defence & police ---
gv("Sepoy / Soldier", DP, 3, "10th / 12th pass", "Army recruitment rally")
add("Agniveer (year 1-4 package)", DP, 35000, 30000, 40000, "10th / 12th pass", "Agnipath scheme (approx.)")
gv("Police Constable (central)", DP, 3, "12th pass", "SSC GD / CAPF")
gv("Sub-Inspector (CAPF)", DP, 6, "Graduate", "SSC CPO")
gv("IPS: Superintendent of Police", DP, 11, "Graduate", "UPSC Civil Services")
gv("IPS: IG / Director General", DP, 15, "Graduate", "Seniority + empanelment")
gv("Army Lieutenant", DP, 10, "Graduate / 12th (NDA)", "NDA / CDS / TES")
gv("Army Colonel", DP, 13, "Graduate", "Promotion ladder")
add("Chief of Army Staff", DP, 250000, 250000, 250000, "Service career", "Appointed (fixed pay)")
# --- Maharashtra (Pune) state government: approximate gross ---
for n, t, lo, hi, q, r in [("Peon / Class-4 (State)", 32000, 28000, 55000, "4th-10th pass", "District office recruitment"),
    ("Talathi (village accountant)", 42000, 35000, 75000, "Graduate", "Revenue dept exam"),
    ("State Police Constable", 40000, 33000, 70000, "12th pass", "Maharashtra Police Bharti"),
    ("Police Sub-Inspector (MPSC)", 70000, 55000, 110000, "Graduate", "MPSC PSI"),
    ("Tahsildar / Deputy Collector (MPSC)", 105000, 80000, 200000, "Graduate", "MPSC State Services"),
    ("Zilla Parishad Teacher", 52000, 38000, 95000, "D.El.Ed / B.Ed + TET", "Maharashtra TET + recruitment")]:
    add(n, SG, t, lo, hi, q, r)
# --- PSU & banks ---
for n, t, lo, hi, q, r in [("Bank Clerk (IBPS)", 38000, 30000, 60000, "Graduate", "IBPS Clerk"),
    ("Bank PO (IBPS / SBI)", 62000, 50000, 120000, "Graduate", "IBPS PO / SBI PO"),
    ("PSU Engineer (Navratna)", 125000, 80000, 250000, "B.Tech (GATE)", "GATE-based hiring")]:
    add(n, PS, t, lo, hi, q, r)
# --- Judiciary & constitution (statutory, approximate) ---
for n, t, lo, hi, q, r in [("Civil Judge (Junior Division)", 85000, 77000, 120000, "LLB", "State judicial services exam"),
    ("High Court Judge", 225000, 225000, 225000, "Advocate / judge", "Collegium appointment"),
    ("Supreme Court Judge", 250000, 250000, 250000, "Advocate / judge", "Collegium appointment"),
    ("Chief Justice of India", 280000, 280000, 280000, "Judge", "Seniority convention"),
    ("MLA (state; varies widely)", 150000, 40000, 250000, "Any (Constitution: age 25+)", "Election"),
    ("Member of Parliament", 290000, 124000, 290000, "Any (age 25+ / 30+)", "Election; salary Rs 1.24 lakh + allowances"),
    ("Prime Minister", 230000, 160000, 230000, "MP", "Leader of majority; pay is MP salary + allowances (approx.)"),
    ("Chief Minister (varies by state)", 250000, 100000, 410000, "MLA", "Leader of state majority"),
    ("Governor", 350000, 350000, 350000, "Appointed", "Appointed by the President"),
    ("Vice President", 400000, 400000, 400000, "Elected", "Electoral college"),
    ("President of India", 500000, 500000, 500000, "Elected", "Electoral college")]:
    add(n, JC, t, lo, hi, q, r)
# --- Private corporate ---
for n, t, lo, hi, q, r in [("Intern", 15000, 8000, 40000, "Pursuing degree", "Campus / portals"),
    ("Graduate Trainee", 30000, 20000, 60000, "Graduate", "Campus hiring"),
    ("Software Engineer", 70000, 40000, 180000, "B.Tech / BCA", "Campus / referrals"),
    ("Chartered Accountant (firm)", 85000, 60000, 250000, "CA", "ICAI + firm"),
    ("Doctor (private hospital)", 110000, 70000, 400000, "MBBS", "NEET-PG pathway"),
    ("Team Lead", 100000, 70000, 160000, "Graduate", "Promotion"), ("Manager", 150000, 100000, 260000, "Graduate / MBA", "Promotion"),
    ("Senior Manager", 220000, 160000, 350000, "MBA preferred", "Promotion"), ("Director (company)", 400000, 250000, 700000, "MBA / experience", "Promotion"),
    ("Vice President (company)", 550000, 350000, 1000000, "MBA / experience", "Promotion"),
    ("CFO", 900000, 500000, 2500000, "CA / MBA", "Board appointment"),
    ("CEO (mid-size company)", 1200000, 500000, 6000000, "Experience", "Board appointment"),
    ("CEO (large listed company)", 4000000, 1500000, 15000000, "Experience", "Board appointment; huge variance")]:
    add(n, PC, t, lo, hi, q, r)
# --- Skilled trades (daily rate x 26 days) ---
for n, d, lo, hi in [("Carpenter", 900, 600, 1500), ("Plumber", 850, 600, 1400), ("Electrician", 900, 650, 1500),
    ("Mason", 800, 550, 1300), ("Painter", 700, 500, 1100), ("Welder", 900, 600, 1500), ("AC / appliance technician", 1000, 700, 1800),
    ("Tailor", 600, 400, 1100)]:
    add(n, TR, d * 26, lo * 26, hi * 26, "Skill / ITI / apprenticeship", "Work under a contractor, then independent")
for n, t, lo, hi in [("Truck driver", 26000, 18000, 40000), ("Cook / chef (restaurant)", 22000, 14000, 60000),
    ("Security guard", 16000, 12000, 25000)]:
    add(n, TR, t, lo, hi, "8th-12th pass / training", "Agency or direct hire")
# --- Informal & daily wage ---
add("MGNREGA worker (100-day cap)", IN, 300 * 100 / 12, 234 * 100 / 12, 374 * 100 / 12, "None", "Job card at gram panchayat (approx. wage range)", 100 / 12)
for n, d, lo, hi in [("Farm labourer", 380, 300, 550), ("Construction helper", 550, 450, 750), ("Loader / hamal", 550, 450, 800)]:
    add(n, IN, d * 26, lo * 26, hi * 26, "None", "Daily labour chowk / contractor", 26)
add("Domestic worker", IN, 11000, 6000, 20000, "None", "Households, multi-home work")
add("Street vendor (net)", IN, 14000, 8000, 30000, "None", "Own stall; vending certificate")
# --- Gig & self-employed ---
for n, t, lo, hi in [("Delivery rider (gig)", 22000, 15000, 35000), ("Cab / auto driver (net)", 26000, 16000, 45000),
    ("Freelancer (junior)", 30000, 10000, 90000), ("Kirana shop owner (net)", 35000, 18000, 100000)]:
    add(n, GG, t, lo, hi, "Varies", "Own vehicle / shop / platform")

DF = pd.DataFrame(CATALOG)
BY = {r["name"]: r for r in CATALOG}
BOARD = ["Peon / MTS (Union)", "Carpenter", "Plumber", "Delivery rider (gig)", "Software Engineer", "Bank PO (IBPS / SBI)",
         "IAS: District Collector", "Police Sub-Inspector (MPSC)", "CEO (large listed company)", "Prime Minister", "President of India"]

def job_index(row: dict, days: int = 365) -> pd.Series:
    drift, vol, amp, peak, step = MOVE[row["family"]]
    ds = C._dates(days)
    r = np.array([drift + vol * C._z(row["name"], d) + (step if (d.month, d.day) in ((1, 1), (7, 1)) else 0) for d in ds])
    season = np.array([1 + amp * np.cos(2 * np.pi * (d.timetuple().tm_yday - peak) / 365) for d in ds])
    s = pd.Series(np.cumprod(1 + r) * season, index=pd.to_datetime(ds))
    return s / s.iloc[-1]

def series(name: str, days: int = 90) -> pd.Series:
    row = BY[name]; return row["typical"] * job_index(row, days)

def live_pay(row: dict, now: datetime) -> float:
    if row["family"] in FIXED: return row["typical"]
    return row["typical"] * (1 + 0.0008 * C._z(row["name"], int(now.timestamp() // 3)))

def day_change(row: dict) -> float:
    x = job_index(row, 3); return float(x.iloc[-1] / x.iloc[-2] - 1)

def stability(fam: str) -> str:
    return "Fixed, steps at DA/statutory revision" if fam in FIXED else ("Slow-moving" if fam == PS else
           "Moderate, appraisal-driven" if fam == PC else "Volatile and seasonal")

def projection(name: str, years: int = 10) -> list[float]:
    row = BY[name]; g = GROWTH[row["family"]]; t = row["typical"] * 12
    return list(np.cumsum([t * (1 + g) ** y for y in range(years)]))
