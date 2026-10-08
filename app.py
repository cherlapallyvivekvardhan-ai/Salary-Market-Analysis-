"""National Occupational & Corporate Salary Market Intelligence System (India Edition)."""
import datetime as dt
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.express as px
import streamlit as st

st.set_page_config(page_title="India Salary Intelligence", page_icon="🧭", layout="wide")
BG, CARD, TXT, GREEN, AMBER, BLUE = "#0B0E14", "#161B22", "#F0F6FC", "#00E676", "#FFB300", "#2979FF"
CSS = f"""<style>
.stApp{{background:radial-gradient(1200px 600px at 10% -10%,#16264a55,transparent),radial-gradient(900px 500px at 100% 0,#00e67618,transparent),{BG};color:{TXT}}}
[data-testid=stSidebar]{{background:#0d1117;border-right:1px solid #ffffff12}}
.hero{{padding:28px 32px;border-radius:22px;background:linear-gradient(120deg,#2979ff33,#00e67622 60%,#ffb30018);border:1px solid #ffffff1f;backdrop-filter:blur(12px);margin-bottom:18px}}
.hero h1{{margin:0;font-size:2.1rem;letter-spacing:-.02em}} .hero p{{margin:.4rem 0 0;opacity:.75}}
.kpi{{background:{CARD}cc;border:1px solid #ffffff14;border-radius:18px;padding:16px 18px;backdrop-filter:blur(10px);transition:.25s}}
.kpi:hover{{transform:translateY(-3px);border-color:{BLUE}88;box-shadow:0 10px 30px #2979ff22}}
.kpi small{{opacity:.6;text-transform:uppercase;letter-spacing:.08em;font-size:.68rem}}
.kpi b{{display:block;font-size:1.6rem;margin-top:4px}} .g{{color:{GREEN}}} .a{{color:{AMBER}}} .b{{color:{BLUE}}}
.pulse{{display:inline-block;width:9px;height:9px;border-radius:50%;background:{GREEN};margin-right:7px;animation:p 1.6s infinite}}
@keyframes p{{0%{{box-shadow:0 0 0 0 #00e67699}}100%{{box-shadow:0 0 0 12px #00e67600}}}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

def style(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(color=TXT),
                      colorway=[BLUE, GREEN, AMBER, "#B388FF", "#FF5252", "#18FFFF"])
    return fig

def hero(title, sub, live=False):
    dot = '<span class="pulse"></span>' if live else ""
    st.markdown(f'<div class="hero"><h1>{dot}{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def kpis(items):
    for col, (l, v, c) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><small>{l}</small><b class="{c}">{v}</b></div>', unsafe_allow_html=True)

def inr(x):
    return f"₹{x/1e7:.2f} Cr" if x >= 1e7 else f"₹{x/1e5:.2f} L" if x >= 1e5 else f"₹{x:,.0f}"

# ------------------------------------------------------------------ data
SECTORS = {"Business Services": 533996, "Community, Personal & Social Services": 329483, "Trading (Wholesale & Retail)": 299016,
 "Construction & Civil Infrastructure": 160996, "Manufacturing (Metals & Chemicals)": 136025, "Real Estate & Commercial Renting": 106221,
 "Manufacturing (Machinery & Equipments)": 97003, "Agriculture & Allied Activities": 93335, "Transport, Storage & Communications": 89152,
 "Financial Intermediation & Banking": 81680, "Manufacturing (Food & Beverages)": 66642, "Manufacturing (Textiles & Apparel)": 45761,
 "Electricity, Gas & Water Utilities": 36653, "Manufacturing (Miscellaneous)": 28406, "Paper, Publishing & Media": 20872,
 "Mining & Mineral Quarrying": 16147, "Manufacturing (Leather)": 4579, "Manufacturing (Wood & Furniture)": 4396,
 "Insurance & Pension Funding": 1396, "Unclassified Corporate Operations": 3051}
FIVE = {"Primary (extraction)": "Agriculture & Allied Activities|Mining & Mineral Quarrying",
 "Secondary (industry)": "Construction & Civil Infrastructure|Manufacturing (Metals & Chemicals)|Manufacturing (Machinery & Equipments)|Manufacturing (Food & Beverages)|Manufacturing (Textiles & Apparel)|Manufacturing (Miscellaneous)|Manufacturing (Leather)|Manufacturing (Wood & Furniture)|Electricity, Gas & Water Utilities|Paper, Publishing & Media",
 "Tertiary (services)": "Trading (Wholesale & Retail)|Transport, Storage & Communications|Financial Intermediation & Banking|Insurance & Pension Funding|Real Estate & Commercial Renting|Community, Personal & Social Services",
 "Quaternary (knowledge)": "Business Services", "Quinary (sovereign)": "Unclassified Corporate Operations"}
STATES = {"Maharashtra": 401886, "Delhi": 274958, "Uttar Pradesh": 190342, "West Bengal": 158010, "Karnataka": 150806,
          "Tamil Nadu": 143958, "Telangana": 132010, "Gujarat": 121419}
CPC = [(1,18000,56900,"Peon / MTS / Safaiwala"),(2,19900,63200,"LDC / DEO-A"),(3,21700,69100,"Postal Asst / CAPF Constable"),
 (4,25500,81100,"UDC / Head Constable"),(5,29200,92300,"Jr Accountant / Steno-I"),(6,35400,112400,"Sub-Inspector / PRT"),
 (7,44900,142400,"ASO / TGT"),(8,47600,151100,"AAO / PGT"),(9,53100,167800,"Under Secretary"),(10,56100,177500,"IAS / IPS / IFS entry"),
 (11,67700,208700,"Deputy Secretary / Exec Engineer"),(12,78800,209200,"Supdt Engineer / Director (field)"),
 (13,123100,215900,"Director (Centre) / DIG"),(14,144200,218200,"Joint Secretary / IG Police"),(15,182200,224100,"Addl Secretary"),
 (16,205400,224400,"Special Secretary / DG"),(17,225000,225000,"Secretary to GoI / Chief Secretary"),(18,250000,250000,"Cabinet Secretary")]
CPSE = [("E0",30e3,120e3,"Executive Trainee",9.5,13),("E1",40e3,140e3,"Asst Manager / MT",12,17.5),("E2",50e3,160e3,"Deputy Manager",15.5,22),
 ("E3",60e3,180e3,"Manager",18.5,26),("E4",70e3,200e3,"Senior Manager",22.5,31),("E5",80e3,220e3,"Chief Manager",26,36),
 ("E6",90e3,240e3,"AGM",30,42),("E7",100e3,260e3,"DGM",35,49),("E8",120e3,280e3,"GM",43,60),("E9",150e3,300e3,"Executive Director",54,78),
 ("Director",180e3,340e3,"Board Director",70,95),("CMD",200e3,370e3,"CMD",80,125)]
SOVEREIGN = {"President":500000,"Vice President":400000,"Chief Minister – Telangana":410000,"Chief Minister – Delhi":390000,
 "Chief Minister – Uttar Pradesh":365000,"Chief Minister – Maharashtra":340000,"Chief Minister – Tripura":105500,"Prime Minister":280000,
 "Chief Justice of India":280000,"Supreme Court Judge":250000,"High Court Chief Justice":250000,"High Court Judge":225000,
 "MP (gross)":286000,"MLA – Maharashtra":260000,"MLA – Telangana":250000,"MLA – Kerala":70000}
CAPTIERS = {"Large-Cap (>₹1,06,346 Cr)":(8.5,22,32,70,110,280,1200,8500,2e5),"Mid-Cap (₹33,664–1,06,346 Cr)":(6,14,22,45,65,140,400,1800,5e4),
 "Small-Cap (<₹33,664 Cr)":(4.2,8.5,14,28,38,75,150,600,1e4),"Unlisted growth / VC-backed":(7,18,25,55,70,150,120,450,2e3),
 "Micro & MSME (<₹50 Cr)":(2.16,4.2,6,12,14,24,18,45,50)}
STAGES = [("Unskilled manual labourer",.10),("Semi-skilled operator",.28),("Skilled tradesman / technician",.52),("Entry corporate analyst",.95),
          ("Mid-tier specialist / manager",1.30),("Senior corporate VP",1.85),("Chief Executive Officer",2.45),("Founder / owner",3.80)]
ZONE_WF = {"Area A (metro)":821*26*12,"Area B (secondary urban)":693*26*12,"Area C (rural / small town)":556*26*12}
QUAL = {"Below 10th":0.0,"12th pass / ITI":.07,"Graduate":.25,"Professional (CA/CS/CMA/MBBS)":.80,"Postgraduate / MBA":.45,"IIT / IIM / elite":1.20}
SECTOR_MULT = {"Primary":.70,"Secondary":.95,"Tertiary":1.00,"Quaternary":1.45,"Quinary":1.10}

@st.cache_data
def cpc_gross(start, ceiling, step, da, cls, ta):
    basic = min(start * 1.03 ** step, ceiling)
    hra = basic * {"X":.30,"Y":.20,"Z":.10}[cls]
    return basic, basic*da/100, hra, ta*(1+da/100)

def sml(wf, beta, wm, theta, cap_cr, rev_hc_lakh, phi=9000, psi=18000):
    return wf + beta*(wm-wf) + theta*wf + phi*np.log(max(cap_cr,1)) + psi*rev_hc_lakh

def gamma(t, sigma, lam, tau=200, seed=7):
    xi = np.random.default_rng(seed).standard_normal(len(t))
    return sigma*np.cos(2*np.pi*(t-tau)/365) + lam*xi

def tax_new(gross):
    ti = max(gross-75000, 0)
    slabs = [(4e5,0),(8e5,.05),(12e5,.10),(16e5,.15),(20e5,.20),(24e5,.25),(1e18,.30)]
    tax, lo = 0.0, 0.0
    for hi, r in slabs:
        if ti > lo: tax += (min(ti, hi)-lo)*r
        lo = hi
    if ti <= 12e5: tax = 0.0
    else: tax = min(tax, ti-12e5)
    return tax*1.04, ti

def sip(sm, rate, years):
    r, n = rate/1200, years*12
    return sm*(((1+r)**n-1)/r)*(1+r)

# ------------------------------------------------------------------ pages
def macro():
    hero("Macro Intelligence", "Five-sector labour stratification anchored to the MCA registry (Aug 2026)", True)
    kpis([("Registered (ever)","31,92,605","b"),("Active companies","21,54,810","g"),("Struck off / closed","10,37,795","a"),("Global enterprises","≈360 M","b")])
    st.write("")
    rows = [(s, k, v) for s, v in SECTORS.items() for k, names in FIVE.items() if s in names.split("|")]
    df = pd.DataFrame(rows, columns=["Activity","Sector","Active"])
    a, b = st.columns([1.2, 1])
    a.plotly_chart(style(px.treemap(df, path=["Sector","Activity"], values="Active", color="Active", color_continuous_scale=["#0d2a66",BLUE,GREEN], title="Active enterprises by five-sector tier")), use_container_width=True)
    g = df.groupby("Sector", as_index=False).Active.sum()
    b.plotly_chart(style(px.pie(g, names="Sector", values="Active", hole=.62, title="Share of active registry")), use_container_width=True)
    s = pd.Series(STATES).reset_index(); s.columns = ["State","Active"]
    st.plotly_chart(style(px.bar(s, x="State", y="Active", color="Active", color_continuous_scale=["#0d2a66",BLUE,GREEN], title="Top corporate states"), 340), use_container_width=True)
    st.info("Tertiary + quaternary activity (services, trade, business services) is the majority of active companies; the top-8 states hold ~75% of the registry.")

def explorer():
    hero("Corporate Registry Explorer", "2,154,810 active companies — pre-aggregated counts plus an on-demand parametric cohort generator")
    c = st.columns(3)
    sec = c[0].selectbox("Economic activity", list(SECTORS)); stt = c[1].selectbox("State", list(STATES)); tier = c[2].selectbox("Scale tier", list(CAPTIERS))
    est = int(SECTORS[sec]*STATES[stt]/2154810)
    kpis([("Sector active",f"{SECTORS[sec]:,}","b"),("State active",f"{STATES[stt]:,}","g"),("Est. sector × state",f"{est:,}","a")])
    n = st.slider("Sample cohort size", 50, 2000, 400, 50)
    t = CAPTIERS[tier]; rng = np.random.default_rng(abs(hash((sec, stt, tier))) % 2**32)
    hc = rng.lognormal(np.log((t[6]*t[7])**.5), .6, n).astype(int)+5
    sal = rng.normal(t[2], t[2]*.18, n).clip(t[0]*.7)
    cap = rng.lognormal(np.log(max(t[8]/20, 1)), 1.0, n)
    df = pd.DataFrame({"Company":[f"{stt[:3].upper()}-{i:05d}" for i in range(n)],"Headcount":hc,"Median CTC (₹ L)":sal.round(2),"Paid-up / cap (₹ Cr)":cap.round(1)})
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(df, x="Median CTC (₹ L)", nbins=40, title="Salary distribution of cohort"), 340), use_container_width=True)
    b.plotly_chart(style(px.scatter(df, x="Headcount", y="Median CTC (₹ L)", size="Paid-up / cap (₹ Cr)", log_x=True, opacity=.7, title="Scale premium"), 340), use_container_width=True)
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)
    st.caption("Synthetic, parametric micro-sample calibrated to tier ranges — not individual company data.")

def deepdive():
    hero("Occupational Deep-Dive Engine", "E[W] = W_f(Z) + β(W_m − W_f) + Σθ·Q + φ·ln(Cap) + ψ·Rev/HC + Γ(t)", True)
    c = st.columns(4)
    zone = c[0].selectbox("Zone", list(ZONE_WF)); stage = c[1].selectbox("Occupational stratum", [s for s,_ in STAGES], 3)
    sector = c[2].selectbox("Sector", list(SECTOR_MULT), 2); qual = c[3].selectbox("Qualification", list(QUAL), 2)
    d = st.columns(3)
    cap = d[0].number_input("Employer capitalisation (₹ Cr)", 10, 2000000, 5000); rev = d[1].number_input("Revenue / employee (₹ L)", 1.0, 500.0, 25.0)
    wm = d[2].number_input("Median market comp (₹ L / yr)", 3.0, 60.0, 12.0)*1e5*SECTOR_MULT[sector]
    beta = dict(STAGES)[stage]; wf = ZONE_WF[zone]
    base = sml(wf, beta, wm, QUAL[qual], cap, rev/100)
    t = np.arange(90); g = gamma(t, .10*(1-min(beta,1.5)/3), .04)
    daily = base/300*(1+g)
    today = daily[-1]
    kpis([("Expected annual",inr(base),"g"),("Monthly",inr(base/12),"b"),("Today's spot daily",inr(today),"a"),("Human-capital β",f"{beta:.2f}","b")])
    f = go.Figure(go.Scatter(x=[dt.date.today()-dt.timedelta(days=int(89-i)) for i in t], y=daily, fill="tozeroy", line=dict(color=GREEN), fillcolor="rgba(0,230,118,.10)"))
    st.plotly_chart(style(f.update_layout(title="90-day daily spot wage — seasonal cosine + stochastic shock Γ(t)"), 360), use_container_width=True)
    comp = pd.DataFrame({"Component":["Zone floor W_f","β·(W_m−W_f)","Qualification θ·W_f","Scale φ·ln(Cap)","Productivity ψ·Rev/HC"],
        "₹/yr":[wf,beta*(wm-wf),QUAL[qual]*wf,9000*np.log(cap),18000*rev/100]})
    st.plotly_chart(style(px.bar(comp, x="Component", y="₹/yr", color="Component", title="Wage build-up"), 320), use_container_width=True)

def l2o():
    hero("Labour → Owner Wealth Simulator", "Eight-stage climb along the Human Capital Security Market Line")
    zone = st.selectbox("Starting zone", list(ZONE_WF)); wf = ZONE_WF[zone]; wm = 12e5
    rows = [(n, b, sml(wf, b, wm, [0,.07,.15,.25,.35,.60,.90,1.2][i], [1,1,50,2000,10000,60000,150000,500000][i], .05*(i+1)), ) for i,(n,b) in enumerate(STAGES)]
    df = pd.DataFrame(rows, columns=["Stage","Beta","Annual"]); df["Annual (₹ L)"] = (df.Annual/1e5).round(2)
    f = go.Figure(go.Scatter(x=df.Beta, y=df["Annual (₹ L)"], mode="lines+markers+text", text=df.Stage, textposition="top center", line=dict(color=BLUE, width=3), marker=dict(size=12, color=GREEN)))
    st.plotly_chart(style(f.update_layout(xaxis_title="Systematic human-capital β", yaxis_title="₹ lakh / yr", title="Human Capital SML", yaxis_type="log"), 480), use_container_width=True)
    i = st.select_slider("Your stage", [n for n,_ in STAGES], STAGES[3][0]); r = df[df.Stage==i].iloc[0]
    kpis([("Stage β",f"{r.Beta:.2f}","b"),("Expected annual",inr(r.Annual),"g"),("Risk profile","Wage-certain" if r.Beta<.6 else "Mixed" if r.Beta<1.6 else "Equity-driven","a")])
    st.dataframe(df[["Stage","Beta","Annual (₹ L)"]], hide_index=True, use_container_width=True)

def parity():
    hero("Sovereign & Public Pay Parity", "7th CPC · DPE CPSE · Constitutional posts vs. private market")
    t1, t2, t3, t4 = st.tabs(["7th CPC engine","CPSE E0–E9","Sovereign posts","Public vs private"])
    with t1:
        c = st.columns(4)
        lv = c[0].selectbox("Pay level", [x[0] for x in CPC], 9); step = c[1].slider("Years of service", 0, 39, 5)
        da = c[2].slider("Dearness Allowance %", 0, 80, 55); cls = c[3].selectbox("HRA city class", ["X","Y","Z"])
        _, s, e, d = next(x for x in CPC if x[0]==lv)
        ta = 1350 if lv<=2 else 3600 if lv<=8 else 7200
        b, da_amt, hra, ta_amt = cpc_gross(s, e, step, da, cls, ta)
        gross = b+da_amt+hra+ta_amt
        kpis([("Basic",inr(b),"b"),("DA",inr(da_amt),"g"),("HRA + TA",inr(hra+ta_amt),"a"),("Gross / month",inr(gross),"g")])
        st.caption(f"{d} — annual cost to govt ≈ {inr(gross*12)} (excludes pension/NPS employer share). TA band is a simplified level-based assumption.")
        m = pd.DataFrame([(x[0], cpc_gross(x[1],x[2],0,da,cls,1350)[0]*(1+da/100+{"X":.3,"Y":.2,"Z":.1}[cls])) for x in CPC], columns=["Level","Gross"])
        st.plotly_chart(style(px.bar(m, x="Level", y="Gross", color="Gross", color_continuous_scale=["#0d2a66",BLUE,GREEN], title="Entry gross by level"), 320), use_container_width=True)
    with t2:
        df = pd.DataFrame(CPSE, columns=["Grade","Min","Max","Designation","CTC low (₹L)","CTC high (₹L)"])
        f = go.Figure([go.Bar(x=df.Grade, y=df["CTC low (₹L)"], name="Low", marker_color=BLUE), go.Bar(x=df.Grade, y=df["CTC high (₹L)"]-df["CTC low (₹L)"], name="Range", marker_color=GREEN)])
        st.plotly_chart(style(f.update_layout(barmode="stack", title="CPSE CTC band (₹ lakh)"), 360), use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
    with t3:
        s = pd.Series(SOVEREIGN).sort_values().reset_index(); s.columns = ["Post","Monthly"]
        st.plotly_chart(style(px.bar(s, x="Monthly", y="Post", orientation="h", color="Monthly", color_continuous_scale=["#0d2a66",BLUE,AMBER], title="Monthly emoluments (₹)"), 560), use_container_width=True)
    with t4:
        lv = st.selectbox("Compare level", [x[0] for x in CPC], 9, key="cmp"); tier = st.selectbox("Private tier", list(CAPTIERS), key="ct")
        g_ = cpc_gross(next(x for x in CPC if x[0]==lv)[1], 9e9, 0, 55, "X", 3600); gov = sum(g_)*12/1e5
        pri = CAPTIERS[tier][:2]
        f = go.Figure(go.Bar(x=["Govt entry (gross)","Private entry low","Private entry high"], y=[gov, pri[0], pri[1]], marker_color=[GREEN,BLUE,AMBER]))
        st.plotly_chart(style(f.update_layout(title="Annual ₹ lakh — public entry vs private entry"), 340), use_container_width=True)

def wealth():
    hero("Personal Wealth Accelerator", "New Tax Regime (115BAC) · multi-income · SIP compounding")
    c = st.columns(3)
    g = c[0].number_input("Primary gross income (₹/yr)", 0, 100000000, 1500000, 50000)
    sec = c[1].number_input("Secondary active income (₹/yr)", 0, 50000000, 120000, 10000)
    save = c[2].slider("Invest % of in-hand", 5, 60, 20)
    tx, ti = tax_new(g+sec); inhand = g+sec-tx
    kpis([("Taxable income",inr(ti),"b"),("Tax + cess",inr(tx),"a"),("Effective rate",f"{tx/max(g+sec,1)*100:.2f}%","a"),("In-hand / month",inr(inhand/12),"g")])
    sm = inhand*save/100/12; st.write(""); st.markdown(f"**Monthly SIP:** {inr(sm)}")
    yrs = np.arange(1, 21); f = go.Figure()
    for nm, r, col in [("Conservative 8%",8,BLUE),("Balanced 11%",11,GREEN),("Aggressive 14%",14,AMBER)]:
        f.add_trace(go.Scatter(x=yrs, y=[sip(sm, r, y)/1e5 for y in yrs], name=nm, line=dict(color=col, width=3)))
    st.plotly_chart(style(f.update_layout(title="SIP corpus (₹ lakh)", xaxis_title="Years"), 400), use_container_width=True)
    st.caption("Zero tax up to ₹12.75 L gross for salaried (₹75k std deduction + 87A rebate). Surcharge above ₹50 L is not modelled. Illustrative, not financial advice.")

pg = st.navigation([st.Page(macro, title="Macro Overview", icon="🌐", url_path="macro", default=True),
    st.Page(explorer, title="Corporate Explorer", icon="🏢", url_path="explorer"),
    st.Page(deepdive, title="Occupational Deep-Dive", icon="🧮", url_path="deepdive"),
    st.Page(l2o, title="Labour → Owner", icon="📈", url_path="labour-to-owner"),
    st.Page(parity, title="Govt Parity Engine", icon="🏛️", url_path="parity"),
    st.Page(wealth, title="Wealth Accelerator", icon="💰", url_path="wealth")])
st.sidebar.caption("Data: MCA Aug 2026 · 7th CPC · DPE 3rd PRC · Central minimum wages Apr 2026")
pg.run()
