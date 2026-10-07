import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import core as C

st.set_page_config(page_title="Labour-to-Owner Salary Ledger", page_icon="₹", layout="wide")
TEAL, GOLD, INK = "#0F5C5E", "#E0A100", "#12302F"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] {font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3 {font-family:'Bricolage Grotesque',sans-serif; letter-spacing:-0.01em;}
.block-container {padding-top:1.6rem; max-width:1250px;}
.tick {display:flex; flex-wrap:wrap; gap:.4rem .5rem; margin:.4rem 0 1rem;}
.chip {background:#fff; border:1px solid #cfdcda; border-radius:6px; padding:.25rem .6rem; font-size:.85rem;}
.up {color:#0F7B4B; font-weight:500;} .dn {color:#B3261E; font-weight:500;}
[data-testid="stMetric"] {background:#fff; border-left:4px solid #0F5C5E; padding:.6rem .9rem; border-radius:4px;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def companies(): return C.load_companies()
@st.cache_resource
def model(): return C.train_model()
@st.cache_data
def ridx(role, days): return C.role_index(role, days)
@st.cache_data
def sidx(sec, days): return C.sector_index(sec, days)

inr = lambda v: f"₹{v:,.0f}"
co = companies()

st.title("Labour-to-Owner Salary Ledger")
st.caption(f"{C.MCA_ACTIVE_COMPANIES:,} active companies are registered with the MCA (Aug 2026). "
           f"This build simulates {len(co):,} of them. Figures are re-computed for {C.TODAY:%d %b %Y}.")

moves = sorted(((r, ridx(r, 3).iloc[-1] / ridx(r, 3).iloc[-2] - 1) for r in C.ROLES), key=lambda x: -abs(x[1]))[:10]
st.markdown('<div class="tick">' + "".join(
    f'<span class="chip">{r} <span class="{"up" if m >= 0 else "dn"}">{"▲" if m >= 0 else "▼"} {abs(m)*100:.2f}%</span></span>'
    for r, m in moves) + "</div>", unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6 = st.tabs(["Market today", "Company", "Is my pay fair?", "Qualification", "Government jobs", "Method"])

with t1:
    span = st.radio("Window", [30, 90, 180, 365], index=1, horizontal=True, format_func=lambda d: f"{d} days")
    secs = st.multiselect("Sectors", list(C.SECTORS), default=list(C.SECTORS))
    fig = go.Figure()
    for s in secs:
        x = sidx(s, span); fig.add_scatter(x=x.index, y=x / x.iloc[0] * 100, name=s, mode="lines")
    fig.update_layout(height=340, yaxis_title="Wage index (start = 100)", margin=dict(t=10), legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    rows = []
    for r, (sec, _) in C.ROLES.items():
        if sec in secs:
            x = ridx(r, 60)
            rows.append({"Occupation": r, "Sector": sec, "Typical monthly pay": C.fair_salary(r, 5, "Graduate", "Bengaluru"),
                         "1-day %": (x.iloc[-1] / x.iloc[-2] - 1) * 100, "30-day %": (x.iloc[-1] / x.iloc[-31] - 1) * 100})
    st.subheader("Every occupation, today")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "Typical monthly pay": st.column_config.NumberColumn(format="₹%d", help="5 yrs experience, graduate, Bengaluru"),
        "1-day %": st.column_config.NumberColumn(format="%.2f"), "30-day %": st.column_config.NumberColumn(format="%.2f")})

with t2:
    q = st.text_input("Search company name or CIN", "")
    pool = co[co.name.str.contains(q, case=False) | co.cin.str.contains(q, case=False)] if q else co
    if pool.empty: st.warning("No company matches. Clear the search box to see all."); st.stop()
    pick = st.selectbox("Company", pool.index, format_func=lambda i: f"{co.name[i]}  |  {co.cin[i]}  |  {co.city[i]}")
    c = co.loc[pick]
    a, b, d = st.columns(3)
    exp = a.slider("Experience (years)", 0, 20, 5, key="e2")
    qual = b.selectbox("Qualification", list(C.QUALS), index=3, key="q2")
    role = d.selectbox("Occupation", [r for r, v in C.ROLES.items() if v[0] == c.sector])
    eco = C.company_economics(c, exp, qual)
    m = st.columns(4)
    m[0].metric("Headcount", f"{int(c.headcount):,}"); m[1].metric("Revenue", f"₹{c.revenue_cr:,.1f} Cr")
    m[2].metric("Labour share of value added", f"{eco['labour_share']:.0f}%")
    m[3].metric("Owner surplus / employee / yr", inr(eco["owner_surplus"]))
    l, r_ = st.columns([2, 1])
    s = C.company_daily(c, role, exp, qual, 90)
    f = px.line(x=s.index, y=s.values, labels={"x": "", "y": "Monthly pay (₹)"}, title=f"{role}: daily pay, last 90 days")
    f.update_traces(line_color=TEAL); f.update_layout(height=340, margin=dict(t=40)); l.plotly_chart(f, use_container_width=True)
    lab = min(eco["ctc"], eco["value_added"]); own = max(eco["value_added"] - eco["ctc"], 0)
    p = go.Figure(go.Pie(labels=["Workers (CTC)", "Owners (surplus)"], values=[lab, own], hole=.55, marker_colors=[TEAL, GOLD]))
    p.update_layout(height=340, title="Who gets the value an employee creates", margin=dict(t=40), showlegend=True)
    r_.plotly_chart(p, use_container_width=True)
    if eco["labour_share"] > 100: st.info("Wages exceed value added here: this company is running on reserves or funding.")
    st.caption(f"Value added per employee: {inr(eco['value_added'])}/yr, about {inr(eco['value_added'] / 300)} per working day.")

with t3:
    a, b, d, e = st.columns(4)
    role = a.selectbox("Occupation", list(C.ROLES), key="r3")
    exp = b.slider("Experience (years)", 0, 20, 4, key="e3")
    qual = d.selectbox("Qualification", list(C.QUALS), index=3, key="q3")
    city = e.selectbox("City", list(C.CITIES), key="c3")
    cc = st.columns(2)
    cur = cc[0].number_input("Your current monthly pay (₹)", 0, 2_000_000, 50000, 1000)
    sel = cc[1].selectbox("Company (optional)", [None] + list(co.index), format_func=lambda i: "Market average" if i is None else f"{co.name[i]} | {co.city[i]}")
    prem = 1.0 if sel is None else float(co.premium[sel])
    p10, p50, p90 = C.predict_band(model(), role, exp, qual, city, prem)
    k = st.columns(3); k[0].metric("Low (10th pct)", inr(p10)); k[1].metric("Fair market pay", inr(p50)); k[2].metric("High (90th pct)", inr(p90))
    gap = p50 - cur
    if cur < p10: st.error(f"You are below the market range by {inr(p10 - cur)}/month. Gap to fair pay: {inr(gap * 12)} a year.")
    elif cur > p90: st.success(f"You are above the usual range by {inr(cur - p90)}/month.")
    else: st.info(f"You are inside the market range. Distance from the midpoint: {inr(gap)}/month.")
    bar = go.Figure(go.Bar(x=[p10, p50, p90, cur], y=["Low", "Fair", "High", "You"], orientation="h", marker_color=[TEAL, TEAL, TEAL, GOLD]))
    bar.update_layout(height=240, margin=dict(t=10), xaxis_title="₹ per month"); st.plotly_chart(bar, use_container_width=True)

with t4:
    sec_avg = {s: np.mean([v[1] for v in C.ROLES.values() if v[0] == s]) / 1.35 * 1.0 for s in C.SECTORS}
    z = pd.DataFrame({qq: {s: sec_avg[s] * C.QUALS[qq] * 1.35 for s in C.SECTORS} for qq in C.QUALS})
    h = px.imshow(z, text_auto=",.0f", aspect="auto", color_continuous_scale=["#EEF3F2", TEAL], labels=dict(color="₹/month"))
    h.update_layout(height=380, margin=dict(t=10)); st.subheader("Typical monthly pay, 5 years experience, Tier-1 city")
    st.plotly_chart(h, use_container_width=True)
    ro = st.selectbox("Pay growth with experience for", list(C.ROLES), key="r4")
    curve = pd.DataFrame({"Years": range(0, 21), **{qq: [C.fair_salary(ro, y, qq, "Bengaluru") for y in range(21)] for qq in ["12th pass", "Graduate", "Post-graduate / MBA"]}})
    st.plotly_chart(px.line(curve, x="Years", y=curve.columns[1:], labels={"value": "₹/month", "variable": ""}), use_container_width=True)

with t5:
    post = st.selectbox("Post", list(C.GOV_POSTS)); lvl, need = C.GOV_POSTS[post]
    a, b, d = st.columns(3)
    da = a.slider("Dearness allowance %", 0, 80, 58) / 100
    hc = b.selectbox("City class (HRA)", list(C.HRA)); yrs = d.slider("Years of service", 0, 30, 0)
    g = C.gov_pay(lvl, da, hc, yrs)
    k = st.columns(4); k[0].metric("Pay level", f"Level {lvl}"); k[1].metric("Basic", inr(g["basic"]))
    k[2].metric("Gross / month", inr(g["gross"])); k[3].metric("Take-home (after NPS)", inr(g["net"]))
    ps = C.private_start(need)
    st.write(f"Qualification asked: **{need}**. A private-sector fresher with the same qualification earns about **{inr(ps)}** a month on average, "
             f"against **{inr(C.gov_pay(lvl, da, hc, 0)['gross'])}** gross for this post at entry, with job security and a pension scheme on top.")
    path = pd.DataFrame({"Year": range(0, 31), "Basic pay": [C.gov_pay(lvl, da, hc, y)["basic"] for y in range(31)]})
    st.plotly_chart(px.area(path, x="Year", y="Basic pay", color_discrete_sequence=[TEAL]), use_container_width=True)
    st.caption("Based on the 7th Pay Commission matrix with a flat 3% yearly increment. Check DA, HRA and the post's level against the latest official notification.")

with t6:
    st.markdown("""
**What is real, what is simulated**
- Real: the MCA company count, the 7th CPC pay-matrix levels, the economic formulas.
- Simulated: company records, daily wage movements and the salary training data. Same day, same numbers.
- To go live, drop an MCA-derived `data/mca_companies.csv` (columns: cin, name, sector, city, paid_up_cr, revenue_cr, headcount, premium) and replace `core.sector_index` with a job-board or payroll feed.

**Labour-to-owner maths** (per employee, per year)
- Value added = revenue per employee × sector value-added ratio
- Labour share = CTC ÷ value added; owner surplus = value added − CTC
- The earlier idea of dividing (MRPL − wage) by owner surplus is circular, because both are the same number. Labour share and owner share are the clean version.

**Fair-salary model**: a random forest on experience, qualification, city, occupation and company premium, shown as a 10th to 90th percentile band across trees.
""")
