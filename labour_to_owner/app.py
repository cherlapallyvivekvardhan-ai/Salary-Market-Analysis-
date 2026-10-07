from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import core as C, db, jobs as J

IST = ZoneInfo("Asia/Kolkata")
st.set_page_config(page_title="India Pay Ladder", page_icon="₹", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600&family=Roboto+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] {font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3 {font-family:'Oswald',sans-serif; letter-spacing:.01em; font-weight:600;}
.block-container {padding-top:1.2rem; max-width:1280px;}
.hero h1 {font-size:3.1rem; line-height:1.05; margin:0;} .hero p {font-size:1.1rem; max-width:62ch; margin:.4rem 0 1rem;}
.board {background:#0B1F2A; border-radius:10px; padding:.9rem 1.1rem; border-bottom:6px solid #E0A100;}
.bh {display:flex; justify-content:space-between; color:#9fb8c2; font-family:'Oswald'; font-size:.95rem; padding-bottom:.4rem; border-bottom:1px solid #254655;}
.row {display:grid; grid-template-columns:2.3fr 1.7fr 2.3fr .9fr .7fr; align-items:center; gap:.5rem; padding:.28rem 0; border-bottom:1px solid #153240;}
.nm {color:#fff; font-family:'Oswald'; font-size:1.05rem;} .fm {font-size:.8rem; padding:.1rem .45rem; border-radius:4px; color:#fff; width:fit-content;}
.fl span {display:inline-block; width:1.15ch; text-align:center; margin:0 1px; padding:.1rem .12rem; color:#FFC93C; font:500 1.25rem 'Roboto Mono',monospace;
  background:linear-gradient(#1b3d4e 49%, #0a1a23 51%); border-radius:3px;}
.fl span.c {width:.7ch; background:none; color:#9fb8c2;}
.fl span.ch {animation:flip .45s ease-out;} @keyframes flip {from {transform:rotateX(90deg);} to {transform:rotateX(0);}}
.up {color:#4ade80;} .dn {color:#f87171;} .fx {color:#9fb8c2;} .chg {font:500 .95rem 'Roboto Mono',monospace;}
@media (prefers-reduced-motion: reduce) {.fl span.ch {animation:none;}}
[data-testid="stMetric"] {background:#fff; border-left:4px solid #0F5C5E; padding:.6rem .9rem; border-radius:4px;}
</style>""", unsafe_allow_html=True)

inr = lambda v: f"₹{v:,.0f}"
now0 = datetime.now(IST)
st.session_state.setdefault("snap", db.save_snapshot(C.TODAY, {r["name"]: float(J.series(r["name"], 2).iloc[-1]) for r in J.CATALOG}))

st.markdown(f"""<div class="hero"><h1>India Pay Ladder</h1>
<p>What every kind of work pays, from a daily-wage labourer to the President. {len(J.CATALOG)} jobs, {len(J.FAMILIES)} sectors,
refreshed every 3 seconds on the board and every day in the saved history. Private, trade and informal pay is simulated until you plug in a live feed;
government pay follows the 7th Pay Commission and moves only when allowances are revised.</p></div>""", unsafe_allow_html=True)

@st.fragment(run_every=3)
def board():
    now = datetime.now(IST); prev = st.session_state.get("prev", {}); cur = {}; rows = ""
    for n in J.BOARD:
        r = J.BY[n]; s = f"{J.live_pay(r, now):,.0f}"; p = prev.get(n, s).rjust(len(s)); cur[n] = s
        cells = "".join(f'<span class="c">{ch}</span>' if ch == "," else f'<span class="{"ch" if ch != p[i] else ""}">{ch}</span>' for i, ch in enumerate(s))
        fixed = r["family"] in J.FIXED; ch_ = J.day_change(r)
        chg = '<span class="chg fx">FIXED</span>' if fixed else f'<span class="chg {"up" if ch_ >= 0 else "dn"}">{"▲" if ch_ >= 0 else "▼"}{abs(ch_)*100:.2f}%</span>'
        rows += (f'<div class="row"><div class="nm">{n}</div><div><span class="fm" style="background:{J.FAMILIES[r["family"]]}">{r["family"]}</span></div>'
                 f'<div class="fl">₹ {cells}</div><div>{chg}</div><div class="fx">{"statutory" if fixed else "live"}</div></div>')
    st.session_state["prev"] = cur
    st.markdown(f'<div class="board"><div class="bh"><span>Monthly pay now</span><span>{now:%d %b %Y, %H:%M:%S} IST</span></div>{rows}</div>', unsafe_allow_html=True)
board()
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["Pay ladder", "Job explorer", "Compare and save", "Between jobs", "Daily data pipeline"])

with t1:
    fams = st.multiselect("Sectors", list(J.FAMILIES), default=list(J.FAMILIES))
    d = J.DF[J.DF.family.isin(fams)].sort_values("typical")
    f = px.bar(d, x="typical", y="name", color="family", color_discrete_map=J.FAMILIES, orientation="h", log_x=True,
               hover_data={"low": ":,.0f", "high": ":,.0f", "typical": ":,.0f", "family": False},
               labels={"typical": "Typical monthly pay (₹, log scale)", "name": ""})
    f.update_layout(height=max(420, 20 * len(d) + 120), margin=dict(t=10), legend_orientation="h", legend_y=1.04, legend_title="")
    st.plotly_chart(f, use_container_width=True)
    st.caption("Log scale: each gridline is 10 times the last, so a farm labourer and a President fit on one chart. Constitutional and judicial pay is approximate; verify against official notifications.")

with t2:
    a, b = st.columns([1, 2])
    fam = a.selectbox("Sector", ["All"] + list(J.FAMILIES)); q = a.text_input("Search job")
    pool = J.DF[(J.DF.family == fam) | (fam == "All")]
    pool = pool[pool.name.str.contains(q, case=False)] if q else pool
    if pool.empty: st.warning("No job matches. Clear the search box."); st.stop()
    name = b.selectbox("Job", pool.name); r = J.BY[name]
    k = st.columns(4); k[0].metric("Typical / month", inr(r["typical"])); k[1].metric("Per working day", inr(r["typical"] / r["days"]))
    k[2].metric("Per hour (8h day)", inr(r["typical"] / r["days"] / 8)); k[3].metric("Range / month", f"{inr(r['low'])} to {inr(r['high'])}")
    st.write(f"**Qualification:** {r['qualification']}   |   **How to get in:** {r['route']}   |   **Pay behaviour:** {J.stability(r['family'])}")
    days = st.radio("Window", [30, 90, 365], index=1, horizontal=True, format_func=lambda x: f"{x} days")
    s = J.series(name, days)
    g = go.Figure(); g.add_hrect(y0=r["low"], y1=r["high"], fillcolor=J.FAMILIES[r["family"]], opacity=.08, line_width=0)
    g.add_scatter(x=s.index, y=s.values, mode="lines", line=dict(color=J.FAMILIES[r["family"]], width=3, shape="hv" if r["family"] in J.FIXED else "linear"), name="Daily pay")
    g.update_layout(height=340, margin=dict(t=10), yaxis_title="₹ per month", showlegend=False); st.plotly_chart(g, use_container_width=True)
    h = db.history(name)
    st.caption(f"Saved daily snapshots for this job: {len(h)} day(s)." + ("" if len(h) < 2 else ""))
    if len(h) > 1: st.line_chart(h.set_index("day"))
    lad = J.DF[J.DF.family == r["family"]].sort_values("typical")
    st.plotly_chart(px.bar(lad, x="name", y="typical", labels={"typical": "₹/month", "name": ""}, title=f"Career ladder in this sector",
                           color_discrete_sequence=[J.FAMILIES[r["family"]]]).update_layout(height=320), use_container_width=True)

with t3:
    saved = db.list_cmp()
    c1, c2 = st.columns([3, 1]); pre = c2.selectbox("Load saved set", [""] + list(saved))
    default = saved.get(pre) or ["Carpenter", "Software Engineer", "IAS: District Collector", "Police Sub-Inspector (MPSC)"]
    picks = c1.multiselect("Jobs to compare (up to 6)", list(J.DF.name), default=default, max_selections=6)
    if not picks: st.info("Pick at least one job."); st.stop()
    nm = st.text_input("Name this comparison"); 
    if st.button("Save comparison") and nm: db.save_cmp(nm, picks); st.success(f"Saved '{nm}'. It appears in 'Load saved set'.")
    l, r_ = st.columns(2)
    fig = go.Figure()
    for n in picks: s = J.series(n, 90); fig.add_scatter(x=s.index, y=s / s.iloc[0] * 100, name=n, mode="lines")
    fig.update_layout(height=340, title="Last 90 days, rebased to 100", margin=dict(t=40), legend_orientation="h"); l.plotly_chart(fig, use_container_width=True)
    pj = go.Figure()
    for n in picks: pj.add_scatter(x=list(range(1, 11)), y=J.projection(n), name=n, mode="lines")
    pj.update_layout(height=340, title="Cumulative earnings over 10 years (₹)", xaxis_title="Year", margin=dict(t=40), showlegend=False); r_.plotly_chart(pj, use_container_width=True)
    tb = J.DF[J.DF.name.isin(picks)].assign(daily=lambda x: x.typical / x.days, hourly=lambda x: x.typical / x.days / 8, behaviour=lambda x: x.family.map(J.stability))
    st.dataframe(tb[["name", "family", "typical", "daily", "hourly", "low", "high", "behaviour"]], hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="₹%d") for k in ["typical", "daily", "hourly", "low", "high"]})
    st.caption("10-year projection assumes yearly growth by sector (government 3-5%, private 8%, trades 5%). It is a comparison tool, not a forecast.")

with t4:
    st.subheader("If you are between jobs")
    a, b, c = st.columns(3)
    sav = a.number_input("Savings (₹)", 0, 10_000_000, 100000, 5000); exp_ = b.number_input("Monthly expenses (₹)", 1000, 500000, 20000, 1000)
    tgt = c.selectbox("Target", list(J.FAMILIES))
    months = {J.UG: (12, 30), J.SG: (10, 24), J.DP: (8, 20), J.JC: (24, 60), J.PS: (8, 18), J.PC: (2, 5), J.TR: (0.5, 1.5), J.IN: (0.1, 0.5), J.GG: (0.1, 0.5)}[tgt]
    run = sav / exp_
    k = st.columns(3); k[0].metric("Savings last", f"{run:.1f} months"); k[1].metric("Typical time to land a job (simulated)", f"{months[0]} to {months[1]} months")
    k[2].metric("Gap to cover", "None" if run >= months[1] else f"{(months[1] - run) * exp_:,.0f} ₹")
    bridge = J.DF[(J.DF.family.isin([J.TR, J.IN, J.GG])) & (J.DF.typical > 0)].sort_values("typical", ascending=False).head(6)
    st.write("**Income bridges you can start within days** (while preparing for the target):")
    st.dataframe(bridge[["name", "typical", "route"]], hide_index=True, use_container_width=True, column_config={"typical": st.column_config.NumberColumn("Typical ₹/month", format="₹%d")})
    st.caption("Time-to-hire ranges are illustrative: government exams run on yearly cycles, private hiring on weeks. Replace with PLFS/portal data for real figures.")

with t5:
    c = st.columns(3); c[0].metric("Jobs tracked", len(J.CATALOG)); c[1].metric("Days saved in database", db.days_stored()); c[2].metric("Sectors", len(J.FAMILIES))
    st.markdown("""
**How data arrives, day by day**
1. `scripts/daily_snapshot.py` runs every morning (GitHub Actions workflow included) and stores every job's pay in SQLite and `data/history.csv`.
2. The app also saves a snapshot the first time it opens each day, so history builds even without the workflow.
3. Charts read this history. Streamlit Cloud's disk resets on redeploy, so keep `data/history.csv` in Git, or point `db.py` at Supabase/Postgres.

**Where real data plugs in**
- Union government: official pay-matrix and DA circulars (Dept. of Expenditure). State: Maharashtra finance department GRs.
- Private and trades: job-board listings, payroll or survey data. Replace `jobs.job_index()` and keep everything else.
- Informal wages: Labour Bureau rural wage rates, PLFS, MGNREGA notified rates.

**What is simulated:** private, trade, informal and gig wage movement and the 3-second board tick. **What is not:** government levels, the DA mechanism, the economics.
""")
