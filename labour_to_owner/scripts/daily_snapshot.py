"""Run daily (cron / GitHub Actions): stores today's pay for every job and appends data/history.csv."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, jobs as J, db, core as C
pays = {r["name"]: float(J.series(r["name"], 2).iloc[-1]) for r in J.CATALOG}
n = db.save_snapshot(C.TODAY, pays)
out = Path(__file__).resolve().parents[1] / "data" / "history.csv"
new = pd.DataFrame({"day": str(C.TODAY), "job": list(pays), "pay": list(pays.values())})
if out.exists():
    old = pd.read_csv(out); new = pd.concat([old[old.day != str(C.TODAY)], new])
new.to_csv(out, index=False); print(f"saved {len(pays)} jobs for {C.TODAY} ({n} new db rows)")
