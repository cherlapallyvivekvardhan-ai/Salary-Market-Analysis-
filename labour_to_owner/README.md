# India Pay Ladder
Live pay board, 78-job ladder (daily wage to President), job explorer, compare-and-save, between-jobs planner, daily pipeline, plus a Company lab page (labour vs owner value split, fair-pay ML).

    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud (main file: app.py). Daily history: scripts/daily_snapshot.py + .github/workflows/daily.yml.
Private/trade/informal movement is simulated until a real feed replaces jobs.job_index(). Verify statutory and state pay against official notifications.
