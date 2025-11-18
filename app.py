from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from models import *
from recommender import recommend
import progress as prg

app = FastAPI(title="PulseFit API")

# إعداد الـCORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    try:
        from pathlib import Path
        import numpy as np
        import os

        data_dir = Path("data")
        parquet_exists = (data_dir / "prog_df.parquet").exists()
        npz_exists = (data_dir / "programs_features.npz").exists()

        if parquet_exists:
            import pandas as pd
            df = pd.read_parquet(data_dir / "prog_df.parquet")
            rows = len(df)
        else:
            rows = 0

        if npz_exists:
            feats = np.load(data_dir / "programs_features.npz")
            shape = feats["features"].shape
        else:
            shape = (0, 0)

        return {
            "status": "ok",
            "parquet_exists": parquet_exists,
            "npz_exists": npz_exists,
            "program_rows": rows,
            "features_shape": shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ Recommendation ============
@app.post("/recommend", response_model=RecommendResponse)
def api_recommend(req: RecommendRequest):
    try:
        df = recommend(req.goal, req.days_per_week, req.level, req.top_n)
        items = []
        for _, r in df.iterrows():
            items.append(RecommendItem(
                title=str(r.get("title","")),
                goal=str(r.get("goal","")),
                level_list=list(r.get("level_list",[]) or []),
                days_per_week=int(r["days_per_week"]) if pd.notna(r.get("days_per_week")) else None,
                cosine_similarity=float(r["cosine_similarity"]),
                description=str(r.get("description",""))
            ))
        return RecommendResponse(items=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ Progress Tracking ============
@app.post("/progress/ingest", response_model=IngestResponse)
def api_ingest(req: IngestRequest):
    try:
        sdf = pd.DataFrame([{"user_id":req.user_id, "date":s.date, "effort":s.effort} for s in req.sessions])
        before_weeks = 0
        if req.user_id in prg.SESS_STORE:
            before_weeks = prg.level_from_sessions(prg.SESS_STORE[req.user_id], req.planned_per_week)["week"].nunique()
        prg.ingest_sessions(req.user_id, req.planned_per_week, sdf)
        after_weeks = prg.level_from_sessions(prg.SESS_STORE[req.user_id], req.planned_per_week)["week"].nunique()
        return IngestResponse(weeks_added=max(0, after_weeks - before_weeks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/progress/compute", response_model=ComputeProgressResponse)
def api_compute(req: ComputeProgressRequest):
    try:
        out, _ = prg.compute_progress(req.user_id, decision_period_weeks=req.decision_period_weeks)
        return ComputeProgressResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/progress/trajectory", response_model=TrajectoryResponse)
def api_traj(req: TrajectoryRequest):
    try:
        _, prog = prg.compute_progress(req.user_id, decision_period_weeks=4)
        if req.last_n_weeks > 0:
            prog = prog.tail(req.last_n_weeks)
        rows = []
        for _, r in prog.iterrows():
            rows.append(TrajectoryRow(
                week=str(r["week"])[:10],
                days_trained=int(r["days_trained"]),
                LevelScore=float(r["LevelScore"]),
                Readiness=float(r["Readiness"]),
                cum_readiness_mean=float(r["cum_readiness_mean"]),
                progress_bar=float(r["progress_bar"]),
                is_decision_week=bool(r["is_decision_week"]),
                level_final=str(r["level_final"]),
                level_progressive=str(r["level_progressive"])
            ))
        return TrajectoryResponse(weeks=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
