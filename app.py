from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

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

# ========= Load Excel "Detailed" sheet once =========
DATASETS_DIR = Path("datasets")
DETAILED_PATH = DATASETS_DIR / "full_Programs_Summury.xlsx"

try:
    DETAILED_DF = pd.read_excel(DETAILED_PATH, sheet_name="Detailed")
    print(f"[Excel] Loaded {DETAILED_PATH} with {len(DETAILED_DF)} rows")
except Exception as e:
    print(f"[Excel] WARNING: could not load {DETAILED_PATH}: {e}")
    DETAILED_DF = None

def build_schedule_from_excel(program_title: str) -> dict:
    """
    Build a weekly schedule for a given program title using the 'Detailed' sheet.
    For now we use week = 1 and group by 'day'.
    """
    if DETAILED_DF is None:
        raise HTTPException(
            status_code=500,
            detail="Excel data not loaded on server",
        )

    df = DETAILED_DF
    prog = df[df["title"] == program_title]

    if prog.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Program '{program_title}' not found in Excel",
        )

    # Use only week 1 for now (change later if needed)
    if "week" in prog.columns:
        prog = prog[prog["week"] == 1]

    prog = prog.sort_values(["day"])

    meta = prog.iloc[0]

    schedule: dict = {
        "title": meta["title"],
        "description": meta["description"],
        "level": meta.get("level", ""),
        "program_length": int(meta.get("program_length", 0) or 0),
        "days_per_week": int(meta.get("days", 0) or 0),
        "days": [],
    }

    for day_value, day_df in prog.groupby("day"):
        day_df = day_df.sort_values(["exercise_name"])
        exercises = []

        for i, row in enumerate(day_df.itertuples(), start=1):
            intensity = row.intensity
            # clean NaNs
            if pd.isna(intensity):
                intensity = None

            exercises.append(
                {
                    "index": i,
                    "exercise_name": row.exercise_name,
                    "sets": int(row.sets),
                    "reps": int(row.reps),
                    "intensity": int(intensity) if intensity is not None else None,
                }
            )

        schedule["days"].append(
            {
                "dayIndex": int(day_value),
                "number_of_exercises": len(exercises),
                "exercises": exercises,
            }
        )

    schedule["days"].sort(key=lambda d: d["dayIndex"])
    return schedule


# ========= Models for schedule response =========
class Exercise(BaseModel):
    index: int
    exercise_name: str
    sets: int
    reps: int
    intensity: Optional[int] = None

class DayPlan(BaseModel):
    dayIndex: int
    number_of_exercises: int
    exercises: List[Exercise]

class Schedule(BaseModel):
    title: str
    description: str
    level: str
    program_length: int
    days_per_week: int
    days: List[DayPlan]

class RecommendWithScheduleResponse(BaseModel):
    program_title: str
    schedule: Schedule

# ============ Schedule Endpoint ============
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
    
# ============ Recommendation with schedule ============
@app.post("/recommend_with_schedule", response_model=RecommendWithScheduleResponse)
def api_recommend_with_schedule(req: RecommendRequest):
    try:
        # 1) Use existing ML recommender to get ranked programs
        df = recommend(req.goal, req.days_per_week, req.level, req.top_n)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No recommendation found")

        # 2) Pick the top program title
        top = df.iloc[0]
        program_title = str(top.get("title", ""))

        if not program_title:
            raise HTTPException(status_code=500, detail="Top recommendation has no title")

        # 3) Build schedule from Excel based on that title
        schedule_dict = build_schedule_from_excel(program_title)

        # 4) FastAPI will validate/convert dict → Schedule
        return RecommendWithScheduleResponse(
            program_title=program_title,
            schedule=schedule_dict,
        )
    except HTTPException:
        raise
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
    


