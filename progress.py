import pandas as pd
import numpy as np
from typing import Dict, Tuple

SESS_STORE: Dict[str, pd.DataFrame] = {}

# ------------------- ingestion -------------------
def ingest_sessions(user_id: str, planned_per_week: int, new_sessions: pd.DataFrame):
    new_sessions = new_sessions.copy()
    new_sessions["date"] = pd.to_datetime(new_sessions["date"])
    if user_id in SESS_STORE:
        SESS_STORE[user_id] = pd.concat([SESS_STORE[user_id], new_sessions], ignore_index=True).drop_duplicates()
    else:
        SESS_STORE[user_id] = new_sessions

# ------------------- core rule-based level -------------------
def level_from_sessions(sessions_df: pd.DataFrame, planned_per_week=4):
    df = sessions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    effort_map = {"easy":0.0,"medium":0.5,"hard":1.0}
    df["effort_score"] = df["effort"].map(effort_map).fillna(0.5)
    wk = df.groupby("week").agg(
        days_trained=("date","nunique"),
        effort_mean=("effort_score","mean")
    ).reset_index().sort_values("week")
    wk["adherence"] = (wk["days_trained"]/planned_per_week).clip(0,1)
    wk["freq_norm"] = (wk["days_trained"]/planned_per_week).clip(0,1)
    wk["effort_norm"] = wk["effort_mean"].clip(0,1)
    threshold = max(1,int(np.ceil(planned_per_week*0.7)))
    streak=0; streaks=[]
    for d in wk["days_trained"]:
        streak = streak+1 if d>=threshold else 0
        streaks.append(streak)
    wk["streak_norm"] = np.clip(np.array(streaks)/6.0,0,1)
    wk["LevelScore"]=100*(0.55*wk["freq_norm"]+0.25*wk["adherence"]+0.10*wk["streak_norm"]+0.10*wk["effort_norm"])
    def band(s): return "beginner" if s<60 else "intermediate" if s<80 else "advanced"
    current=None; up=down=0; levels=[]
    rank={"beginner":0,"intermediate":1,"advanced":2}
    for s in wk["LevelScore"]:
        target=band(s)
        if current is None: current=target
        elif rank[target]>rank[current]:
            up+=1; down=0
            if up>=2: current=target; up=0
        elif rank[target]<rank[current]:
            down+=1; up=0
            if down>=2: current=target; down=0
        else: up=down=0
        levels.append(current)
    wk["level_final"]=levels
    return wk

# ------------------- readiness & progress -------------------
def _ema(s,alpha=0.25): return s.ewm(alpha=alpha,adjust=False).mean()

def build_readiness(wk_df: pd.DataFrame):
    df=wk_df.copy()
    df["ema_freq"]=_ema(df["freq_norm"].clip(0,1))
    df["ema_adher"]=_ema(df["adherence"].clip(0,1))
    df["ema_effort"]=_ema(df["effort_norm"].clip(0,1))
    df["ema_streak"]=_ema(df["streak_norm"].clip(0,1))
    df["Readiness"]=100*(0.45*df["ema_freq"]+0.2*df["ema_adher"]+0.15*df["ema_effort"]+0.2*df["ema_streak"])
    df["cum_readiness_mean"]=df["Readiness"].expanding().mean()
    return df

def apply_progress_bar(df: pd.DataFrame, decision_period_weeks:int=4):
    df=df.copy()
    cur=df.iloc[0]["level_final"]; prog=50
    rank={"beginner":0,"intermediate":1,"advanced":2}
    out_levels=[]; out_bar=[]; out_check=[]
    for i,row in df.iterrows():
        delta=0.6*(row["Readiness"]-65)
        prog=np.clip(prog+delta,-100,100)
        is_decision=(i+1)%decision_period_weeks==0
        if is_decision:
            if prog>=100 and cur!="advanced":
                cur=["beginner","intermediate","advanced"][rank[cur]+1]; prog=50
            elif prog<=0 and cur!="beginner":
                cur=["beginner","intermediate","advanced"][rank[cur]-1]; prog=50
        out_levels.append(cur)
        out_bar.append(round(prog,1))
        out_check.append(is_decision)
    df["progress_bar"]=out_bar
    df["is_decision_week"]=out_check
    df["level_progressive"]=out_levels
    return df

# ------------------- compute wrapper -------------------
def compute_progress(user_id:str, decision_period_weeks:int=4) -> Tuple[dict,pd.DataFrame]:
    if user_id not in SESS_STORE:
        raise ValueError(f"User {user_id} has no sessions.")
    df=SESS_STORE[user_id]
    wk=level_from_sessions(df)
    wk=build_readiness(wk)
    wk=apply_progress_bar(wk,decision_period_weeks=decision_period_weeks)
    last=wk.iloc[-1]
    out={
        "final_level":str(last["level_progressive"]),
        "last_progress":float(last["progress_bar"]),
        "last_readiness":float(last["Readiness"]),
        "weeks_observed":int(len(wk))
    }
    return out,wk
