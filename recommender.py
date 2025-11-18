import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import joblib

DATA_DIR = Path("data")
pq_path = DATA_DIR / "prog_df.parquet"
csv_path = DATA_DIR / "prog_df.csv"
def _sanitize(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    # استبدال NaN و±Inf بـ 0
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def _load_table():
    if pq_path.exists() and pq_path.stat().st_size > 0:
        return pd.read_parquet(pq_path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path)
    raise FileNotFoundError("No valid program table found in ./data")

prog_df = _load_table()

# === Load Features ===
feats_npz = np.load(DATA_DIR / "programs_features.npz", allow_pickle=False)
features = _sanitize(feats_npz["features"])
meta_has_days = bool(np.asarray(feats_npz.get("meta_has_days", [0])).ravel()[0])
scaler = joblib.load(DATA_DIR / "days_scaler.joblib") if meta_has_days and (DATA_DIR/"days_scaler.joblib").exists() else None

# --- Normalize and Clean ---
if "goal_clean" in prog_df.columns:
    prog_df["goal_clean"] = prog_df["goal_clean"].astype(str).str.strip().str.lower()
if "days_per_week" in prog_df.columns:
    prog_df["days_per_week"] = pd.to_numeric(prog_df["days_per_week"], errors="coerce")

import numpy as np
def _to_list(v):
    if isinstance(v, (list, tuple)):
        vals = list(v)
    elif isinstance(v, np.ndarray):
        vals = v.tolist()
    elif v is None:
        return []
    else:
        if isinstance(v, float):
            if np.isnan(v):
                return []
            return [str(v).strip().lower()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].replace("'", "").replace('"', "")
            vals = [p for p in s.split(",")]
        else:
            vals = [str(v)]
    out = []
    for x in vals:
        if x is None: continue
        if isinstance(x, (list, tuple, np.ndarray)):
            out.extend([str(xx).strip().lower() for xx in (x.tolist() if isinstance(x, np.ndarray) else x) if str(xx).strip()])
            continue
        if isinstance(x, float) and np.isnan(x): continue
        s = str(x).strip().lower()
        if s: out.append(s)
    return out

if "level_list" in prog_df.columns:
    prog_df["level_list"] = prog_df["level_list"].apply(_to_list)
else:
    prog_df["level_list"] = [[] for _ in range(len(prog_df))]

level_norm = {"beginner":"beginner","novice":"novice","intermediate":"intermediate","advanced":"advanced"}

def _as_bool(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        arr = np.array([bool(x) if isinstance(x, (bool, np.bool_)) else False for x in arr])
    return arr.astype(bool)

def _filter_indices(user_goal: str, user_days: int, user_level: str, days_tolerance: int = 1) -> np.ndarray:
    g = str(user_goal).strip().lower()
    lv = level_norm.get(str(user_level).strip().lower(), str(user_level).strip().lower())
    d = int(user_days)

    N = len(prog_df)
    mask = np.ones(N, dtype=bool)

    if "goal_clean" in prog_df.columns:
        goal_mask = _as_bool(prog_df["goal_clean"].astype(str).str.lower().eq(g).to_numpy())
        mask &= goal_mask

    if "days_per_week" in prog_df.columns:
        dpw = pd.to_numeric(prog_df["days_per_week"], errors="coerce")
        within = dpw.between(d - days_tolerance, d + days_tolerance, inclusive="both") | dpw.isna()
        mask &= _as_bool(within.to_numpy())

    if "level_list" in prog_df.columns:
        level_ok = prog_df["level_list"].apply(lambda L: lv in L if isinstance(L, list) else False)
        mask &= _as_bool(level_ok.to_numpy())

    idx = np.where(mask)[0]
    if idx.size == 0:
        if "goal_clean" in prog_df.columns:
            idx = np.where(prog_df["goal_clean"].to_numpy() == g)[0]
    if idx.size == 0:
        idx = np.arange(N)
    L = min(idx.size, features.shape[0], len(prog_df))
    return idx[:L]

def recommend(goal: str, days: int, level: str, top_n: int = 10) -> pd.DataFrame:
    cand_idx = _filter_indices(goal, days, level, days_tolerance=1)
    if cand_idx.size == 0:
        return pd.DataFrame(columns=["title","goal","goal_clean","level","level_list","days_per_week","description","cosine_similarity"])
    cand_mat = _sanitize(features[cand_idx])
# لو كل شيء صفر، نحط متجه بسيط لتفادي division by zero داخل cosine
    if not np.isfinite(cand_mat).all():
        cand_mat = _sanitize(cand_mat)
    q_vec = _sanitize(cand_mat.mean(axis=0, keepdims=True))

# لو المتجه صفر كليًا، خلّي عنصر واحد 1e-9 علشان التشابه ما ينهار
    if np.linalg.norm(q_vec) == 0:
        q_vec[0, 0] = 1e-9
    sim = cosine_similarity(q_vec, cand_mat).ravel()
    sim = _sanitize(sim)

    order_rel = np.argsort(-sim)[: int(max(1, min(top_n, cand_idx.size)))]
    order = cand_idx[order_rel]
    top_sim = sim[order_rel]
    cols = [c for c in ["title","goal","goal_clean","level","level_list","days_per_week","description"] if c in prog_df.columns]
    out = prog_df.iloc[order][cols].copy()
    out["cosine_similarity"] = np.round(top_sim, 6)
    return out.reset_index(drop=True)
