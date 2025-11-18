# build_data.py  — robust Excel -> parquet + features
import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler
import joblib

# ---- paths ----

# base directory of the project (same folder as build_data.py)
BASE_DIR = Path(__file__).resolve().parent

# correct paths to your Excel files inside the repo
DATASET_SUMMARY = BASE_DIR / "datasets" / "full_Programs_Summury.xlsx"
DATASET_DETAILED = BASE_DIR / "datasets" / "programs_detailed.xlsx"  # optional

# output directory
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(exist_ok=True)

print("Summary file:", DATASET_SUMMARY)
print("Detailed file:", DATASET_DETAILED)


# ---- helpers ----
def clean_text(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.lower()
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s:x×/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_col(df: pd.DataFrame, col: str, default="") -> pd.Series:
    """Return a series for column or default scalar repeated."""
    if col in df.columns:
        return df[col]
    # make same length
    return pd.Series([default] * len(df), index=df.index)

# ---- read summary ----
xls = pd.ExcelFile(DATASET_SUMMARY)
# حاول نلاقي sheet بالاسم الأقرب
sheet_names = [s.strip().lower() for s in xls.sheet_names]
def pick_sheet(candidates, fallback_index=0):
    for want in candidates:
        for i, s in enumerate(sheet_names):
            if s == want:
                return xls.sheet_names[i]
    return xls.sheet_names[fallback_index]

summary_sheet = pick_sheet(["programs summary", "summary", "sheet1"], 0)
summary_df = pd.read_excel(xls, sheet_name=summary_sheet)

# أعمدة أساسية
summary_df["title"] = safe_col(summary_df, "title", "").astype(str).str.strip()
summary_df["description"] = safe_col(summary_df, "description", "").astype(str)
if "goal" in summary_df.columns:
    summary_df["goal_clean"] = summary_df["goal"].astype(str).str.strip().str.lower()
else:
    summary_df["goal_clean"] = "unknown"

# days_per_week (اختياري)
if "days_per_week" in summary_df.columns:
    summary_df["days_per_week"] = pd.to_numeric(summary_df["days_per_week"], errors="coerce")
else:
    summary_df["days_per_week"] = np.nan

# level_list (قد تكون نص/قائمة/NaN)
def norm_level_list(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return []
    if isinstance(v, list): vals = v
    elif isinstance(v, (tuple, np.ndarray)): vals = list(v)
    elif isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].replace("'", "").replace('"', "")
        vals = [p for p in s.split(",")]
    else:
        vals = [str(v)]
    out = []
    for x in vals:
        if x is None: continue
        if isinstance(x, float) and np.isnan(x): continue
        out.append(str(x).strip().lower())
    return [z for z in out if z]
summary_df["level_list"] = safe_col(summary_df, "level_list", []).apply(norm_level_list)

# ---- read detailed (optional) & build common_exercises ----
exercise_blob = pd.DataFrame({"title": summary_df["title"], "common_exercises": ""})
if DATASET_DETAILED.exists() and DATASET_DETAILED.stat().st_size > 0:
    try:
        detail_df = pd.read_excel(DATASET_DETAILED)
        if "title" in detail_df.columns:
            detail_df["title"] = detail_df["title"].astype(str).str.strip()
            ex_name_col = "exercise_name" if "exercise_name" in detail_df.columns else None
            if ex_name_col:
                tmp = detail_df.copy()
                tmp["ex_clean"] = tmp[ex_name_col].astype(str).map(clean_text)
                exercise_blob = (
                    tmp.groupby("title")["ex_clean"]
                    .apply(lambda s: " ".join(pd.Series(s.dropna().unique()).head(10)))
                    .reset_index()
                    .rename(columns={"ex_clean": "common_exercises"})
                )
    except Exception as e:
        print("⚠️ Detailed read failed, continue without it:", e)

# ---- merge & build text_input ----
summary_df = summary_df.merge(exercise_blob, on="title", how="left")
summary_df["common_exercises"] = summary_df["common_exercises"].fillna("")
summary_df["clean_title"] = summary_df["title"].map(clean_text)
summary_df["clean_desc"] = summary_df["description"].map(clean_text)
summary_df["text_input"] = (
    summary_df["clean_title"] + " " +
    summary_df["clean_desc"] + " " +
    summary_df["common_exercises"].map(clean_text)
).str.strip()

# تأكيد أن text_input موجود
if "text_input" not in summary_df.columns:
    raise RuntimeError("text_input column was not created.")

# ---- features: TF-IDF + SVD ----
tfidf = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), min_df=2)
svd = TruncatedSVD(n_components=256, random_state=42)
pipe = make_pipeline(tfidf, svd, Normalizer(copy=False))

features = pipe.fit_transform(summary_df["text_input"].fillna(""))

# ---- optional meta: days_per_week ----
meta_has_days = 0
scaler = None
if "days_per_week" in summary_df.columns:
    summary_df["days_per_week_filled"] = summary_df["days_per_week"].fillna(summary_df["days_per_week"].median())
    try:
        scaler = MinMaxScaler()
        meta = scaler.fit_transform(summary_df[["days_per_week_filled"]].values.astype(float))
        features = np.hstack([features, meta * 0.25])
        meta_has_days = 1
        joblib.dump(scaler, OUT_DIR / "days_scaler.joblib")
    except Exception as e:
        print("⚠️ days_per_week scaling failed; continue without it:", e)
        meta_has_days = 0

# ---- save ----
summary_df.to_parquet(OUT_DIR / "prog_df.parquet", index=False)
np.savez_compressed(OUT_DIR / "programs_features.npz", features=features, meta_has_days=meta_has_days)

print("✅ Saved:")
print(" -", (OUT_DIR / "prog_df.parquet").resolve())
print(" -", (OUT_DIR / "programs_features.npz").resolve())
if meta_has_days:
    print(" -", (OUT_DIR / "days_scaler.joblib").resolve())
print(f"Rows: {len(summary_df)}  | Features: {features.shape}")
