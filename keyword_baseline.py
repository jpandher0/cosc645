import ast
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RESUME_COLS = [
    "career_objective", "skills", "degree_names", "major_field_of_studies",
    "positions", "responsibilities", "related_skils_in_job",
    "certification_skills", "role_positions",
]
JOB_COLS = [
    "job_position_name", "educationaL_requirements", "experiencere_requirement",
    "responsibilities.1", "skills_required",
]
SKILL_COL = "skills_required"
TARGET = "matched_score"

path = sys.argv[1] if len(sys.argv) > 1 else "data/resume_data.csv"
df = pd.read_csv(path)
df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]  
print("--------------------------------")
print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("--------------------------------")
print("\nColumns and missing values:")
print(df.isna().sum().to_string())
if TARGET not in df.columns:
    sys.exit(f"\nCould not find '{TARGET}' column. Columns are: {list(df.columns)}")
print(f"\n{TARGET} summary:")
print(df[TARGET].describe().to_string())
plt.figure(figsize=(7, 4))
df[TARGET].hist(bins=30, edgecolor="black")
plt.title("Distribution of ground-truth matched_score")
plt.xlabel("matched_score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/matched_score_distribution.png")
plt.close()
job_cols = [c for c in JOB_COLS if c in df.columns]
resume_cols = [c for c in RESUME_COLS if c in df.columns]
missing_cols = [c for c in JOB_COLS + RESUME_COLS if c not in df.columns]
if missing_cols:
    print(f"\nWarning: columns not found and skipped: {missing_cols}")
print(f"\nJob description columns: {job_cols}")
print(f"Resume columns: {resume_cols}")
def flatten(value):
    """Turn list-like strings such as "['Python', 'SQL']" into plain text."""
    if pd.isna(value):
        return ""
    value = str(value)
    if value.startswith("["):
        try:
            items = ast.literal_eval(value)
            return " ".join(str(i) for i in items if i is not None)
        except (ValueError, SyntaxError):
            pass
    return value
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text) 
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


df["resume_text"] = df[resume_cols].apply(lambda r: " ".join(flatten(v) for v in r), axis=1).map(clean)
df["job_text"] = df[job_cols].apply(lambda r: " ".join(flatten(v) for v in r), axis=1).map(clean)
df = df[(df["resume_text"].str.len() > 0) & (df["job_text"].str.len() > 0)].reset_index(drop=True)
print(f"\nRows after removing empty resumes/jobs: {len(df)}")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
vectorizer.fit(pd.concat([df["resume_text"], df["job_text"]]))
R = vectorizer.transform(df["resume_text"])
J = vectorizer.transform(df["job_text"])
df["keyword_score"] = [cosine_similarity(R[i], J[i])[0, 0] for i in range(len(df))]
def skill_gap(row):
    if SKILL_COL not in row.index:
        return [], []
    raw = flatten(row[SKILL_COL])
    skills = [clean(s) for s in re.split(r"[,;\n|]", raw)]
    skills = list(dict.fromkeys(s for s in skills if s)) 
    padded = f" {row['resume_text']} "
    matched = [s for s in skills if f" {s} " in padded] 
    missing = [s for s in skills if f" {s} " not in padded]
    return matched, missing
gaps = df.apply(skill_gap, axis=1)
df["matched_skills"] = [g[0] for g in gaps]
df["missing_skills"] = [g[1] for g in gaps]
df["skill_coverage"] = [len(m) / (len(m) + len(x)) if (m or x) else float("nan") for m, x in gaps]
print("--------------------------------")
print("Baseline evaluation vs matched_score")
print("--------------------------------")
for col in ["keyword_score", "skill_coverage"]:
    sub = df[[col, TARGET]].dropna()
    p, _ = pearsonr(sub[col], sub[TARGET])
    s, _ = spearmanr(sub[col], sub[TARGET])
    print(f"{col:15s}  n = {len(sub):5d}   Pearson r = {p:.3f}   Spearman rho = {s:.3f}")
plt.figure(figsize=(6, 5))
plt.scatter(df["keyword_score"], df[TARGET], alpha=0.25, s=8)
plt.title("Keyword (TF-IDF) score vs ground truth")
plt.xlabel("keyword_score")
plt.ylabel("matched_score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/keyword_vs_groundtruth.png")
plt.close()
ex_idx = df["skill_coverage"].first_valid_index()
ex = df.loc[ex_idx]
print("--------------------------------")
print(f"Example match explanation (row {ex_idx}: {ex['job_position_name']})")
print("--------------------------------")
print(f"Keyword score:   {ex['keyword_score']:.3f}")
print(f"Ground truth:    {ex[TARGET]}")
print(f"Matched skills:  {ex['matched_skills'][:10]}")
print(f"Missing skills:  {ex['missing_skills'][:10]}")

df[["keyword_score", "skill_coverage", TARGET, "matched_skills", "missing_skills"]].to_csv(
    f"{OUT_DIR}/baseline_results.csv", index=False
)
print(f"\nSaved charts and results to ./{OUT_DIR}/")
