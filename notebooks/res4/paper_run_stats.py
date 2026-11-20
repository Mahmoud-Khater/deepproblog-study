"""Statistics for the paper run (result_4) that need no retraining.

Inputs: test_prob.npy and baseline_test_prob.npy from the result_4 Kaggle output
(this folder), the four CBIS-DDSM CSVs in ../../data, and ../y_test_true.npy.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
PAPER_THRESHOLD = 0.55
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
Z95 = 1.959963984540054

STRING_COLUMNS = ["patient_id", "abnormality type", "mass shape", "mass margins",
                  "calc type", "calc distribution", "pathology"]


def load(name, kind, split):
    df = pd.read_csv(DATA / name)
    df.columns = [c.strip() for c in df.columns]
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df["abnormality_kind"] = kind
    df["official_split"] = split
    return df


def build_test():
    train = pd.concat([load("mass_case_description_train_set.csv", "mass", "train"),
                       load("calc_case_description_train_set.csv", "calcification", "train")], ignore_index=True)
    test = pd.concat([load("mass_case_description_test_set.csv", "mass", "test"),
                      load("calc_case_description_test_set.csv", "calcification", "test")], ignore_index=True)
    overlap = set(train["patient_id"].dropna()) & set(test["patient_id"].dropna())
    train = pd.concat([train, test[test["patient_id"].isin(overlap)]], ignore_index=True)
    train = train.drop_duplicates().reset_index(drop=True)
    test = test[~test["patient_id"].isin(overlap)].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    next(gss.split(train, groups=train["patient_id"]))
    return test, len(overlap)


def tokens(value):
    if pd.isna(value):
        return []
    text = str(value).strip().upper()
    for sep in ("-", "/", "_", " "):
        text = text.replace(sep, " ")
    return [t for t in text.split() if t and t not in {"NAN", "NONE"}]


def shape_concept(v):
    t = tokens(v)
    if any(x in {"IRREGULAR", "ARCHITECTURAL", "DISTORTION"} for x in t):
        return "irregular"
    if any(x in {"ROUND", "OVAL", "LOBULATED"} for x in t):
        return "regular"
    return "unknown"


def margin_original(v):
    t = tokens(v)
    if not t:
        return "unknown"
    if "SPICULATED" in t:
        return "spiculated"
    if "CIRCUMSCRIBED" in t:
        return "circumscribed"
    return "other"


def margin_revised(v):
    t = tokens(v)
    if not t:
        return "unknown"
    if "SPICULATED" in t:
        return "spiculated"
    if "INDISTINCT" in t or ("ILL" in t and "DEFINED" in t):
        return "indistinct"
    if "MICROLOBULATED" in t:
        return "microlobulated"
    if "OBSCURED" in t:
        return "obscured"
    if "CIRCUMSCRIBED" in t:
        return "circumscribed"
    return "unknown"


def calc_suspicious(row):
    if row["abnormality_kind"] != "calcification":
        return False
    t = tokens(row.get("calc type"))
    return any(x in {"PLEOMORPHIC", "AMORPHOUS", "BRANCHING"} for x in t) or ("FINE" in t and "LINEAR" in t)


def rule_fires(row, revised):
    if row["abnormality_kind"] == "calcification":
        return calc_suspicious(row)
    if shape_concept(row["mass shape"]) == "irregular":
        return True
    if revised:
        return margin_revised(row["mass margins"]) in {"spiculated", "indistinct", "microlobulated"}
    return margin_original(row["mass margins"]) == "spiculated"


def binary(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def structural(y, s):
    pos, neg = s[y == 1], s[y == 0]
    r = midrank(np.concatenate([pos, neg]))
    rp, rn = r[:len(pos)], r[len(pos):]
    auc = rp.sum() / (len(pos) * len(neg)) - (len(pos) + 1.0) / (2.0 * len(neg))
    return auc, (rp - midrank(pos)) / len(neg), 1.0 - (rn - midrank(neg)) / len(pos)


def delong(y, a, b):
    auc_a, pa, na = structural(y, a)
    auc_b, pb, nb = structural(y, b)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    c = lambda u, v: float(np.cov(u, v, ddof=1)[0, 1])
    var = (c(pa, pa) / npos + c(na, na) / nneg + c(pb, pb) / npos + c(nb, nb) / nneg
           - 2 * c(pa, pb) / npos - 2 * c(na, nb) / nneg)
    se = math.sqrt(max(0.0, var))
    diff = auc_a - auc_b
    z = diff / se if se > 0 else 0.0
    return {"auc_deepproblog": auc_a, "auc_resnet18": auc_b, "difference": diff, "se": se, "z": z,
            "p_value": math.erfc(abs(z) / math.sqrt(2.0)),
            "ci95": [diff - Z95 * se, diff + Z95 * se]}


def bootstrap(y, scores):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {k: [] for k in scores}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        for k, s in scores.items():
            draws[k].append(roc_auc_score(y[idx], s[idx]))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in draws.items()}


def specificity_at_sensitivity(y, s, target):
    fpr, tpr, thr = roc_curve(y, s)
    ok = tpr >= target
    i = int(np.argmax(ok))
    return {"threshold": float(thr[i]), "sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i])}


def main():
    test, n_overlap = build_test()
    y_csv = (test["pathology"].str.upper() == "MALIGNANT").astype(int).to_numpy()
    y = np.load(HERE.parent / "y_test_true.npy").astype(int)
    dpl = np.load(HERE / "test_prob.npy").astype(float)
    base = np.load(HERE / "baseline_test_prob.npy").astype(float)
    assert len(y) == len(dpl) == len(base) == len(test) == 644
    assert np.array_equal(y, y_csv), "y_test_true.npy does not follow the rebuilt test order"
    print(f"Test rebuilt from CSVs: {len(test)} rows, {n_overlap} leaked patients moved; labels match y_test_true.npy")

    out = {"n_test": int(len(y)), "malignant": int(y.sum()), "benign": int((1 - y).sum())}
    out["delong"] = delong(y, dpl, base)
    out["bootstrap_ci95"] = bootstrap(y, {"deepproblog": dpl, "resnet18": base})

    gt = {}
    for name, revised in (("original", False), ("revised", True)):
        fires = np.array([rule_fires(r, revised) for _, r in test.iterrows()], dtype=float)
        m = binary(y, fires, 0.5)
        m["roc_auc"] = float(roc_auc_score(y, fires))
        gt[name] = m
    out["ground_truth_program"] = gt

    gt_pred = np.array([rule_fires(r, False) for _, r in test.iterrows()], dtype=int)
    dpl_pred = (dpl >= PAPER_THRESHOLD).astype(int)
    wrong = dpl_pred != y
    rule_side = wrong & (gt_pred != y)
    head_side = wrong & (gt_pred == y)
    out["error_attribution"] = {
        "threshold": PAPER_THRESHOLD, "errors": int(wrong.sum()),
        "rule_side": int(rule_side.sum()), "head_side": int(head_side.sum()),
        "rule_side_fn": int((rule_side & (y == 1)).sum()), "rule_side_fp": int((rule_side & (y == 0)).sum()),
        "head_side_fn": int((head_side & (y == 1)).sum()), "head_side_fp": int((head_side & (y == 0)).sum()),
    }
    out["at_paper_threshold"] = binary(y, dpl, PAPER_THRESHOLD)
    out["specificity_at_test_sensitivity_0.80"] = {
        "deepproblog": specificity_at_sensitivity(y, dpl, 0.80),
        "resnet18": specificity_at_sensitivity(y, base, 0.80),
    }
    print(json.dumps(out, indent=2))
    (HERE / "paper_run_stats.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
