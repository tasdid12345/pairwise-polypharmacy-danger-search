"""Baseline MLP (raw substance levels) vs class-balanced LightGBM (levels + pairwise co-use)
for a rule-based 'dangerous polypharmacy' label on the UCI Drug Consumption (quantified) data."""
import json, socket, sys, time
from itertools import combinations

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

SEED = 42
SEEDS = [42, 43, 44]  # model seeds; the single train/test split uses SEED
SUBS = ["Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke", "Crack",
        "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "VSA"]
# A-priori sedation-risk pairs (stand-in label, not clinical truth)
DANGER_PAIRS = [("Alcohol", "Benzos"), ("Alcohol", "Heroin"), ("Benzos", "Heroin"), ("Benzos", "Ketamine")]
URL = "https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    socket.setdefaulttimeout(60)  # 60 s timeout for network calls
    for attempt in range(5):  # retry with exponential backoff on 429 / 5xx / network errors
        try:
            ds = fetch_ucirepo(id=373)
            break
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt == 4 or (status is not None and status != 429 and status < 500):
                raise
            log(f"download failed ({e}); retry in {2 ** attempt}s")
            time.sleep(2 ** attempt)
    X = ds.data.features
    # Levels 0-6; parse 'CLn' strings if the package returns them as text
    L = np.column_stack([
        X[c].to_numpy(float) if pd.api.types.is_numeric_dtype(X[c])
        else X[c].astype(str).str.extract(r"(\d)")[0].astype(float).to_numpy()
        for c in SUBS])
    use = L >= 3  # "using" = level >= 3
    n_use = use.sum(1)
    pairs = list(combinations(range(len(SUBS)), 2))  # 153 pairs
    co = np.column_stack([use[:, i] & use[:, j] for i, j in pairs]).astype(float)
    has_danger = np.zeros(len(L), bool)
    for a, b in DANGER_PAIRS:
        has_danger |= use[:, SUBS.index(a)] & use[:, SUBS.index(b)]
    y = ((n_use >= 5) & has_danger).astype(int)  # polypharmacy AND a listed co-use
    X_raw = L                                     # baseline features
    X_pair = np.column_stack([L, n_use, co])      # improved features
    return X_raw, X_pair, y, len(ds.data.features)


def best_threshold(y, p):
    """Threshold maximising F1 on out-of-fold TRAINING predictions only."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return thr[np.argmax(f1[:-1])]


def run_seed(seed, Xr_tr, Xp_tr, y_tr, Xr_te, Xp_te, y_te):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    # Baseline: standardised MLP (64-32) on raw levels, alpha tuned by CV on training data
    base = GridSearchCV(
        make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                      early_stopping=True, random_state=seed)),
        {"mlpclassifier__alpha": [1e-4, 1e-2]}, scoring="average_precision", cv=skf)
    # Improved: class-balanced LightGBM, small CV grid on training data
    imp = GridSearchCV(
        LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1),
        {"num_leaves": [15, 31], "n_estimators": [200, 400]}, scoring="average_precision", cv=skf)
    out = {}
    for name, est, Xtr, Xte in [("baseline", base, Xr_tr, Xr_te), ("improved", imp, Xp_tr, Xp_te)]:
        est.fit(Xtr, y_tr)
        best = est.best_estimator_
        oof = cross_val_predict(clone(best), Xtr, y_tr, cv=skf, method="predict_proba")[:, 1]
        thr = best_threshold(y_tr, oof)
        p = best.predict_proba(Xte)[:, 1]  # test set used once, here
        out[name] = dict(pr_auc=average_precision_score(y_te, p),
                         f1=f1_score(y_te, (p >= thr).astype(int)),
                         roc_auc=roc_auc_score(y_te, p), params=est.best_params_)
        log(f"seed {seed} {name}: PR-AUC={out[name]['pr_auc']:.4f} F1={out[name]['f1']:.4f} "
            f"ROC-AUC={out[name]['roc_auc']:.4f} params={est.best_params_}")
    return out


def main():
    t0 = time.time()
    Xr, Xp, y, n_feat = load_data()
    log(f"rows={len(y)} raw_features={n_feat} dangerous positives={y.sum()} ({y.mean():.3f})")
    if y.sum() < 20 or y.sum() > len(y) - 20:
        log("label is degenerate; cannot produce meaningful metrics")
        sys.exit(1)
    # Single stratified 70/30 split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    log(f"train={len(tr)} test={len(te)}")
    res = {s: run_seed(s, Xr[tr], Xp[tr], y[tr], Xr[te], Xp[te], y[te]) for s in SEEDS}

    def agg(model, key):
        v = np.array([res[s][model][key] for s in SEEDS], float)
        return float(v.mean()), float(v.std())

    b, bs = agg("baseline", "pr_auc")
    i, is_ = agg("improved", "pr_auc")
    if not np.isfinite([b, i]).all():
        log("non-finite scores")
        sys.exit(1)
    bf, bfs = agg("baseline", "f1"); if_, ifs = agg("improved", "f1")
    br, brs = agg("baseline", "roc_auc"); ir, irs = agg("improved", "roc_auc")
    metrics = {
        "baseline_score": b, "baseline_score_std": bs,
        "improved_score": i, "improved_score_std": is_,
        "improvement": i - b,
        "baseline_f1": bf, "baseline_f1_std": bfs, "improved_f1": if_, "improved_f1_std": ifs,
        "baseline_roc_auc": br, "baseline_roc_auc_std": brs,
        "improved_roc_auc": ir, "improved_roc_auc_std": irs,
        "n_test_positive": int(y[te].sum()), "test_positive_rate": float(y[te].mean()),
    }
    results = {
        "question": ("Can a class-balanced LightGBM model with pairwise co-use features flag dangerous "
                     "polypharmacy better than a standard MLP surrogate that sees only the raw per-substance "
                     "levels, on a public substance-use dataset with a rule-based danger label?"),
        "method": ("UCI Drug Consumption (quantified); use = level>=3; polypharmacy = >=5 substances used; "
                   "dangerous = polypharmacy AND an a-priori sedation-risk co-use pair. One stratified 70/30 split; "
                   "3-fold CV on training data for hyperparameters and decision threshold; model seeds 42-44; "
                   "PR-AUC main metric, F1 and ROC-AUC secondary."),
        "dataset": {"name": "UCI Drug Consumption (quantified)", "source_url": URL,
                    "n_rows": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "PR-AUC (average precision) on dangerous-polypharmacy label",
        "higher_is_better": True,
        "baseline": "Standardised MLPClassifier (64-32 hidden units) on the 18 raw substance levels; "
                    "alpha tuned by CV. Stands in for the neural reward model; not the paper's neural bandit.",
        "improved": "Class-balanced LightGBM on raw levels + total substance count + all 153 pairwise "
                    "co-use indicators; small CV grid over num_leaves and n_estimators.",
        "metrics": metrics,
        "notes": ("Stand-in rule-based label, not clinical truth. The improved feature set contains the exact "
                  "pairwise indicators used to define the label, so the gain partly reflects label construction. "
                  "Only one train/test split was used (split-once rule), so std covers model seeds 42-44 only, "
                  "not split variability. Threshold for F1 chosen on training out-of-fold predictions. "
                  "Test set was not used for any selection. Semer (near-constant, fictitious drug) excluded. "
                  "Dataset is small (about 1.9k rows), so no subsampling was needed."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Optional bar chart
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["MLP baseline", "LightGBM+pairs"], [b, i], yerr=[bs, is_], capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel("PR-AUC (dangerous label, test)")
    ax.set_title("Mean +/- std over model seeds")
    fig.tight_layout()
    fig.savefig("figures/pr_auc_comparison.png", dpi=120)
    log(f"done in {time.time() - t0:.1f}s; baseline PR-AUC={b:.4f}, improved PR-AUC={i:.4f}")


if __name__ == "__main__":
    main()
