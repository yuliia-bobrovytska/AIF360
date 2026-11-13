"""
Assignment 4 - Task 2: Evaluating Reweighing (Kamiran & Calders, 2012)
=====================================================================

Pipeline (one run = one random seed):
  1. Load Adult, German Credit and COMPAS.
  2. Fairness of the ORIGINAL labels (whole dataset, before any split or model):
     DP_Diff, DIR, consistency.
  3. Stratified 70/30 train/test split (stratified on label x protected group).
  4. Reweighing (AIF360) is fitted on the TRAINING data only and gives every
     training instance a weight. A logistic regression is trained with those
     weights. The protected attribute is NOT a model input (assumption A2 of the
     paper); it is only used to compute the weights and the metrics.
     For comparison, the same model is trained once without weights (baseline).
  5. On the untouched test set: DP_Diff, DIR, consistency, accuracy,
     balanced accuracy and F1 of the final predictions.
  6. Everything is repeated for 10 seeds; mean and std are reported.

Run:    pip install -r requirements.txt
        python run_reweighing.py            (all datasets, 10 seeds)
        python run_reweighing.py --runs 2   (quick test)
Output: results/summary.md, results/all_runs.csv, results/fairness_before_after.png
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing

warnings.filterwarnings("ignore")

PROT = "prot"     # binary protected attribute: 1 = privileged, 0 = unprivileged
LABEL = "label"   # binary label: 1 = favourable outcome
PRIV = [{PROT: 1}]
UNPRIV = [{PROT: 0}]
COMPAS_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
              "master/compas-scores-two-years.csv")


# --------------------------------------------------------------------------
# 1. Data loading. Each loader returns (X, y, s, description):
#    X = feature DataFrame WITHOUT the protected attribute,
#    y = 0/1 labels (1 = favourable), s = 0/1 protected attribute (1 = privileged)
# --------------------------------------------------------------------------
def load_adult():
    df = fetch_openml("adult", version=2, as_frame=True).frame
    df = df.dropna().reset_index(drop=True)
    y = (df["class"].astype(str) == ">50K").astype(int)
    s = (df["sex"].astype(str) == "Male").astype(int)
    X = df.drop(columns=["class", "sex", "fnlwgt"])          # fnlwgt dropped as in the paper
    return X, y, s, "Adult: favourable = income >50K, protected = sex (privileged: Male)"


def load_german():
    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    y = (df["class"].astype(str) == "good").astype(int)
    s = (df["age"].astype(float) > 25).astype(int)
    X = df.drop(columns=["class", "age"])                      # raw age removed: it defines S
    return X, y, s, "German: favourable = good credit, protected = age (privileged: >25)"


def load_compas():
    df = pd.read_csv(COMPAS_URL)
    # standard ProPublica filtering
    df = df[(df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1) & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "N/A")]
    df = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)
    y = (df["two_year_recid"] == 0).astype(int)                # favourable = no recidivism
    s = (df["race"] == "Caucasian").astype(int)
    X = df[["sex", "age", "age_cat", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "priors_count", "c_charge_degree"]]
    return X, y, s, "COMPAS: favourable = no 2-year recidivism, protected = race (privileged: Caucasian)"


DATASETS = {"Adult": load_adult, "German": load_german, "COMPAS": load_compas}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def one_hot(X):
    """One-hot encode categorical columns, keep numeric columns as float."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return pd.get_dummies(X, columns=cat, dtype=float).astype(float)


def to_bld(X_num, y, s, weights=None):
    """Build an AIF360 BinaryLabelDataset (features + protected attribute + label)."""
    df = pd.DataFrame(X_num, columns=[f"f{i}" for i in range(X_num.shape[1])])
    df[PROT] = np.asarray(s)
    df[LABEL] = np.asarray(y)
    ds = BinaryLabelDataset(df=df, label_names=[LABEL], protected_attribute_names=[PROT],
                            favorable_label=1, unfavorable_label=0)
    if weights is not None:
        ds.instance_weights = np.asarray(weights, dtype=float)
    return ds


def fairness(ds, k=5):
    """DP_Diff, DIR and consistency of the labels stored in an AIF360 dataset.
    DP_Diff = P(fav | unpriv) - P(fav | priv)   (0 = parity, negative = unprivileged worse off)
    DIR     = P(fav | unpriv) / P(fav | priv)   (1 = parity, < 0.8 fails the 80% rule)
    consistency = 1 - mean |y_i - mean of y over the k nearest neighbours of i|  (1 = best)"""
    m = BinaryLabelDatasetMetric(ds, unprivileged_groups=UNPRIV, privileged_groups=PRIV)
    return {"DP_Diff": m.statistical_parity_difference(),
            "DIR": m.disparate_impact(),
            "Consistency": float(np.ravel(m.consistency(n_neighbors=k))[0])}


def predictive(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)}


# --------------------------------------------------------------------------
# one run on one dataset
# --------------------------------------------------------------------------
def run_once(name, X, y, s, seed, test_size=0.3):
    Xn = one_hot(X)
    rows = []

    # step 2: original labels, whole dataset, before any split or model
    full = MinMaxScaler().fit_transform(Xn)          # scaling only so kNN distances are fair
    rows.append({"Dataset": name, "Seed": seed, "Stage": "1_Original labels (full data)",
                 **fairness(to_bld(full, y, s))})

    # step 3: stratified split on label x protected group
    strata = y.astype(str) + "_" + s.astype(str)
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        Xn, y, s, test_size=test_size, stratify=strata, random_state=seed)
    scaler = MinMaxScaler().fit(Xtr)                 # fitted on training data only
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # step 4: Reweighing on the training data only
    train_ds = to_bld(Xtr_s, ytr, str_)
    rw = Reweighing(unprivileged_groups=UNPRIV, privileged_groups=PRIV)
    train_rw = rw.fit_transform(train_ds)
    w = train_rw.instance_weights

    # sanity check: the weighted training labels should have DP_Diff ~ 0 and DIR ~ 1
    m_w = BinaryLabelDatasetMetric(train_rw, unprivileged_groups=UNPRIV, privileged_groups=PRIV)
    rows.append({"Dataset": name, "Seed": seed, "Stage": "2_Training labels after reweighing",
                 "DP_Diff": m_w.statistical_parity_difference(), "DIR": m_w.disparate_impact()})

    # baseline model (no mitigation) and mitigated model; S is not a model input
    for stage, weights in [("3_Test predictions, no mitigation", None),
                           ("4_Test predictions, Reweighing", w)]:
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Xtr_s, ytr, sample_weight=weights)
        y_pred = clf.predict(Xte_s)
        # step 5: metrics on the held-out test set, using the final predictions
        rows.append({"Dataset": name, "Seed": seed, "Stage": stage,
                     **fairness(to_bld(Xte_s, y_pred, ste)),
                     **predictive(yte, y_pred)})

    weight_table = {f"W(s={int(k_s)}, y={int(k_y)})": float(w[(str_.values == k_s) & (ytr.values == k_y)][0])
                    for k_s in (0, 1) for k_y in (0, 1)}
    return rows, weight_table


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def fmt(mean, sd):
    return "" if pd.isna(mean) else f"{mean:.3f} ± {sd:.3f}"


def write_summary(df, info, weights, out_dir):
    metrics = ["DP_Diff", "DIR", "Consistency", "Accuracy", "Balanced_Acc", "F1"]
    g = df.groupby(["Dataset", "Stage"])[metrics]
    mean, sd = g.mean(), g.std().fillna(0)
    lines = ["# Reweighing (AIF360) - results\n",
             f"Mean ± std over {df['Seed'].nunique()} seeds; 70/30 stratified split; "
             "logistic regression; protected attribute not used as a model input.\n",
             "DP_Diff = P(fav|unpriv) − P(fav|priv) (ideal 0). DIR = ratio (ideal 1, fair if ≥ 0.8). "
             "Consistency: kNN, k = 5 (ideal 1).\n"]
    for name in df["Dataset"].unique():
        lines.append(f"\n## {name}\n\n{info[name]}\n")
        lines.append("| Stage | " + " | ".join(metrics) + " |")
        lines.append("|---" * (len(metrics) + 1) + "|")
        for stage in sorted(df.loc[df["Dataset"] == name, "Stage"].unique()):
            cells = [fmt(mean.loc[(name, stage), m], sd.loc[(name, stage), m]) for m in metrics]
            lines.append(f"| {stage[2:]} | " + " | ".join(cells) + " |")
        lines.append("\nReweighing weights (seed 0): " +
                     ", ".join(f"{k} = {v:.3f}" for k, v in weights[name].items()) +
                     "  (s=0 unprivileged, y=1 favourable)\n")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return mean


def plot(mean, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    stages = ["1_Original labels (full data)", "3_Test predictions, no mitigation",
              "4_Test predictions, Reweighing"]
    labels = ["Original labels", "No mitigation", "Reweighing"]
    names = list(dict.fromkeys(mean.index.get_level_values(0)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, ideal in zip(axes, ["DP_Diff", "DIR", "Consistency"], [0, 1, 1]):
        x = np.arange(len(names))
        for i, (st, lab) in enumerate(zip(stages, labels)):
            ax.bar(x + (i - 1) * 0.27, [mean.loc[(n, st), metric] for n in names], 0.27, label=lab)
        ax.axhline(ideal, color="black", lw=0.8, ls="--")
        if metric == "DIR":
            ax.axhline(0.8, color="red", lw=0.8, ls=":")
        ax.set_xticks(x, names)
        ax.set_title(metric)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fairness_before_after.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_rows, info, weights = [], {}, {}
    for name in args.datasets:
        print(f"Loading {name} ...")
        X, y, s, desc = DATASETS[name]()
        info[name] = f"{desc}. n = {len(y)}, favourable rate = {y.mean():.3f}, " \
                     f"privileged share = {s.mean():.3f}."
        for seed in range(args.runs):
            rows, wt = run_once(name, X, y, s, seed)
            all_rows += rows
            if seed == 0:
                weights[name] = wt
        print(f"  {name}: {args.runs} runs done")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)
    mean = write_summary(df, info, weights, args.out)
    plot(mean, args.out)
    print(f"\nSaved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
