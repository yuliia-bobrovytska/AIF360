# Assignment 4 – Evaluating Reweighing (Kamiran & Calders, 2012)

This folder is my addition to a fork of [Trusted-AI/AIF360](https://github.com/Trusted-AI/AIF360).
It evaluates the **Reweighing** pre-processing algorithm
(`aif360/algorithms/preprocessing/reweighing.py`, used unchanged) on three datasets:
Adult, German Credit and COMPAS.

| File | Purpose |
|---|---|
| `run_reweighing.py` | Full pipeline: data loading, fairness of original labels, stratified split, Reweighing, weighted logistic regression, test-set metrics, 10 seeds |
| `requirements.txt` | Pinned package versions |
| `results/summary.md` | Mean ± std tables reported in the assignment |
| `results/all_runs.csv` | Every metric for every dataset, stage and seed |
| `results/fairness_before_after.png` | Figure 1 of the report |

## What the script does

1. Downloads the datasets (Adult and German Credit from OpenML, COMPAS from the ProPublica GitHub repository).
2. Computes DP_Diff, DIR and consistency on the **original labels** of each full dataset.
3. Makes a 70/30 train/test split, stratified on label × protected group.
4. Fits AIF360 `Reweighing` on the **training split only** and trains a logistic regression with the resulting
   instance weights. The protected attribute is not a model input. An unweighted model is trained as baseline.
5. Reports DP_Diff, DIR, consistency, accuracy, balanced accuracy and F1 on the **test split**.
6. Repeats steps 3–5 for seeds 0–9 and reports mean ± std.

| Dataset | Protected attribute | Privileged group | Favourable label |
|---|---|---|---|
| Adult | sex | male | income > 50K |
| German Credit | age | > 25 | good credit |
| COMPAS | race | Caucasian | no recidivism within 2 years |

## Reproduce the results

Requirements: Python 3.10 or 3.11, Git, and an internet connection (the datasets are downloaded on the first run).

```bash
# 1. Clone the fork and go to this folder
git clone https://github.com/yuliia-bobrovytska/AIF360.git
cd AIF360/assignment4_reweighing

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install the pinned dependencies (includes aif360==0.6.1)
pip install -r requirements.txt

# 4. Run the full experiment (3 datasets × 10 seeds)
python run_reweighing.py
```

The script overwrites `results/summary.md`, `results/all_runs.csv` and
`results/fairness_before_after.png`. Because the seeds are fixed, the numbers should match the committed
results (small differences in the last digit are possible with other library versions).

Optional flags:

```bash
python run_reweighing.py --runs 2                  # quick check with 2 seeds
python run_reweighing.py --datasets Adult German   # only some datasets
python run_reweighing.py --out my_results          # other output folder
```

## Metric conventions

- **DP_Diff** = favourable rate of the unprivileged group − favourable rate of the privileged group (0 = parity, negative = unprivileged group disadvantaged).
- **DIR** = ratio of the same two rates (1 = parity, below 0.8 fails the four-fifths rule).
- **Consistency** = AIF360 kNN consistency with k = 5 (1 = similar individuals get the same outcome).

## Reference

Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.
*Knowledge and Information Systems*, 33(1), 1–33. https://doi.org/10.1007/s10115-011-0463-8
