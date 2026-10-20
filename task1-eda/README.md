# Task 1 — Exploratory Data Analysis (Titanic)

CodeNova AI/ML Internship · Batch 01

Cleans a public dataset (Titanic passengers, 891 × 15, from [`seaborn-data`](https://github.com/mwaskom/seaborn-data)), visualises it and documents what the data actually says.

## Contents
| File | Purpose |
|---|---|
| `titanic_eda.ipynb` | Fully executed notebook with markdown commentary under every chart |
| `figures/*.png` | Six 300-dpi exported figures |
| `build_nb.py` | Script that regenerates the notebook |
| `data/titanic.csv` | Raw data |

## Features
- Profiling: shape, dtypes, missing-value table, `describe()`
- Cleaning with written justification: `deck` dropped (77% missing); `age` imputed by class × sex median + missing-flag; 2 missing ports filled with the mode; redundant columns dropped
- Duplicates: 107 exact-duplicate rows detected; a **sensitivity check** showed headline rates barely change (female survival 74.2% vs 74.1%), and because there is no passenger ID they are most likely different people, so they are **kept** (switch `REMOVE_DUPS = True` to drop them)
- Outliers: IQR rule, flagged not deleted; `log1p` transform for fare
- Visuals: distributions, survival bars, boxplots, Spearman heatmap, pairplot, family-size chart

## Key insights
- **Sex dominates survival**: women 74% vs men 19%. **Class is second**: 63% (1st) vs 24% (3rd). Combined: 1st-class women 97% vs 3rd-class men 13.5%.
- Fare is heavily skewed (skew 4.79 → 0.39 after `log1p`); 104 of the 116 fare outliers are 1st-class tickets, so they are real, not errors.
- Spearman ρ with survival: sex_female +0.54, pclass −0.34, fare +0.32.
- Family size is non-linear: survival 55–72% for sizes 2–4, 30% solo, ≤33% for 5+.
- Caveat: observational, single historical event — associations are not causal.

## Setup
```bash
pip install -r ../requirements.txt
jupyter notebook titanic_eda.ipynb      # or: jupyter nbconvert --to notebook --execute --inplace titanic_eda.ipynb
```
