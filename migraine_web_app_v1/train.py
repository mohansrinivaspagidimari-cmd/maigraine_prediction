"""
Training pipeline for the Migraine Prediction System.   Run:  python train.py

Every CSV in data/ (except template*) needs a "Type" column; every other column is a feature.
A file whose Type is Migraine / No migraine trains step 1; a file with migraine types trains step 2.
Without a step 1 file, step 1 is built from the types: NO_MIGRAINE_TYPES count as "No migraine".

Research design from the synopsis: preprocessing (Yes/No -> 1/0, duplicates, invalid ages, missing
values) -> feature importance -> Decision Tree, Random Forest, SVM -> accuracy, precision, recall,
F1-score, k-fold CV -> best model = highest macro F1-score on the test set.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA, MODELS = ROOT / "data", ROOT / "models"
# types that mean "no migraine" when step 1 is built from a category dataset (lower case)
NO_MIGRAINE_TYPES = ["typical aura without migraine", "other", "no migraine", "none", "normal"]
# notes shown under the sliders for the Kaggle "Migraine Symptom Classification" columns
SCALES = {"Location": "0 none · 1 one side · 2 both sides", "Character": "0 none · 1 throbbing · 2 constant",
          "Intensity": "0 none · 1 mild · 2 medium · 3 severe", "Visual": "number of visual symptoms",
          "Sensory": "number of sensory symptoms"}
YES = {"yes", "y", "true", "present", "migraine"}
NO = {"no", "n", "false", "absent", "no migraine", "nomigraine"}
TITLES = {"screening": "Migraine or not", "category": "Migraine category"}


def build_models():  # class_weight="balanced" gives small classes a fair say; SVM needs scaled inputs
    return {"Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=3, class_weight="balanced",
                                                    random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=1,
                                                    class_weight="balanced", random_state=42),
            "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, probability=True,
                                                       class_weight="balanced", random_state=42))}


def to_number(value):
    text = str(value).strip().lower()
    if pd.isna(value) or text in YES or text in NO:
        return np.nan if pd.isna(value) else int(text in YES)
    try:
        return float(text)
    except ValueError:
        return np.nan


def screening_label(value):
    text = str(value).strip().lower()
    if text in NO or text in {"0", "0.0"} or text.startswith("no"):
        return "No migraine"
    return "Migraine" if text in YES or text in {"1", "1.0"} or "migraine" in text else None


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    type_col = next((c for c in df.columns if c.lower() == "type"), None)
    if type_col is None:
        raise SystemExit(f"\n{path.name} has no 'Type' column (the known answer). Add it and run again.")
    df = df.rename(columns={type_col: "Type"})
    df["Type"] = df["Type"].astype(str).str.strip()
    df = df[~df["Type"].isin(["", "nan"])]
    converted = {c: df[c].map(to_number) for c in df.columns if c != "Type"}
    dropped = [c for c, v in converted.items() if v.isna().mean() > 0.5]  # text columns, e.g. a patient id
    df = df.drop(columns=dropped)
    for c in df.columns.drop("Type"):
        df[c] = converted[c]
    return df, dropped


def find_datasets():
    files = sorted(p for p in DATA.glob("*.csv") if not p.name.lower().startswith("template"))
    screening = category = None
    for path in files:
        df, dropped = load(path)
        labels = df["Type"].map(screening_label)
        if df["Type"].nunique() == 2 and labels.notna().all() and labels.nunique() == 2:
            df["Type"] = labels
            screening = screening or (path, df, dropped)
        else:
            category = category or (path, df, dropped)
    return files, screening, category


def preprocess(df):
    features, report = [c for c in df.columns if c != "Type"], {"raw_rows": int(len(df))}
    duplicated = df.duplicated()
    if duplicated.mean() > 0.5:  # most rows repeat: they are different patients with the same answers
        report["duplicates_removed"] = 0
        print("  duplicate rows kept: most rows repeat, so they are different patients with the same answers")
    else:
        report["duplicates_removed"], df = int(duplicated.sum()), df[~duplicated]
    invalid = df["Age"].notna() & ~df["Age"].between(1, 120) if "Age" in df.columns else df["Type"].isna()
    report["invalid_removed"], df = int(invalid.sum()), df[~invalid]
    report["missing_filled"] = int(df[features].isna().sum().sum())
    for col in features:
        df[col] = df[col].fillna(df[col].median())
    counts = df["Type"].value_counts()  # types with too few rows cannot be split into train and test
    report["small_classes_removed"] = counts[counts < 5].index.tolist()
    df = df[~df["Type"].isin(report["small_classes_removed"])].reset_index(drop=True)
    report["clean_rows"] = int(len(df))
    return df, features, report


def is_chip(values):
    return set(values.unique()) <= {0, 1}


def input_spec(df, features):
    """How each feature appears in the web form: sliders first, then 0/1 symptom buttons."""
    sliders = []
    for f in (f for f in features if not is_chip(df[f])):
        step = 1 if (df[f] % 1 == 0).all() else 0.5
        sliders.append({"name": f, "kind": "slider", "min": float(df[f].min()), "max": float(df[f].max()),
                        "step": step, "default": round(float(df[f].median()) / step) * step,
                        "scale": SCALES.get(f, "")})
    return sliders + [{"name": f, "kind": "chip"} for f in features if is_chip(df[f])]


def evaluate(model, X, y, split, labels, folds):
    X_train, X_test, y_train, y_test = split
    pred = model.fit(X_train, y_train).predict(X_test)
    cv = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y_test, pred),
            **{k: fn(y_test, pred, average="macro", zero_division=0)
               for k, fn in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]},
            "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
            "per_class": {k: v for k, v in report.items() if k in labels}}


def run_task(name, raw, source, dropped, note=""):
    print(f"\n[{TITLES[name]}]  data: {source}")
    df, features, prep = preprocess(raw)
    prep["ignored_columns"] = dropped
    df.to_csv(MODELS / f"clean_{name}.csv", index=False)
    X, y = df[features], df["Type"]
    labels = sorted(y.unique())
    if len(labels) < 2:
        raise SystemExit(f"\n{source}: needs at least 2 different Types with 5+ rows each.")
    folds = int(min(5, y.value_counts().min()))
    split = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    file = lambda model: f"{name}_{model.lower().replace(' ', '_')}.joblib"

    results = {}
    for model_name, model in build_models().items():
        r = results[model_name] = evaluate(model, X, y, split, labels, folds)
        joblib.dump(model, MODELS / file(model_name))
        print(f"  {model_name:<14} acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
              f"rec={r['recall']:.3f}  f1={r['f1']:.3f}  cv={r['cv_mean']:.3f}")
    best = max(results, key=lambda m: results[m]["f1"])
    print(f"  -> best model: {best}")

    rf = joblib.load(MODELS / file("Random Forest"))
    importance = sorted([{"feature": f, "rf": float(i), "mi": float(m)} for f, i, m in
                         zip(features, rf.feature_importances_, mutual_info_classif(X, y, random_state=42))],
                        key=lambda d: d["rf"], reverse=True)
    return {"source": source, "note": note, "features": features, "inputs": input_spec(df, features),
            "labels": labels, "best_model": best, "best_file": file(best), "results": results,
            "importance": importance, "preprocessing": prep, "cv_folds": folds,
            "train_rows": int(len(split[0])), "test_rows": int(len(split[1])),
            "class_counts": {k: int(v) for k, v in y.value_counts().items()},
            "numeric_summary": {c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
                                for c in features if not is_chip(df[c])},
            "sample": df.head(8).to_dict(orient="records")}


def main():
    MODELS.mkdir(exist_ok=True)
    for old in MODELS.glob("*"):
        old.unlink()
    files, screening, category = find_datasets()
    if not files:
        raise SystemExit("\nNo dataset found. Put your reference CSV file in the data/ folder "
                         "with a Type column and run python train.py again.")

    metrics = {}
    if screening:
        path, df, dropped = screening
        metrics["screening"] = run_task("screening", df, path.name, dropped)
    elif category:  # build step 1 from the category dataset
        path, df, dropped = category
        no_types = sorted(t for t in df["Type"].unique() if t.lower() in NO_MIGRAINE_TYPES)
        if no_types:
            derived = df.assign(Type=np.where(df["Type"].str.lower().isin(NO_MIGRAINE_TYPES), "No migraine", "Migraine"))
            metrics["screening"] = run_task("screening", derived, path.name, dropped,
                                            "Built from the category dataset: " + ", ".join(no_types) +
                                            " counted as No migraine, all other types as Migraine.")
        else:
            print("\n[Migraine or not]  skipped: none of the types is in NO_MIGRAINE_TYPES. "
                  "The app starts at the migraine category step.")
    if category:
        path, df, dropped = category
        metrics["category"] = run_task("category", df, path.name, dropped)

    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved models and metrics to", MODELS)
    if metrics:  # redraw the black-book charts so they match the new models
        try:
            import charts
            charts.main()
        except ImportError:
            print("Charts skipped: install matplotlib (pip install matplotlib), then run python charts.py")
        except Exception as error:  # a chart problem must never stop training
            print(f"Charts skipped ({error}). The models are trained; run python charts.py to see the details.")


if __name__ == "__main__":
    main()
