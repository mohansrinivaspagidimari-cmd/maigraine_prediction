"""
Charts for the black book: one dashboard picture, charts/dashboard.png, shown on the Charts tab.
Run:  python charts.py   (train.py also runs it).  The charts follow the trained dataset: Age,
Intensity and Frequency when it has them, otherwise its other number columns or its symptoms.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "charts"
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text())
MAIN = "category" if "category" in METRICS else "screening"  # chart the category dataset when there is one
DATA = pd.read_csv(ROOT / "models" / f"clean_{MAIN}.csv")
TITLES = {"Age": "Age Distribution", "Intensity": "Pain Intensity", "Frequency": "Attacks per Month",
          "Duration": "Attack Duration"}
# short names so long type names fit under the bars and in the confusion matrix
SHORT = {"Typical aura with migraine": "TAM", "Migraine without aura": "MWA", "Familial hemiplegic migraine": "FHM",
         "Typical aura without migraine": "TAWM", "Basilar-type aura": "BTA", "Other": "OTH",
         "Sporadic hemiplegic migraine": "SHM", "Migraine with aura": "MA", "Chronic migraine": "CM",
         "Tension-type headache": "TTH", "Cluster headache": "CH"}

SHORT_NAMES = {}
for label in METRICS[MAIN]["labels"]:  # unknown long names become their initials, kept unique
    name = SHORT.get(label) or (label if len(label) <= 12 else
                                "".join(w[0] for w in label.replace("-", " ").split()).upper())
    while name in SHORT_NAMES.values():
        name += "'"
    SHORT_NAMES[label] = name


def type_distribution(ax):
    counts = DATA["Type"].value_counts()
    ax.bar([SHORT_NAMES[t] for t in counts.index], counts.values)
    ax.set_title("Migraine Type Distribution" if MAIN == "category" else "Migraine Distribution")
    ax.set_ylabel("Number of Records")


def column_chart(ax, column):
    """A bar per value for columns with few whole-number values, otherwise a histogram."""
    values = DATA[column]
    if values.nunique() <= 12 and (values % 1 == 0).all():
        counts = values.value_counts().sort_index()
        if column == "Intensity" and values.max() <= 3:
            ax.bar([["None", "Mild", "Medium", "Severe"][int(v)] for v in counts.index], counts.values)
            ax.set_ylabel("Number of Records")
        else:
            ax.bar(counts.index.astype(int), counts.values, width=0.9, edgecolor="black")
            ax.set_xticks(counts.index.astype(int))
            ax.set_ylabel("Frequency")
    else:
        ax.hist(values, bins=10, edgecolor="black")
        ax.set_ylabel("Frequency")
    ax.set_title(TITLES.get(column, f"{column.replace('_', ' ')} Distribution"))
    ax.set_xlabel(column.replace("_", " "))


def horizontal_bars(ax, names, values, title, xlabel):
    ax.barh([n.replace("_", " ") for n in names], values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def symptom_chart(ax):  # share of patients with each yes/no symptom (the 8 most common)
    share = (DATA[[i["name"] for i in METRICS[MAIN]["inputs"] if i["kind"] == "chip"]].mean() * 100).sort_values().tail(8)
    horizontal_bars(ax, share.index, share.values, "Most Common Symptoms", "Patients with symptom (%)")


def feature_importance(ax):
    top = METRICS[MAIN]["importance"][:10][::-1]
    horizontal_bars(ax, [f["feature"] for f in top], [f["rf"] for f in top], "Top 10 Features (Random Forest)",
                    "Importance")


def grouped_bars(ax, groups, width, fmt, title, ylabel):
    """Bars per model, one colour per group: groups = [(legend label, values per model)]."""
    models = list(METRICS[MAIN]["results"])
    for n, (label, values) in enumerate(groups):
        x = [i + (n - (len(groups) - 1) / 2) * width for i in range(len(models))]
        ax.bar_label(ax.bar(x, values, width, label=label), fmt=fmt, fontsize=7)
    ax.set_xticks(range(len(models)), models, rotation=15)
    ax.set_ylim(0, 125)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="upper center", ncol=len(groups) if len(groups) == 3 else 2)


def accuracy_comparison(ax):
    tasks = [(t, n) for t, n in [("screening", "Migraine or not"), ("category", "Migraine type")] if t in METRICS]
    grouped_bars(ax, [(name, [r["accuracy"] * 100 for r in METRICS[t]["results"].values()]) for t, name in tasks],
                 0.8 / len(tasks), "%.1f", "Model Accuracy Comparison", "Accuracy (%)")


def model_scores(ax):
    results = METRICS[MAIN]["results"].values()
    grouped_bars(ax, [(label, [r[key] * 100 for r in results])
                      for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]],
                 0.27, "%.0f", "Model Scores", "Score (%)")


def confusion(ax):
    m = METRICS[MAIN]
    labels = [SHORT_NAMES[l] for l in m["labels"]]
    ConfusionMatrixDisplay(np.array(m["results"][m["best_model"]]["confusion_matrix"]), display_labels=labels).plot(
        ax=ax, colorbar=False, xticks_rotation=45 if len(labels) > 2 else 0)
    ax.set_title(f"Confusion Matrix ({m['best_model']})")


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.png"):  # charts from an earlier dataset
        old.unlink()
    # three dataset charts, chosen from the columns this dataset has
    sliders = [i["name"] for i in METRICS[MAIN]["inputs"] if i["kind"] == "slider"]
    preferred = [c for c in ("Age", "Intensity", "Frequency") if c in sliders]
    columns = (preferred + [c for c in sliders if c not in preferred])[:3]
    a, b, c = ([lambda ax, c=c: column_chart(ax, c) for c in columns] + [symptom_chart, feature_importance,
                                                                          model_scores])[:3]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))  # one picture: 2 rows x 3 charts
    for ax, draw in zip(axes.flat, [type_distribution, a, accuracy_comparison, b, c, confusion]):
        draw(ax)
    fig.suptitle("Migraine Prediction - ML Dashboard", fontsize=14)
    fig.text(0.5, 0.945, f"Dataset: {METRICS[MAIN]['source']}", ha="center", fontsize=9, color="dimgray")
    key = "   ".join(f"{s} = {t}" for t, s in SHORT_NAMES.items() if s != t)
    if key:
        fig.text(0.5, 0.01, key, ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.04 if key else 0, 1, 0.94))
    fig.savefig(OUT / "dashboard.png", dpi=200)
    plt.close(fig)
    print("Saved", OUT / "dashboard.png")


if __name__ == "__main__":
    main()
    if hasattr(os, "startfile"):  # show the dashboard in the default picture viewer (Windows)
        os.startfile(OUT / "dashboard.png")
