"""
Migraine Prediction System - Flask web application.

  /           Do you have migraine?      /models    Decision Tree vs Random Forest vs SVM
  /category   Which migraine category?   /dataset   dataset and preprocessing summary
  /charts     dashboard picture          /api/predict/<screening|category>   JSON API

Run:  python train.py   then   python app.py   and open http://127.0.0.1:5051
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

from config import CHIPS, FORM_CHIPS, LABELS, SLIDERS

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
METRICS = json.loads((MODELS / "metrics.json").read_text()) if (MODELS / "metrics.json").exists() else {}
BEST = {task: joblib.load(MODELS / METRICS[task]["best_file"]) for task in METRICS}
COVERED = {s[0] for s in SLIDERS.values()} | {c for _, columns in CHIPS.values() for c in columns}


def extra_sliders(task):
    """Number columns of this dataset that the plain form does not know, e.g. Sleep_hours."""
    return [i for i in METRICS[task]["inputs"] if i["kind"] == "slider" and i["name"] not in COVERED]


def plain_form(task):
    """Inputs for a step, keeping only the sliders and symptoms the dataset has a column for."""
    features, label = METRICS[task]["features"], 2 if task == "category" else 1
    return [{"name": n, "label": s[label], "min": s[3], "max": s[4], "step": s[5], "default": s[6], "kind": "slider",
             "scale": ""} for n, s in SLIDERS.items() if s[0] in features] + extra_sliders(task) + [
        {"name": k, "label": "Visual warning (aura)" if (task, k) == ("screening", "VisualAura") else CHIPS[k][0],
         "kind": "chip"} for k in FORM_CHIPS[task] if any(c in features for c in CHIPS[k][1])]


def to_dataset(answers, task):
    """Convert plain answers to the dataset's columns and scales; unasked columns keep typical values."""
    spec = {i["name"]: i for i in METRICS[task]["inputs"]}
    row = {f: spec[f]["default"] if spec[f]["kind"] == "slider" else 0.0 for f in METRICS[task]["features"]}
    number = lambda key, fallback: float(answers.get(key, fallback) or fallback)

    def fit(column, value):  # set a slider column, kept inside the dataset's range
        if spec.get(column, {}).get("kind") == "slider":
            row[column] = float(min(max(value, spec[column]["min"]), spec[column]["max"]))

    hours, pain = number("HeadacheHours", 6), number("PainLevel", 5)
    fit("Age", number("HeadacheAge", 25))
    fit("Frequency", number("HeadachesPerMonth", 4))
    if "Duration" in spec:  # datasets that record duration in days (max 3) get days, others get hours
        fit("Duration", -(-hours // 24) if spec["Duration"]["max"] <= 3 else hours)
    if "Intensity" in spec:
        top = spec["Intensity"]["max"]
        fit("Intensity", max(1, round(pain / 10 * top)) if top < 10 else pain)
    for i in extra_sliders(task):
        if i["name"] in answers:
            fit(i["name"], number(i["name"], i["default"]))
    for key, (_, columns) in CHIPS.items():
        column = next((c for c in columns if c in row), None)
        if column and key in answers:
            on = float(answers[key] or 0) == 1
            # Location / Character: 1 one side / throbbing, 2 both sides / constant
            row[column] = (1.0 if on else 2.0) if column in ("Location", "Character") else float(on)
    return row


def predict(task, payload):
    features, model = METRICS[task]["features"], BEST[task]
    probs = model.predict_proba(pd.DataFrame([{f: float(payload.get(f, 0) or 0) for f in features}]))[0]
    return {"label": model.classes_[probs.argmax()], "confidence": round(float(probs.max()) * 100, 1),
            "probabilities": [{"label": c, "value": round(float(p) * 100, 1)} for c, p in zip(model.classes_, probs)],
            "model": METRICS[task]["best_model"]}


class Labels(dict):
    def __missing__(self, key):
        return CHIPS[key][0] if key in CHIPS else str(key).replace("_", " ")


app = Flask(__name__)
app.context_processor(lambda: {"labels": Labels(LABELS), "nav": request.endpoint,
                               "has_category": "category" in METRICS, "has_screening": "screening" in METRICS})


@app.route("/")
def screening():
    if "screening" not in METRICS:  # a dataset with only migraine types starts at the category step
        return redirect(url_for("category")) if "category" in METRICS else render_template("base.html")
    return render_template("predict.html", step="screening", inputs=plain_form("screening"), prefill={}, carried={})


@app.route("/category")
def category():
    if "category" not in METRICS:
        return render_template("base.html")
    # step 1 answers pre-fill the form; the ones this form does not ask travel as hidden fields
    prefill, inputs = dict(request.args), plain_form("category")
    carried = {k: v for k, v in prefill.items() if k in CHIPS and k not in {i["name"] for i in inputs}}
    return render_template("predict.html", step="category", inputs=inputs, prefill=prefill, carried=carried)


def report(page):
    if not METRICS:
        return render_template("base.html")
    charts = sorted((p.name for p in (ROOT / "charts").glob("*.png")), key=lambda n: (n != "dashboard.png", n))
    return render_template("report.html", page=page, metrics=METRICS, files=charts)


for page in ("models", "dataset", "charts"):
    app.add_url_rule(f"/{page}", page, lambda page=page: report(page))
app.add_url_rule("/charts/<path:name>", "chart_file", lambda name: send_from_directory(ROOT / "charts", name))


@app.post("/api/predict/<task>")
def api_predict(task):
    payload = request.get_json(force=True)
    for wrong, message, code in [
        (task not in ("screening", "category"), "Unknown task. Use /api/predict/screening or /api/predict/category", 404),
        (task not in METRICS, "No dataset trained yet. Add your CSV to data/ and run python train.py", 404),
        (not isinstance(payload, dict), 'Send a JSON object, e.g. {"HeadacheAge": 30, "Nausea": 1}', 400)]:
        if wrong:
            return jsonify({"error": message}), code
    # the forms send plain answers; the API also accepts the dataset's own columns directly
    plain = any(k in payload for k in ["HeadacheAge", "HeadacheHours", *CHIPS])
    try:
        return jsonify(predict(task, to_dataset(payload, task) if plain else payload))
    except (TypeError, ValueError):
        return jsonify({"error": "Every value must be a number (symptoms: 1 = yes, 0 = no)"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5051)
