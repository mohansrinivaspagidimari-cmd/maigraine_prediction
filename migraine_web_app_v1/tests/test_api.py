"""
API tests for the Migraine Prediction System.

Run from the project folder (the models must be trained first: python train.py):
    python -m pytest -v

They use Flask's test client, so the server does not need to be running.
They work with whatever dataset was trained: checks that only make sense for some datasets
(e.g. an aura type, a duration in days) are skipped when the dataset does not have them.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as webapp  # noqa: E402

# plain answers, as the web forms send them
MIGRAINE = {"HeadacheAge": 30, "HeadacheHours": 12, "HeadachesPerMonth": 4, "PainLevel": 8,
            "OneSide": 1, "Throbbing": 1, "Nausea": 1, "LightSensitivity": 1,
            "SoundSensitivity": 1, "VisualAura": 1, "FamilyHistory": 1}
MILD = {"HeadacheAge": 40, "HeadacheHours": 2, "HeadachesPerMonth": 6, "PainLevel": 3,
        "OneSide": 0, "Throbbing": 0, "Nausea": 0, "LightSensitivity": 0, "SoundSensitivity": 0}


@pytest.fixture(scope="module")
def client():
    if not webapp.METRICS:
        pytest.skip("No trained model. Run python train.py first.")
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as c:
        yield c


def post(client, task, body):
    return client.post(f"/api/predict/{task}", json=body)


def needs(task):
    if task not in webapp.METRICS:
        pytest.skip(f"this dataset has no '{task}' step")


def spec(column):
    """How the trained dataset stores a column (its slider range), or skip if it has none."""
    inputs = {i["name"]: i for i in webapp.METRICS["category" if "category" in webapp.METRICS else "screening"]["inputs"]}
    if column not in inputs or inputs[column]["kind"] != "slider":
        pytest.skip(f"this dataset has no {column} column")
    return inputs[column]


def main_task():
    return "category" if "category" in webapp.METRICS else "screening"


# ------------------------------------------------------ response format ---
@pytest.mark.parametrize("task", ["screening", "category"])
def test_response_format(client, task):
    needs(task)
    r = post(client, task, MIGRAINE)
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"label", "confidence", "probabilities", "model"}
    assert body["model"] in {"Decision Tree", "Random Forest", "SVM"}
    assert body["label"] in [p["label"] for p in body["probabilities"]]
    assert body["confidence"] == max(p["value"] for p in body["probabilities"])
    assert sum(p["value"] for p in body["probabilities"]) == pytest.approx(100, abs=0.5)


# ------------------------------------------------- step 1: migraine or not ---
def test_screening_migraine(client):
    needs("screening")
    assert post(client, "screening", MIGRAINE).get_json()["label"] == "Migraine"


def test_screening_no_migraine(client):
    needs("screening")
    assert post(client, "screening", MILD).get_json()["label"] == "No migraine"


def test_screening_two_classes(client):
    needs("screening")
    labels = {p["label"] for p in post(client, "screening", MIGRAINE).get_json()["probabilities"]}
    assert labels == {"Migraine", "No migraine"}


# ----------------------------------------------- step 2: migraine category ---
def test_category_has_every_type(client):
    needs("category")
    labels = {p["label"] for p in post(client, "category", MIGRAINE).get_json()["probabilities"]}
    assert labels == set(webapp.METRICS["category"]["labels"])


def test_category_with_aura(client):
    # a migraine with visual aura should land on the dataset's "with aura" type
    needs("category")
    aura_types = [l for l in webapp.METRICS["category"]["labels"] if "aura" in l.lower() and "without" not in l.lower()]
    if not aura_types:
        pytest.skip("this dataset has no 'with aura' type")
    assert post(client, "category", MIGRAINE).get_json()["label"] in aura_types


def test_category_without_aura(client):
    needs("category")
    if "Migraine without aura" not in webapp.METRICS["category"]["labels"]:
        pytest.skip("this dataset has no 'Migraine without aura' type")
    body = {**MIGRAINE, "VisualAura": 0}
    assert post(client, "category", body).get_json()["label"] == "Migraine without aura"


def test_category_accepts_dataset_columns(client):
    needs("category")
    # the API also takes the dataset's own column names directly
    r = post(client, "category", {"Age": 30, "Duration": 1, "Intensity": 3, "Visual": 2, "Nausea": 1})
    assert r.status_code == 200


# ------------------------------------------------------------ conversion ---
def test_duration_unit(client):
    # datasets that record attacks in days (max 3) get days, others keep hours
    in_days = spec("Duration")["max"] <= 3
    row = webapp.to_dataset({"HeadacheHours": 30}, main_task())
    assert row["Duration"] == (2 if in_days else 30)


def test_pain_scale(client):
    top = spec("Intensity")["max"]
    assert webapp.to_dataset({"PainLevel": 10}, main_task())["Intensity"] == min(top, 10)
    assert webapp.to_dataset({"PainLevel": 1}, main_task())["Intensity"] == 1


def test_values_are_kept_in_range(client):
    age, frequency = spec("Age"), spec("Frequency")
    row = webapp.to_dataset({"HeadacheAge": 500, "HeadachesPerMonth": 99}, main_task())
    assert row["Age"] == age["max"]
    assert row["Frequency"] == frequency["max"]


def test_empty_body_uses_defaults(client):
    assert post(client, main_task(), {}).status_code == 200


# ---------------------------------------------------------------- errors ---
def test_unknown_task(client):
    r = post(client, "nonsense", MIGRAINE)
    assert r.status_code == 404
    assert "Unknown task" in r.get_json()["error"]


def test_text_instead_of_number(client):
    r = post(client, main_task(), {**MIGRAINE, "HeadacheAge": "abc"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_json_list_instead_of_object(client):
    assert post(client, main_task(), [1, 2, 3]).status_code == 400


def test_not_json(client):
    r = client.post(f"/api/predict/{main_task()}", data="hello", content_type="text/plain")
    assert r.status_code == 400


def test_get_not_allowed(client):
    assert client.get(f"/api/predict/{main_task()}").status_code == 405


# ----------------------------------------------------------------- pages ---
@pytest.mark.parametrize("path", ["/", "/category", "/models", "/dataset", "/charts"])
def test_pages_load(client, path):
    assert client.get(path, follow_redirects=True).status_code == 200


def test_step1_answers_reach_step2(client):
    needs("screening")
    needs("category")
    html = client.get("/category?HeadacheAge=30&Nausea=1&OneSide=1").get_data(as_text=True)
    assert 'name="OneSide" value="1"' in html  # not asked on step 2, carried as a hidden field
    assert 'value="30"' in html                # age slider pre-filled


# ------------------------------------------------- all three models work ---
MODEL_FILES = {"Decision Tree": "decision_tree", "Random Forest": "random_forest", "SVM": "svm"}


@pytest.mark.parametrize("model_name", MODEL_FILES)
@pytest.mark.parametrize("task", ["screening", "category"])
def test_every_model_works(client, task, model_name):
    """Each saved model loads, predicts a valid type for every patient, and beats guessing."""
    needs(task)
    m = webapp.METRICS[task]
    model = joblib.load(webapp.MODELS / f"{task}_{MODEL_FILES[model_name]}.joblib")
    data = pd.read_csv(webapp.MODELS / f"clean_{task}.csv")
    predictions = model.predict(data[m["features"]])
    assert set(predictions) <= set(m["labels"])

    # better than always answering the most common type
    guess = data["Type"].value_counts(normalize=True).max()
    accuracy = m["results"][model_name]["accuracy"]
    assert accuracy > guess, f"{model_name}: test accuracy {accuracy:.1%} is not better than guessing ({guess:.1%})"
