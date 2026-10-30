# Migraine Prediction System using Machine Learning

Field project by Nuzha Logde (56004), Mohan Pagidimari (56005) and Aditya Rane (56006).

A two-step web application trained on the dataset from your reference:

1. **Check migraine:** predicts *Migraine* or *No migraine*.
2. **Migraine category:** the answers carry over, and a second model predicts the most likely category.

Both steps use the best of three models from the synopsis: Decision Tree, Random Forest and SVM.
Tabs: **Check migraine**, **Migraine category**, **Model comparison**, **Dataset**, **Charts**.

## How to run

```bash
pip install -r requirements.txt
python train.py      # trains on the CSV file(s) in data/
python app.py        # open http://127.0.0.1:5051
```

The project contains no built-in data. Until a dataset is trained, every page shows how to add one.

## Add the dataset from your reference

1. Save the dataset as a **CSV file** in the `data` folder. Any file name works (files starting with
   `template` are ignored).
2. It needs a **`Type`** column with the known answer. **Every other column is an input**:
   - columns with only 0/1 (or Yes/No) become symptom buttons;
   - number columns become sliders, with the range taken from the data;
   - text columns (e.g. a patient name or ID) are skipped.
3. Run `python train.py`, then restart `python app.py`.

### Two types of dataset

| Type | `Type` column holds | Trains | Example |
|---|---|---|---|
| **Category dataset** | the migraine category, e.g. "Typical aura with migraine" | Step 2, and step 1 is built from it | `data/migraine_data.csv` (Kaggle) |
| **Migraine or not dataset** (optional) | Migraine / No migraine (or Yes/No, 1/0) | Step 1 | - |

When only a category dataset is present, step 1 is built from it. The types listed in
`NO_MIGRAINE_TYPES` in `train.py` count as *No migraine* ("Typical aura without migraine", "Other"),
and every other type counts as *Migraine*. Edit that list if your reference groups them differently.

### The two forms

Both steps show short, plain-language forms with Age, Headache/attack length (hours), Headaches/attacks
per month, Pain intensity (1 to 10) and symptom buttons. Step 1 asks about pain on one side, throbbing
pain, nausea, light/sound sensitivity, visual warning and family history. Step 2 asks about nausea,
vomiting, light/sound sensitivity, visual disturbance, tingling, dizziness, speech trouble, ringing in
ears and pins and needles. `to_dataset()` in `app.py` converts these answers to your dataset's columns
and scales before predicting; columns that are not asked keep the dataset's typical value. For the Kaggle dataset: hours → days (1-3), pain 1-10 →
intensity 1-3, "Pain on one side" → Location 1 (else 2), "Throbbing pain" → Character 1 (else 2),
"Visual warning" → Visual, "Family history" → DPF. A symptom button is only shown when the dataset has a
matching column (the Kaggle data has none for "Worse with activity"). Step 1 answers pre-fill step 2.

Cleaning is automatic: duplicates removed, impossible ages dropped, missing values filled with the
column median, and Types with fewer than 5 rows left out. The **Dataset** tab shows what was done.

### Current dataset

`data/my_screening_data.csv.csv` is the Kaggle "Migraine Symptom Classification" dataset (400 patients,
23 inputs, 7 migraine types). It is a category dataset, so it trains both steps. Results on the 20% test set:

| Step | Decision Tree | Random Forest | SVM |
|---|---|---|---|
| Migraine or not (accuracy) | 94.9% | **100%** (5-fold CV 98.2%) | 100% |
| Migraine category (accuracy) | 68.4% | **84.8%** | 73.4% |

In this dataset, Duration is in days (1-3), Frequency is attacks per month, and Intensity, Location,
Character, Visual and Sensory are small scales (shown under each slider).

## Charts for the black book

```bash
python charts.py
```

Saves one picture, `charts/dashboard.png`, with all six charts together: migraine type distribution,
three charts from the dataset's columns, model accuracy comparison and the confusion matrix. The
dataset's file name is printed under the title. The picture opens when it is done, `python train.py`
redraws it automatically, and the web app shows it on the **Charts** tab.
Needs matplotlib (`pip install -r requirements.txt`).

### Check that the charts follow the dataset

Replace the CSV in `data/` with another dataset that has a `Type` column, run `python train.py`,
restart `python app.py` and open the Charts tab (Ctrl+F5). The dataset's file name under the title
and every chart change. Put `migraine_data.csv` back and run `python train.py` to return.

The charts follow whatever dataset is trained. Age, Intensity and Frequency get their own charts
when the dataset has them; otherwise its other number columns are charted (e.g. Sleep_hours), and
a dataset without number columns gets "Most common symptoms", feature importance and model scores.
Long type names get short forms with a key under the dashboard. Old charts are deleted first.

With a dataset that has only migraine types (none listed in `NO_MIGRAINE_TYPES`), the
"migraine or not" step is skipped and the app opens at the category step. The forms hide sliders
for columns the dataset lacks and add a slider for each number column they do not know.
Yes/no columns the forms do not know (e.g. `Tearing_eye`) are not asked and count as "no"; add them
to `CHIPS` and `FORM_CHIPS` in `config.py` to ask about them.

## API

| Method | URL | Body (JSON) | Returns |
|---|---|---|---|
| POST | `/api/predict/screening` | the patient's answers | Migraine / No migraine with probabilities |
| POST | `/api/predict/category` | the patient's answers | the migraine type with a probability for each type |

Body fields (all optional, missing ones use typical values): `HeadacheAge`, `HeadacheHours`,
`HeadachesPerMonth`, `PainLevel` (1-10), and symptoms as 1 (yes) or 0 (no): `OneSide`, `Throbbing`,
`Nausea`, `Vomiting`, `LightSensitivity`, `SoundSensitivity`, `VisualAura`, `Tingling`, `Dizziness`,
`SpeechTrouble`, `RingingEars`, `PinsNeedles`, `FamilyHistory`. The dataset's own column names
(e.g. `Age`, `Visual`) are accepted too.

Example response:

```json
{"label": "Migraine", "confidence": 100.0, "model": "Random Forest",
 "probabilities": [{"label": "Migraine", "value": 100.0}, {"label": "No migraine", "value": 0.0}]}
```

Errors: `400` with `{"error": "..."}` for text instead of numbers or a body that is not a JSON object;
`404` for an unknown task.

### Testing the API

**Automated tests** (no server needed, 30 tests in `tests/test_api.py`):

```bash
python -m pytest -v
```

The tests work with any trained dataset. They also load all three saved models (Decision Tree,
Random Forest, SVM) for each step and check that each one predicts valid types and beats guessing the
most common type. Checks that do not apply to a dataset (e.g. no Age column) are shown as SKIPPED.
To check all models on another dataset: swap the CSV in `data/`, run `python train.py`, then
`python -m pytest -v`.

**By hand with curl** while `python app.py` is running (in PowerShell type `curl.exe`):

```bash
curl.exe -X POST http://127.0.0.1:5051/api/predict/screening -H "Content-Type: application/json" -d "{\"HeadacheAge\":30,\"PainLevel\":8,\"Nausea\":1,\"LightSensitivity\":1,\"VisualAura\":1}"
```

**Postman:** method POST, URL `http://127.0.0.1:5051/api/predict/category`, Body → raw → JSON,
paste `{"HeadacheAge": 30, "HeadacheHours": 12, "PainLevel": 8, "Nausea": 1, "VisualAura": 1}`, Send.

## Project structure

```
migraine_web_app_v1/
├── app.py              Flask web app (form built from the dataset's columns)
├── config.py           Column names, sliders and symptom buttons used by app.py
├── train.py            Preprocessing, training, evaluation (DT, RF, SVM)
├── data/               Your reference dataset(s) + the two template types
├── models/             Trained models (*.joblib), metrics.json, cleaned data
├── templates/ static/  Web pages (base, predict, report), CSS, JS
├── tests/test_api.py   API tests (pytest)
├── charts.py           Draws the charts in charts/
```

*Built for academic use. It does not replace a doctor's diagnosis.*
