"""Names and form settings used by app.py."""

# Friendly names for common dataset columns; any other column shows its own name
LABELS = {
    "Age": "Age", "Duration": "Attack duration", "Frequency": "Attacks per month",
    "Location": "Pain location", "Character": "Pain character", "Intensity": "Pain intensity",
    "Nausea": "Nausea", "Vomit": "Vomiting", "Phonophobia": "Sound sensitivity",
    "Photophobia": "Light sensitivity", "Visual": "Visual symptoms", "Sensory": "Sensory symptoms",
    "Dysphasia": "Speech trouble", "Dysarthria": "Slurred speech", "Vertigo": "Dizziness",
    "Tinnitus": "Ringing in ears", "Hypoacusis": "Hearing loss", "Diplopia": "Double vision",
    "Visual_defect": "Visual field defect", "Defect": "Visual field defect", "Ataxia": "Loss of balance",
    "Conscience": "Reduced consciousness", "Paresthesia": "Pins and needles", "DPF": "Family history",
    "FamilyHistory": "Family history", "Unilateral": "Pain on one side", "Pulsating": "Throbbing pain",
    "ActivityWorse": "Worse with activity", "Aura": "Visual warning (aura)",
}

# The forms ask plain questions; app.to_dataset converts the answers to the dataset's columns and scales.
# slider -> (dataset column, step 1 label, step 2 label, min, max, step, default)
SLIDERS = {
    "HeadacheAge": ("Age", "Age", "Age", 5, 90, 1, 25),
    "HeadacheHours": ("Duration", "Headache length (hours)", "Attack length (hours)", 0.5, 72, 0.5, 6),
    "HeadachesPerMonth": ("Frequency", "Headaches per month", "Attacks per month", 1, 30, 1, 4),
    "PainLevel": ("Intensity", "Pain intensity (1 to 10)", "Pain intensity (1 to 10)", 1, 10, 1, 5),
}

# symptom button -> label, and the dataset columns it can fill (the first one the dataset has is used)
CHIPS = {
    "OneSide": ("Pain on one side", ["Location", "Unilateral"]),
    "Throbbing": ("Throbbing pain", ["Character", "Pulsating"]),
    "ActivityWorse": ("Worse with activity", ["ActivityWorse"]),
    "Nausea": ("Nausea", ["Nausea"]),
    "Vomiting": ("Vomiting", ["Vomit"]),
    "LightSensitivity": ("Light sensitivity", ["Photophobia"]),
    "SoundSensitivity": ("Sound sensitivity", ["Phonophobia"]),
    "VisualAura": ("Visual disturbance", ["Visual", "Aura"]),
    "Tingling": ("Tingling or numbness", ["Sensory"]),
    "Dizziness": ("Dizziness", ["Vertigo"]),
    "SpeechTrouble": ("Speech trouble", ["Dysphasia"]),
    "RingingEars": ("Ringing in ears", ["Tinnitus"]),
    "PinsNeedles": ("Pins and needles", ["Paresthesia"]),
    "FamilyHistory": ("Family history", ["DPF", "FamilyHistory"]),
}

# which symptom buttons each step shows
FORM_CHIPS = {
    "screening": ["OneSide", "Throbbing", "ActivityWorse", "Nausea", "LightSensitivity",
                  "SoundSensitivity", "VisualAura", "FamilyHistory"],
    "category": ["Nausea", "Vomiting", "LightSensitivity", "SoundSensitivity", "VisualAura",
                 "Tingling", "Dizziness", "SpeechTrouble", "RingingEars", "PinsNeedles"],
}
