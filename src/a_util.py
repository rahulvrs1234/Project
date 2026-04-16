"""Clinical-data utility helpers reconstructed from the project snippet."""

from __future__ import annotations

import logging
import math
import os
import re
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("fivethirtyeight")
logging.getLogger().setLevel("INFO")

PATH = "data"
LABEL = "covid19_test_results"
LABEL_VALUES = ["Negative", "Positive"]

SYMPTOMS = [
    "labored_respiration",
    "rhonchi",
    "wheezes",
    "cough",
    "cough_severity",
    "loss_of_smell",
    "loss_of_taste",
    "runny_nose",
    "muscle_sore",
    "sore_throat",
    "fever",
    "sob",
    "sob_severity",
    "diarrhea",
    "fatigue",
    "headache",
    "ctab",
    "days_since_symptom_onset",
]

VITALS = ["temperature", "pulse", "sys", "dia", "rr", "sats"]
COMORBIDITIES = [
    "diabetes",
    "chd",
    "htn",
    "cancer",
    "asthma",
    "copd",
    "autoimmune_dis",
    "smoker",
]
RISKS = ["age", "high_risk_exposure_occupation", "high_risk_interactions"]
TEST_RESULTS = [
    "batch_date",
    LABEL,
    "rapid_flu_results",
    "rapid_strep_results",
    "swab_type",
    "test_name",
]
CXR_FIELDS = ["cxr_findings", "cxr_impression", "cxr_label", "cxr_link"]


def open_data() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(f"{PATH}/{filename}")
            for filename in os.listdir(PATH)
            if filename.endswith(".csv")
        ],
        ignore_index=True,
    )


def get_percent(x: int, total: int) -> float:
    if total == 0:
        logging.info("Returning 0 to avoid division-by-zero error.")
        return 0
    return (x / total) * 100


def filter_pos(data: pd.DataFrame) -> pd.DataFrame:
    return data[data[LABEL] == "Positive"]


def is_any_true(row: pd.Series, cols: List[str]) -> bool:
    return any(row[col] is True for col in cols)


def is_any_nonnull(row: pd.Series, cols: List[str]) -> bool:
    return any(pd.notnull(row[col]) and not math.isnan(row[col]) for col in cols)


def filter_patients(df: pd.DataFrame, cols_to_check: List[str], col_type: str = "bool") -> pd.DataFrame:
    logging.info("Filtering out patients...")

    if col_type == "bool":
        fn = is_any_true
    elif col_type == "numeric":
        fn = is_any_nonnull
    else:
        logging.info("ERROR: `col_type` should be either `bool` or `numeric`.")
        return None

    df_filtered = df[df.apply(lambda row: fn(row, cols_to_check), axis=1)]
    logging.info(
        "    ---- %s --> %s (%.2f%%)",
        len(df),
        len(df_filtered),
        get_percent(len(df_filtered), len(df)),
    )
    return df_filtered


def print_data_info(data: pd.DataFrame):
    for col in data.columns:
        if len(data[col].unique()) == 1:
            logging.info(
                "`%s` only has single unique value of %s in entire dataset.",
                col,
                data[col].iloc[0],
            )


LABELS = [
    "Test Results",
    "Epi Factors",
    "Comorbidities",
    "Vitals",
    "Symptoms",
    "Radiological Findings",
    "Other",
]
COLOR_PALETTE = sns.color_palette("husl", len(LABELS))
COLOR_PALETTE[-1] = "gray"


def add_legend():
    mappings = {label: COLOR_PALETTE[i] for i, label in enumerate(LABELS)}
    patches = [mpatches.Patch(color=color, label=label) for label, color in mappings.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1))


def get_color(col: str) -> str:
    if col in TEST_RESULTS:
        return COLOR_PALETTE[0]
    if col in RISKS:
        return COLOR_PALETTE[1]
    if col in COMORBIDITIES:
        return COLOR_PALETTE[2]
    if col in VITALS:
        return COLOR_PALETTE[3]
    if col in SYMPTOMS:
        return COLOR_PALETTE[4]
    if col in CXR_FIELDS:
        return COLOR_PALETTE[5]
    return "gray"


def plot_fill_rates(data: pd.DataFrame, title: str = ""):
    total = len(data)
    cols = data.columns
    _, ax = plt.subplots(figsize=(7, 15), facecolor="white")
    ax.set_facecolor("white")
    x = range(len(cols))
    y = [sum(~data[col].isnull()) / total for col in cols]
    colors = [get_color(col) for col in cols]
    sns.barplot(x=y, y=list(x), palette=colors, orient="h")
    plt.xlabel("Fill Rate")
    plt.yticks(list(x), cols)
    plt.title(title)
    add_legend()
    plt.show()


ABNORMALITIES = [
    r".+(lobe|RML|peribronchial|basilar) infiltrate",
    "lobe scarring or atelectasis",
    r"(perihilar|Trace).+opacity",
    "Peribronchial thickeneing",
    "Left lower lobe consolidation",
    r"Consolidation in the.+lung",
    r"(?<!No )(Multifocal|lung|pulmonary).+opacities",
    "left pulmonary nodules",
    r"(?<!no ) opacity",
    r".?(left|Left) lung base",
    r"(Subtle left basilar|mass-like spiculated) density",
    "basilar atelectasis or scarring",
    "Elevated right hemidiaphragm",
    "(right hilar|septal) prominence",
]

NO_ABNORMALITIES = [
    r"No.+(acute|significant|definite|suspicious).+(abnormality|disease|opacities)",
    "Normal",
    "No pulmonary opacities visualized",
    "No evidence of acute cardiopulmonary disease",
    "No lobar consolidation",
]


def is_abnormal_cxr(cxr_imp: str):
    if any(re.search(pattern, cxr_imp) for pattern in NO_ABNORMALITIES):
        return False
    if any(re.search(pattern, cxr_imp) for pattern in ABNORMALITIES):
        return True
    return None


SEVERITY_MAPPINGS = {"Mild": 1, "Moderate": 2, "Severe": 3}


def get_sym_severity_score(row: pd.Series) -> int:
    if row["num_symptoms"] == 0:
        return -1
    return (
        SEVERITY_MAPPINGS.get(row["cough_severity"], 0)
        + SEVERITY_MAPPINGS.get(row["sob_severity"], 0)
        + (row["fever"] is True)
    )


def get_sym_severity(score: int) -> str:
    if score < 0:
        return "Asymptomatic"
    if score < 1:
        return "Extremely Mild"
    if score < 2:
        return "Mild"
    if score < 3:
        return "Moderate"
    return "Severe"
