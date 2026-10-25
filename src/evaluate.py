from pathlib import Path

import pandas as pd

from config import VOWELS

SYSTEMS = ["DTW Template", "Model Gaussian"]
SCENARIOS = ["Closed Recordings", "Closed Speakers", "Open Speakers"]
SHORT_NAMES = {"DTW Template": "DTW", "Model Gaussian": "Gaussian"}


def accuracy_pivot(df, index):
    data = df.assign(column=df["system"].map(SHORT_NAMES) + " " + df["scenario"])
    table = data.pivot_table(index=index, columns="column", values="correct", aggfunc="mean") * 100
    order = [f"{SHORT_NAMES[system]} {scenario}" for system in SYSTEMS for scenario in SCENARIOS]
    return table[[column for column in order if column in table.columns]]


def summary_table(df):
    table = df.pivot_table(index="system", columns="scenario", values="correct", aggfunc="mean") * 100
    table = table.reindex(index=SYSTEMS, columns=[s for s in SCENARIOS if s in table.columns])
    table["Rata-rata"] = table.mean(axis=1)
    return table


def vowel_table(df):
    table = accuracy_pivot(df, "vowel")
    table = table.reindex([vowel for vowel in VOWELS if vowel in table.index])
    table["Average"] = table.mean(axis=1)
    table.loc["Average"] = table.mean()
    return table


def speaker_table(df):
    table = accuracy_pivot(df, "speaker")
    table["Average"] = table.mean(axis=1)
    table.loc["Average"] = table.mean()
    return table


def closed_speaker_table(df):
    return speaker_table(df[df["scenario"] != "Open Speakers"])


def open_speaker_table(df):
    return speaker_table(df[df["scenario"] == "Open Speakers"])


def confusion_table(df, system, scenario):
    data = df[(df["system"] == system) & (df["scenario"] == scenario)]
    table = pd.crosstab(data["vowel"], data["predicted"])
    return table.reindex(index=VOWELS, columns=VOWELS, fill_value=0)


def save_results(df, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "predictions.csv", index=False)

    scenarios = [s for s in SCENARIOS if s in set(df["scenario"])]
    tables = [
        ("summary", "Tabel 1. Akurasi per Sistem (%)", summary_table(df)),
        ("per_vowel", "Tabel 2. Akurasi per Vowel (%)", vowel_table(df)),
        ("closed_speakers", "Tabel 3. Closed Experiments per Speaker (%)", closed_speaker_table(df)),
    ]
    if "Open Speakers" in scenarios:
        tables.append(("open_speakers", "Tabel 4. Open Experiments per Speaker (%)", open_speaker_table(df)))

    for system in SYSTEMS:
        for scenario in scenarios:
            name = f"confusion_{SHORT_NAMES[system]}_{scenario}".lower().replace(" ", "_")
            title = f"Confusion Matrix {system}, {scenario} (baris = label, kolom = prediksi)"
            tables.append((name, title, confusion_table(df, system, scenario)))

    lines = []
    for name, title, table in tables:
        table.to_csv(folder / f"{name}.csv")
        lines += [f"## {title}", "", table.to_markdown(floatfmt=".2f"), ""]
    (folder / "results.md").write_text("\n".join(lines), encoding="utf-8")

    return tables
