import sys

import pandas as pd

from config import CLOSED_DIR, OPEN_DIR, RESULTS_DIR, TEMPLATE_DIR, VOWELS
from src.data_loader import load_folder
from src.evaluate import save_results
from src.features import extract_mfcc39
from src.recognizer import build_dtw_templates, build_gaussian_models, recognize_dtw, recognize_gaussian


def load_features(folder):
    items = load_folder(folder)
    for item in items:
        item["features"] = extract_mfcc39(item["signal"])
    return items


def run(template_dir=TEMPLATE_DIR, closed_dir=CLOSED_DIR, open_dir=OPEN_DIR, results_dir=RESULTS_DIR):
    train = load_features(template_dir)
    missing = set(VOWELS) - {item["vowel"] for item in train}
    if missing:
        sys.exit(f"Data template di {template_dir} belum lengkap. Vowel yang belum ada: {', '.join(sorted(missing))}")

    templates = build_dtw_templates(train)
    models = build_gaussian_models(train)

    systems = {
        "DTW Template": lambda features: recognize_dtw(features, templates)[0],
        "Model Gaussian": lambda features: recognize_gaussian(features, models)[0],
    }
    scenarios = {
        "Closed Recordings": train,
        "Closed Speakers": load_features(closed_dir),
        "Open Speakers": load_features(open_dir),
    }

    rows = []
    for scenario, items in scenarios.items():
        if not items:
            print(f"Skenario {scenario} dilewati karena foldernya masih kosong.")
            continue
        for system, recognize in systems.items():
            for item in items:
                predicted = recognize(item["features"])
                rows.append({
                    "system": system,
                    "scenario": scenario,
                    "file": item["file"],
                    "speaker": item["speaker"],
                    "vowel": item["vowel"],
                    "predicted": predicted,
                    "correct": predicted == item["vowel"],
                })

    tables = save_results(pd.DataFrame(rows), results_dir)

    for _, title, table in tables[:2]:
        print(f"\n{title}")
        print(table.to_markdown(floatfmt=".2f"))
    print(f"\nHasil lengkap tersimpan di {results_dir}")


if __name__ == "__main__":
    run()
