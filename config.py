from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TEMPLATE_DIR = DATA_DIR / "template"
CLOSED_DIR = DATA_DIR / "test" / "closed"
OPEN_DIR = DATA_DIR / "test" / "open"
RESULTS_DIR = ROOT / "results"

VOWELS = ["a", "i", "u", "e", "o"]

SAMPLE_RATE = 16000
TRIM_DB = 30

WIN_LEN = 0.025
WIN_STEP = 0.01
NUM_CEP = 13
NUM_FILTERS = 23
NFFT = 512
PREEMPH = 0.97
DELTA_N = 2

COV_REG = 1e-3
