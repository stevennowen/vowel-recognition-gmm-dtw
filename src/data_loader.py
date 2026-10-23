from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import SAMPLE_RATE, TRIM_DB, VOWELS, WIN_LEN, WIN_STEP


def load_wav(path):
    sr, signal = wavfile.read(path)

    if np.issubdtype(signal.dtype, np.integer):
        signal = signal / np.iinfo(signal.dtype).max
    signal = signal.astype(np.float64)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // g, sr // g)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    return signal


def trim_silence(signal, sr=SAMPLE_RATE):
    frame_len = int(WIN_LEN * sr)
    frame_step = int(WIN_STEP * sr)
    if len(signal) <= frame_len:
        return signal

    starts = np.arange(0, len(signal) - frame_len + 1, frame_step)
    energy = np.array([np.sum(signal[s:s + frame_len] ** 2) for s in starts])
    energy_db = 10 * np.log10(energy + 1e-12)

    voiced = np.where(energy_db > energy_db.max() - TRIM_DB)[0]
    start = starts[voiced[0]]
    end = starts[voiced[-1]] + frame_len
    return signal[start:end]


def parse_name(path):
    speaker, vowel = Path(path).stem.rsplit("_", 1)
    vowel = vowel.lower()
    if vowel not in VOWELS:
        raise ValueError(f"Nama file {Path(path).name} tidak sesuai format <speaker>_<vowel>.wav")
    return speaker, vowel


def load_folder(folder):
    items = []
    for path in sorted(Path(folder).glob("*.wav")):
        speaker, vowel = parse_name(path)
        signal = trim_silence(load_wav(path))
        items.append({"file": path.name, "speaker": speaker, "vowel": vowel, "signal": signal})
    return items
