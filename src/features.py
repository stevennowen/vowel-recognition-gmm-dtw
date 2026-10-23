import numpy as np
from python_speech_features import delta, mfcc

from config import DELTA_N, NFFT, NUM_CEP, NUM_FILTERS, PREEMPH, SAMPLE_RATE, WIN_LEN, WIN_STEP


def extract_mfcc39(signal, sr=SAMPLE_RATE):
    static = mfcc(
        signal,
        samplerate=sr,
        winlen=WIN_LEN,
        winstep=WIN_STEP,
        numcep=NUM_CEP,
        nfilt=NUM_FILTERS,
        nfft=NFFT,
        preemph=PREEMPH,
        appendEnergy=True,
        winfunc=np.hamming,
    )
    delta1 = delta(static, DELTA_N)
    delta2 = delta(delta1, DELTA_N)
    return np.hstack([static, delta1, delta2])
