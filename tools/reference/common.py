"""Audio helpers shared by the reference and validation scripts."""
import json
import wave
from pathlib import Path

import numpy as np
from pesq import pesq
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "build" / "corpus"
MODEL_DIR = ROOT / "model"
LYRA_CLIPS = sorted(Path(r"C:\Users\jon\source\repos\lyra\testdata").glob("*.wav"))


def read_wav(path) -> tuple[np.ndarray, int]:
    """Mono 16-bit WAV as float32 in [-1, 1) and its sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data.astype(np.float32) / 32768, w.getframerate()


def write_wav(path, samples: np.ndarray, rate: int):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(to_pcm16(samples).tobytes())


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(samples) * 32768), -32768, 32767).astype("<i2")


def up_16k_to_24k(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, 3, 2).astype(np.float32)


def down_24k_to_16k(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, 2, 3).astype(np.float32)


def corpus(set_name: str) -> list[Path]:
    index = json.loads((CORPUS / set_name / "INDEX.json").read_text())
    return [CORPUS / set_name / entry["file"] for entry in index]


def pesq_wb(reference16: np.ndarray, decoded16: np.ndarray, delay: int = 0) -> float:
    """PESQ-WB of 16 kHz audio, the decoded signal delayed by `delay` samples."""
    decoded16 = decoded16[delay:]
    n = min(len(reference16), len(decoded16))
    return pesq(16000, reference16[:n], decoded16[:n], "wb")


def best_lag(reference: np.ndarray, decoded: np.ndarray, max_lag: int = 960, step: int = 1) -> int:
    """Delay of `decoded` behind `reference` maximising their correlation."""
    best, lag = -2.0, 0
    for shift in range(0, max_lag + 1, step):
        a, b = reference, decoded[shift:]
        n = min(len(a), len(b))
        c = np.dot(a[:n], b[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-9)
        if c > best:
            best, lag = c, shift
    return lag
