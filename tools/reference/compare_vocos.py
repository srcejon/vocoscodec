import time
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from pesq import pesq
from vocos import Vocos

torch.set_num_threads(1)
HERE = Path(__file__).parent
CLIPS = sorted(Path(r"C:\Users\jon\source\repos\lyra\testdata").glob("*.wav"))


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def score(reference, decoded):
    best, lag = -1, 0
    for shift in range(-960, 961, 8):
        a = reference[max(0, shift):]
        b = decoded[max(0, -shift):]
        n = min(len(a), len(b))
        c = np.dot(a[:n], b[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-9)
        if c > best:
            best, lag = c, shift
    a = reference[max(0, lag):]
    b = decoded[max(0, -lag):]
    n = min(len(a), len(b))
    return pesq(16000, a[:n], b[:n], "wb")


up = torchaudio.transforms.Resample(16000, 24000)
down = torchaudio.transforms.Resample(24000, 16000)
encodec = EncodecModel.encodec_model_24khz().eval()
vocos_codec = Vocos.from_pretrained("charactr/vocos-encodec-24khz").eval()
vocos_mel = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

results, timing = {}, {}


def record(name, clip, reference, audio24, seconds, elapsed):
    decoded = down(audio24.reshape(1, -1)).reshape(-1).numpy()
    write_wav(HERE / f"{clip.stem}_{name.replace(' ', '_').replace(',', '')}.wav", decoded)
    results.setdefault(name, []).append(score(reference, decoded))
    timing.setdefault(name, []).append(100 * elapsed / seconds)


with torch.no_grad():
    for clip in CLIPS:
        reference = read_wav(clip)
        seconds = len(reference) / 16000
        x = up(torch.from_numpy(reference).reshape(1, -1))

        t = time.perf_counter()
        mel = vocos_mel.feature_extractor(x)
        y = vocos_mel.decode(mel)
        record("Vocos mel, unquantised", clip, reference, y, seconds, time.perf_counter() - t)

        encodec.set_target_bandwidth(12.0)
        t = time.perf_counter()
        frames = encodec.encode(x.reshape(1, 1, -1))
        encode_time = time.perf_counter() - t
        codes = frames[0][0]  # [1, 16, T]
        for n_q, bandwidth_id, label in ((2, 0, "1.5k"), (3, 0, "2.25k (3 codebooks, 1.5k embedding)"),
                                         (3, 1, "2.25k (3 codebooks, 3k embedding)"), (4, 1, "3.0k"),
                                         (8, 2, "6.0k")):
            t = time.perf_counter()
            features = vocos_codec.codes_to_features(codes[0, :n_q].reshape(n_q, -1))
            y = vocos_codec.decode(features, bandwidth_id=torch.tensor([bandwidth_id]))
            record(f"EnCodec + Vocos {label}", clip, reference, y, seconds,
                   encode_time + time.perf_counter() - t)
            if n_q in (2, 4):
                t = time.perf_counter()
                y = encodec.decode([(codes[:, :n_q], None)])
                record(f"EnCodec own decoder {label}", clip, reference, y, seconds,
                       encode_time + time.perf_counter() - t)

for name in results:
    print(f"{name:48} PESQ-WB {np.mean(results[name]):.2f} ({', '.join(f'{s:.2f}' for s in results[name])})"
          f"   CPU {np.mean(timing[name]):.0f}% of one core")
