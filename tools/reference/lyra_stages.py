import json
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pesq import pesq

LYRA = Path(r"C:\Users\jon\source\repos\lyra")
MODEL = LYRA / "lyra_codec" / "model"
OUT = Path(sys.argv[1])
FRAME = 320


class Streaming:
    def __init__(self, path):
        manifest = json.loads(path.with_suffix(".json").read_text())
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = manifest["input"]["name"]
        self.output_name = manifest["output"]["name"]
        self.states = manifest["states"]
        self.values = {s["input"]: np.zeros(s["shape"], dtype=np.float32) for s in self.states}

    def run(self, value):
        names = [self.output_name] + [s["output"] for s in self.states]
        results = self.session.run(names, {self.input_name: value.astype(np.float32), **self.values})
        self.values = {s["input"]: results[i + 1] for i, s in enumerate(self.states)}
        return results[0]


def load_codebook():
    data = (MODEL / "quantizer_codebook.bin").read_bytes()
    values = np.frombuffer(data[28:], dtype="<f4")
    return values.reshape(46, 16, 64)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.clip(samples * 32768.0, -32768, 32767).astype("<i2").tobytes())


def encode(samples, codebook):
    encoder = Streaming(MODEL / "soundstream_encoder.onnx")
    indices = []
    for start in range(0, len(samples) - FRAME + 1, FRAME):
        features = encoder.run(samples[start:start + FRAME].reshape(1, FRAME)).reshape(-1)
        residual = features.copy()
        packet = []
        for stage in range(46):
            distances = np.sum((codebook[stage] - residual) ** 2, axis=1)
            best = int(np.argmin(distances))
            packet.append(best)
            residual -= codebook[stage, best]
        indices.append(packet)
    return np.array(indices)


def decode(indices, stages, codebook, shape):
    decoder = Streaming(MODEL / "lyragan.onnx")
    output = []
    for packet in indices:
        features = np.zeros(64, dtype=np.float32)
        for stage in range(stages):
            features += codebook[stage, packet[stage]]
        output.append(decoder.run(features.reshape(shape)).reshape(-1))
    return np.concatenate(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codebook = load_codebook()
    decoder_manifest = json.loads((MODEL / "lyragan.json").read_text())
    shape = decoder_manifest["input"].get("shape", [1, 1, 64])
    stage_counts = [4, 6, 8, 10, 12, 14, 16, 30, 46]
    results = {}
    for wav in sorted((LYRA / "testdata").glob("*.wav")):
        reference = read_wav(wav)
        indices = encode(reference, codebook)
        for stages in stage_counts:
            decoded = decode(indices, stages, codebook, shape)
            write_wav(OUT / f"{wav.stem}_{stages * 200}bps.wav", decoded)
            size = min(len(reference), len(decoded))
            score = pesq(16000, reference[:size], decoded[:size], "wb")
            results.setdefault(stages, []).append(score)
            print(f"{wav.stem} stages {stages:2d} ({stages * 200:5d} bit/s) PESQ-WB {score:.2f}", flush=True)
    print()
    for stages in stage_counts:
        print(f"{stages * 200:5d} bit/s  mean PESQ-WB {np.mean(results[stages]):.2f}")


main()
