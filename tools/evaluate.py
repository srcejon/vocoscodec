"""Measure vocoscodec with PESQ-WB (and STOI) through its C++ tools.

Subcommands (results as JSON in build/eval, tables on stdout):
  concealment   dev set: concealment timing against random and burst losses
  bandwidth     dev set: which Vocos bandwidth embedding suits each codebook count
  lookahead     dev set: decoder lookahead and packets per call
  quality       test set: rates, losses and Lyra at matching rates (validation/README.md)

Build first (build.bat Release). Inputs come from tools/prepare_corpus.py.
Each 16 kHz file is resampled to 24 kHz (scipy polyphase), encoded once at
16 codebooks (fewer codebooks are the same packets truncated), decoded by
vocoscodec_decode (which trims the codec delay), resampled back to 16 kHz
and scored against the original.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import zlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
import common  # noqa: E402

ROOT = common.ROOT
TOOLS = ROOT / "build" / "Release" / "vocoscodec"
OUT = ROOT / "build" / "eval"
FRAME = 320


# ---------------------------------------------------------------- packets and losses

def read_packets(path: Path) -> list[bytes]:
    data = path.read_bytes()
    packets, pos = [], 0
    while pos < len(data):
        size = data[pos]
        packets.append(data[pos + 1:pos + 1 + size])
        pos += 1 + size
    return packets


def write_packets(path: Path, packets: list[bytes]):
    path.write_bytes(b"".join(bytes([len(p)]) + p for p in packets))


def loss_mask(frames: int, kind: str, rate: float, seed: int,
              frame_ms: float = 40 / 3) -> np.ndarray:
    """Lost packets of `frame_ms` each: "random" (independent), "burst"
    (Gilbert model, bursts of mean length 53 ms, 4 vocoscodec packets) or
    "radio" (whole 240 ms groups, an Angel61 HF radio frame: 18 vocoscodec
    packets). Durations are in time so that codecs with other frame lengths
    (Lyra: 20 ms) lose comparable audio."""
    rng = np.random.default_rng(seed)
    if rate <= 0:
        return np.zeros(frames, bool)
    if kind == "random":
        return rng.random(frames) < rate
    if kind == "burst":
        q = min(1.0, frame_ms / (4 * 40 / 3))
        p = rate / (1 - rate) * q
        lost, state = np.zeros(frames, bool), rng.random() < rate
        for i in range(frames):
            lost[i] = state
            state = (rng.random() >= q) if state else (rng.random() < p)
        return lost
    if kind == "radio":
        per_group = round(240 / frame_ms)
        groups = rng.random((frames + per_group - 1) // per_group) < rate
        return np.repeat(groups, per_group)[:frames]
    raise ValueError(kind)


def loss_seed(item, condition) -> int:
    """Each file and seed gets its own pattern (the same for every codec)."""
    return zlib.crc32(item["name"].encode()) * 1000 + condition.get("seed", 0)


# ---------------------------------------------------------------- Lyra, for comparison

LYRA_DLL = Path(r"C:\Users\jon\source\repos\lyra\build-final-install\bin\lyracodec.dll")
LYRA_MODELS = Path(r"C:\Users\jon\source\repos\lyra\lyra_codec\model")
_lyra = None


def lyra():
    """lyracodec (the Lyra fork's C ABI 2) through ctypes; OpenCV must be on PATH."""
    global _lyra
    if _lyra is None:
        import ctypes
        lib = ctypes.CDLL(str(LYRA_DLL), winmode=0)  # standard search: OpenCV via PATH
        lib.lyracodec_encoder_create.restype = ctypes.c_void_p
        lib.lyracodec_decoder_create.restype = ctypes.c_void_p
        _lyra = lib
    return _lyra


def lyra_decode(x16: np.ndarray, bitrate: int, lost: np.ndarray | None) -> np.ndarray:
    import ctypes
    lib = lyra()
    encoder = ctypes.c_void_p(lib.lyracodec_encoder_create(str(LYRA_MODELS).encode(), bitrate))
    decoder = ctypes.c_void_p(lib.lyracodec_decoder_create(str(LYRA_MODELS).encode()))
    pcm = common.to_pcm16(x16)
    pcm = np.concatenate([pcm, np.zeros(-len(pcm) % 320, "<i2")])
    packet, audio = (ctypes.c_uint8 * 32)(), (ctypes.c_int16 * 320)()
    out = []
    for f in range(len(pcm) // 320):
        frame = np.ascontiguousarray(pcm[f * 320:(f + 1) * 320])
        size = lib.lyracodec_encode(encoder, frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                    packet, 32)
        if lost is not None and lost[f]:
            n = lib.lyracodec_decode_lost(decoder, audio, 320)
        else:
            n = lib.lyracodec_decode(decoder, packet, size, audio, 320)
        out.append(np.frombuffer(bytes(audio), dtype="<i2")[:n].astype(np.float32) / 32768)
    lib.lyracodec_encoder_destroy(encoder)
    lib.lyracodec_decoder_destroy(decoder)
    return np.concatenate(out)


# ---------------------------------------------------------------- running the codec

def prepare(set_name: str) -> list[dict]:
    """24 kHz inputs and 16-codebook packets for every file of a set (cached)."""
    directory = OUT / set_name
    directory.mkdir(parents=True, exist_ok=True)
    items = []
    for path in common.corpus(set_name):
        x16, rate = common.read_wav(path)
        assert rate == 16000
        wav24 = directory / f"{path.stem}_24k.wav"
        packets = directory / f"{path.stem}.packets"
        if not packets.exists():
            common.write_wav(wav24, common.up_16k_to_24k(x16), 24000)
            subprocess.run([str(TOOLS / "vocoscodec_encode.exe"), str(common.MODEL_DIR),
                            str(wav24), str(packets), "16"], check=True)
        items.append({"name": path.stem, "reference": str(path), "packets": str(packets)})
    return items


def score(reference: np.ndarray, decoded16: np.ndarray, condition, result: dict) -> dict:
    result["pesq"] = common.pesq_wb(reference, decoded16)
    if condition.get("stoi"):
        from pystoi import stoi
        n = min(len(reference), len(decoded16))
        result["stoi"] = float(stoi(reference[:n], decoded16[:n], 16000))
    return result


def decode_score(task) -> dict:
    """decode_score_once, cached on disk and retried: on a busy machine a
    decoder process can fail to allocate memory."""
    item, condition, work = task
    key = json.dumps([item["name"], item["packets"], condition], sort_keys=True)
    cache = OUT / "cache" / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(6):
        try:
            result = decode_score_once(task)
            break
        except (subprocess.CalledProcessError, MemoryError, OSError):
            if attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result


def decode_score_once(task) -> dict:
    """Decode one file under one condition and score it (in a pool thread)."""
    item, condition, work = task
    reference, _ = common.read_wav(item["reference"])
    if condition.get("codec") == "lyra":
        frames = (len(reference) + 319) // 320
        lost = loss_mask(frames, condition.get("loss", "random"), condition.get("rate", 0.0),
                         loss_seed(item, condition), frame_ms=20.0)
        y16 = lyra_decode(reference, condition["bitrate"], lost)
        return score(reference, y16, condition,
                     {"file": item["name"], "lost": float(lost.mean())})
    packets = read_packets(Path(item["packets"]))
    lost = loss_mask(len(packets), condition.get("loss", "random"), condition.get("rate", 0.0),
                     loss_seed(item, condition))
    source = Path(item["packets"])
    tag = f"{item['name']}_{abs(hash(json.dumps(condition, sort_keys=True))) % 10**12}_{threading.get_ident()}"
    if lost.any():
        source = Path(work) / f"{tag}.packets"
        write_packets(source, [b"" if l else p for p, l in zip(packets, lost)])
    out = Path(work) / f"{tag}.wav"
    args = [str(TOOLS / "vocoscodec_decode.exe"), str(common.MODEL_DIR), str(source), str(out),
            "--codebooks", str(condition["codebooks"])]
    for key in ("lookahead", "group", "repeat", "fade", "bandwidth-id"):
        if key in condition:
            args += [f"--{key}", str(condition[key])]
    if condition.get("silence"):
        args.append("--silence")
    subprocess.run(args, check=True)
    y24, _ = common.read_wav(out)
    result = score(reference, common.down_24k_to_16k(y24), condition,
                   {"file": item["name"], "lost": float(lost.mean())})
    out.unlink()
    if source != Path(item["packets"]):
        source.unlink()
    return result


def run(items, conditions, workers=None) -> list[dict]:
    """Every condition on every file; returns conditions with per-file scores."""
    work = OUT / "work"
    work.mkdir(parents=True, exist_ok=True)
    tasks = [(item, c, str(work)) for c in conditions for item in items]
    # Threads, not processes: the work is in the decoder subprocesses, and worker
    # processes each importing scipy can exhaust the Windows commit limit.
    with ThreadPoolExecutor(workers or min(4, os.cpu_count())) as pool:
        results = []
        for i, result in enumerate(pool.map(decode_score, tasks)):
            results.append(result)
            if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} decoded", file=sys.stderr, flush=True)
    out = []
    for i, c in enumerate(conditions):
        scores = results[i * len(items):(i + 1) * len(items)]
        out.append({**c, "pesq": float(np.mean([r["pesq"] for r in scores])),
                    "files": scores})
        if c.get("stoi"):
            out[-1]["stoi"] = float(np.mean([r["stoi"] for r in scores]))
    return out


def save(name: str, results):
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"-> {path}")


# ---------------------------------------------------------------- experiments

LOSSES = [("random", r) for r in (0.01, 0.05, 0.10, 0.20)] + \
         [("burst", r) for r in (0.01, 0.05, 0.10, 0.20)]


def concealment(args):
    """Concealment timing: repeat R frames, then fade over F, by PESQ under losses."""
    items = prepare("dev")
    timings = [(0, 0), (1, 2), (3, 3), (5, 3), (8, 4), (1000, 0)]
    conditions = []
    for codebooks in args.codebooks:
        for loss, rate in LOSSES:
            for seed in range(args.seeds):
                for repeat, fade in timings:
                    conditions.append({"codebooks": codebooks, "loss": loss, "rate": rate,
                                       "seed": seed, "repeat": repeat, "fade": fade})
    results = run(items, conditions)
    save("concealment", results)
    print(f"PESQ-WB, dev set ({len(items)} files, {args.seeds} loss seeds), "
          "columns: repeat R frames then fade over F")
    header = "  ".join(f"{r:>4}/{f:<3}" for r, f in timings)
    for codebooks in args.codebooks:
        print(f"\n{codebooks} codebooks        {header}")
        for loss, rate in LOSSES:
            row = []
            for repeat, fade in timings:
                s = [r["pesq"] for r in results if r["codebooks"] == codebooks and r["loss"] == loss
                     and r["rate"] == rate and r["repeat"] == repeat and r["fade"] == fade]
                row.append(np.mean(s))
            print(f"{loss:6} {100 * rate:4.0f}%        " + "  ".join(f"{v:8.3f}" for v in row))


def bandwidth(args):
    """PESQ for every codebook count with each of the four bandwidth embeddings."""
    items = prepare("dev")
    counts = list(range(1, 17))
    conditions = [{"codebooks": n, "bandwidth-id": b} for n in counts for b in range(4)]
    results = run(items, conditions)
    save("bandwidth", results)
    print(f"PESQ-WB, dev set ({len(items)} files); embeddings trained for 2/4/8/16 codebooks")
    print("codebooks   id0    id1    id2    id3   best")
    for n in counts:
        row = [next(r["pesq"] for r in results if r["codebooks"] == n and r["bandwidth-id"] == b)
               for b in range(4)]
        print(f"{n:9d}  " + "  ".join(f"{v:5.2f}" for v in row) + f"   id{int(np.argmax(row))}")


def lookahead(args):
    """Decoder lookahead (one packet per call), and packets per call at L = 3."""
    items = prepare("dev")
    conditions = [{"codebooks": n, "lookahead": l} for n in (2, 4, 8) for l in (0, 1, 2, 3, 6, 27)]
    conditions += [{"codebooks": n, "lookahead": 3, "group": g} for n in (2, 4, 8) for g in (9, 18)]
    results = run(items, conditions)
    save("lookahead", results)
    print(f"PESQ-WB, dev set ({len(items)} files)")
    for r in results:
        print(f"{r['codebooks']:2d} codebooks, lookahead {r['lookahead']:2d}, "
              f"{r.get('group', 1):2d} per call: {r['pesq']:.3f}")


def upstream(items, counts) -> dict:
    """PyTorch EnCodec (whole file) + Vocos (whole file, unbounded lookahead)."""
    import torch
    from encodec import EncodecModel
    from vocos import Vocos
    torch.set_num_threads(8)
    encodec = EncodecModel.encodec_model_24khz().eval()
    encodec.set_target_bandwidth(12.0)
    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").eval()
    scores = {n: [] for n in counts}
    with torch.no_grad():
        for item in items:
            x16, _ = common.read_wav(item["reference"])
            x = torch.from_numpy(common.up_16k_to_24k(x16))
            codes = encodec.encode(x[None, None])[0][0][0]
            for n in counts:
                bid = 0 if n <= 3 else 1 if n <= 5 else 2 if n <= 10 else 3
                y = vocos.decode(vocos.codes_to_features(codes[:n]),
                                 bandwidth_id=torch.tensor([bid]))[0].numpy()
                scores[n].append(common.pesq_wb(x16, common.down_24k_to_16k(y)))
    return {n: float(np.mean(v)) for n, v in scores.items()}


def isa_agreement(items) -> list[float]:
    """Indices equal between the SIMD and the portable kernels, per codebook."""
    equal, total = np.zeros(16), 0
    for item in items:
        generic = Path(item["packets"]).with_suffix(".generic.packets")
        wav24 = Path(item["packets"]).with_name(Path(item["packets"]).stem + "_24k.wav")
        if not generic.exists():
            subprocess.run([str(TOOLS / "vocoscodec_encode.exe"), str(common.MODEL_DIR),
                            str(wav24), str(generic), "16", "--generic"], check=True)
        sys.path.insert(0, str(ROOT / "tools" / "reference"))
        import vocoscodec_ref as ref
        a = np.array([ref.unpack(p) for p in read_packets(Path(item["packets"]))])
        b = np.array([ref.unpack(p) for p in read_packets(generic)])
        equal += (a == b).sum(axis=0)
        total += len(a)
    return list(100 * equal / total)


QUALITY_RATES = (1, 2, 3, 4, 6, 8, 12, 16)
LYRA_RATES = (3200, 6000, 9200)
LOSS_SEEDS = 3
LOSS_CONDITIONS = [(k, r) for k in ("random", "burst") for r in (0.01, 0.05, 0.10, 0.20)] + \
                  [("radio", r) for r in (0.05, 0.10, 0.20)]


def quality(args):
    """The held-out test set: everything validation/README.md reports."""
    items = prepare("test")
    conditions = [{"codebooks": n, "stoi": True} for n in QUALITY_RATES]
    conditions += [{"codebooks": n, "group": 9, "stoi": True} for n in (2, 3, 4, 8)]
    conditions += [{"codec": "lyra", "bitrate": b, "stoi": True} for b in LYRA_RATES]
    for loss, rate in LOSS_CONDITIONS:
        for seed in range(LOSS_SEEDS):
            conditions.append({"codebooks": 4, "loss": loss, "rate": rate, "seed": seed})
            conditions.append({"codebooks": 4, "loss": loss, "rate": rate, "seed": seed,
                               "silence": True})
            conditions.append({"codec": "lyra", "bitrate": 3200, "loss": loss, "rate": rate,
                               "seed": seed})
    results = run(items, conditions)
    extra = {"upstream": upstream(items, (2, 4, 8)), "isa_agreement": isa_agreement(items)}
    save("quality", {"conditions": results, **extra})

    def find(**keys):
        return next(r for r in results if all(r.get(k) == v for k, v in keys.items())
                    and all(k in keys for k in ("codec", "group", "loss", "silence") if k in r))

    def lossy(**keys):
        """Mean PESQ and realised loss over the seeds."""
        rows = [r for r in results if all(r.get(k) == v for k, v in keys.items())
                and all(k in keys for k in ("codec", "silence") if k in r)]
        lost = np.mean([f["lost"] for r in rows for f in r["files"]])
        return np.mean([r["pesq"] for r in rows]), 100 * lost

    print(f"Test set: {len(items)} files")
    print("rate                              PESQ-WB  STOI")
    for n in QUALITY_RATES:
        r = find(codebooks=n)
        print(f"vocoscodec {n:2d} codebooks ({750 * n:5d} bit/s)  {r['pesq']:.2f}   {r['stoi']:.3f}")
    for n in (2, 3, 4, 8):
        r = find(codebooks=n, group=9)
        print(f"  {n} codebooks, 9 packets per call       {r['pesq']:.2f}   {r['stoi']:.3f}")
    for b in LYRA_RATES:
        r = find(codec="lyra", bitrate=b)
        print(f"Lyra {b} bit/s                    {r['pesq']:.2f}   {r['stoi']:.3f}")
    for n, v in extra["upstream"].items():
        print(f"PyTorch EnCodec + Vocos, whole file, {n} codebooks: {v:.2f}")
    print(f"\nlosses ({LOSS_SEEDS} patterns per file; realised loss in brackets)")
    print("                3.0k comfort noise   3.0k silence   Lyra 3.2k")
    for loss, rate in LOSS_CONDITIONS:
        a, la = lossy(codebooks=4, loss=loss, rate=rate)
        b, _ = lossy(codebooks=4, loss=loss, rate=rate, silence=True)
        c, lc = lossy(codec="lyra", bitrate=3200, loss=loss, rate=rate)
        print(f"{loss:6} {100 * rate:3.0f}%    {a:.2f} ({la:4.1f}%)       {b:.2f}        "
              f"{c:.2f} ({lc:4.1f}%)")
    a = extra["isa_agreement"]
    print(f"\nindices equal, SIMD vs portable kernels, codebooks 1/2/4/8/16: "
          f"{a[0]:.2f} {a[1]:.2f} {a[3]:.2f} {a[7]:.2f} {a[15]:.2f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("concealment")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--codebooks", type=int, nargs="+", default=[4])
    sub.add_parser("bandwidth")
    sub.add_parser("lookahead")
    sub.add_parser("quality")
    args = parser.parse_args()
    {"concealment": concealment, "bandwidth": bandwidth, "lookahead": lookahead,
     "quality": quality}[args.command](args)


if __name__ == "__main__":
    main()
