"""Measure vocoscodec with PESQ-WB (and STOI) through its C++ tools.

Subcommands (results as JSON in build/eval, tables on stdout):
  concealment   dev set: concealment timing against random and burst losses
  bandwidth     dev set: which Vocos bandwidth embedding suits each codebook count
  lookahead     dev set: decoder lookahead and packets per call
  quality       test set: rates, losses and Lyra at matching rates (validation/README.md)
  narrowband    test set: against mbevocoder (AMBE+2, AMBE, IMBE) and Lyra, at 8 kHz
  osr           the same on mbevocoder's 8 kHz Open Speech Repository recordings

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
from scipy.signal import resample_poly

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
        # This OpenCV is a CUDA build that needs cuDNN, even though Lyra runs on the CPU.
        for found in sorted(Path(r"C:\Program Files\NVIDIA\CUDNN").glob("*/bin/*/x64/cudnn64_9.dll")):
            os.environ["PATH"] = f"{found.parent}{os.pathsep}{os.environ['PATH']}"
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


# ---------------------------------------------------------------- mbevocoder, for comparison

# github.com/srcejon/mbevocoder, built with mbe-cli; it is run, not linked.
MBE_CLI = Path(os.environ.get("MBE_CLI", ROOT / "build" / "mbevocoder" / "build-release" /
                              "tools" / "mbe-cli" / "mbe-cli.exe"))
# Voice bit/s and on-air bit/s with FEC.
MBE_RATES = {"dstar": (2400, 3600), "ambe2450": (2450, 3600), "imbe4400": (4400, 7200)}
# Encoder input to decoder output in samples at 8 kHz, measured: mean ESTOI peaks
# here over 11 test-set files (D-STAR 50 later, as mbe_encoderDelaySamples says).
MBE_DELAY = {"dstar": 570, "ambe2450": 520, "imbe4400": 520}
MBE_FRAME = 160


def mbe_decode(item, rate: str, lost: np.ndarray | None, work: str, tag: str) -> np.ndarray:
    """8 kHz input -> encoder -> frames (cached) -> decoder; `lost` frames concealed."""
    frames = Path(item["packets"]).with_suffix(f".{rate}.dvsi")
    if not frames.exists():
        wav8 = Path(item["input8"]) if "input8" in item else Path(work) / f"{tag}_in.wav"
        if "input8" not in item:
            x16, _ = common.read_wav(item["reference"])
            common.write_wav(wav8, down2(x16), 8000)
        temporary = Path(work) / f"{tag}.dvsi"
        subprocess.run([str(MBE_CLI), "encode", "--rate", rate, str(wav8), str(temporary)],
                       check=True, capture_output=True)
        os.replace(temporary, frames)
        if "input8" not in item:
            wav8.unlink()
    out = Path(work) / f"{tag}_out.wav"
    args = [str(MBE_CLI), "decode", "--rate", rate]
    lost_file = None
    if lost is not None and lost.any():
        lost_file = Path(work) / f"{tag}_lost.txt"
        lost_file.write_text(" ".join(str(i) for i in np.nonzero(lost)[0]))
        args += ["--lost", str(lost_file)]
    subprocess.run(args + [str(frames), str(out)], check=True, capture_output=True)
    y8, rate_hz = common.read_wav(out)
    assert rate_hz == 8000
    out.unlink()
    if lost_file:
        lost_file.unlink()
    return y8


def down2(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, 1, 2).astype(np.float32)


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


def score_narrowband(reference: np.ndarray, condition, result: dict, decoded16=None,
                     decoded8=None, delay8: int = 0) -> dict:
    """Narrowband PESQ and ESTOI at 8 kHz, as mbevocoder scores, and PESQ-WB at 16 kHz.

    One of decoded16 and decoded8 is given; the other comes from resampling it.
    `delay8` is the decoded signal's delay at 8 kHz."""
    from pesq import pesq
    from pystoi import stoi
    if decoded8 is None:
        decoded8 = down2(decoded16)
    if decoded16 is None:
        decoded16 = resample_poly(decoded8, 2, 1).astype(np.float32)
    reference8 = down2(reference)
    aligned = decoded8[delay8:]
    n = min(len(reference8), len(aligned))
    result["pesq_nb"] = float(pesq(8000, reference8[:n], aligned[:n], "nb"))
    result["estoi"] = float(stoi(reference8[:n], aligned[:n], 8000, extended=True))
    result["pesq"] = common.pesq_wb(reference, decoded16, 2 * delay8)
    return result


def score(reference: np.ndarray, decoded16: np.ndarray, condition, result: dict) -> dict:
    if condition.get("narrowband"):
        return score_narrowband(reference, condition, result, decoded16=decoded16)
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
    if condition.get("codec") == "mbe":
        frames = (len(reference) // 2 + MBE_FRAME - 1) // MBE_FRAME
        lost = loss_mask(frames, condition.get("loss", "random"), condition.get("rate", 0.0),
                         loss_seed(item, condition), frame_ms=20.0)
        tag = f"{item['name']}_{threading.get_ident()}"
        y8 = mbe_decode(item, condition["mbe"], lost, work, tag)
        return score_narrowband(reference, condition,
                                {"file": item["name"], "lost": float(lost.mean())},
                                decoded8=y8, delay8=MBE_DELAY[condition["mbe"]])
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
        if c.get("narrowband"):
            out[-1]["pesq_nb"] = float(np.mean([r["pesq_nb"] for r in scores]))
            out[-1]["estoi"] = float(np.mean([r["estoi"] for r in scores]))
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


def narrowband(args):
    """mbevocoder's three codecs, vocoscodec and Lyra, on the test set, scored at 8 kHz.

    Each codec gets the speech as it would in use (MBE 8 kHz, Lyra 16 kHz, vocoscodec
    24 kHz), and all are scored against the same originals."""
    items = prepare("test")
    codebooks = (2, 3, 4, 6, 8, 12, 16)
    lyra_rates = (3200, 6000, 9200)
    clean = [{"codebooks": n, "narrowband": True} for n in codebooks]
    clean += [{"codec": "lyra", "bitrate": b, "narrowband": True} for b in lyra_rates]
    clean += [{"codec": "mbe", "mbe": r, "narrowband": True} for r in MBE_RATES]
    lossy = []
    for loss, rate in LOSS_CONDITIONS:
        for seed in range(LOSS_SEEDS):
            common_keys = {"loss": loss, "rate": rate, "seed": seed, "narrowband": True}
            lossy.append({"codec": "mbe", "mbe": "ambe2450", **common_keys})
            lossy.append({"codebooks": 3, **common_keys})
            lossy.append({"codebooks": 4, **common_keys})
            lossy.append({"codec": "lyra", "bitrate": 3200, **common_keys})
    results = run(items, clean + lossy, workers=args.workers)
    save("narrowband", results)

    print(f"Test set: {len(items)} files, 8 kHz scoring")
    print(f"{'codec':32} {'bit/s':>6}  PESQ   ESTOI  PESQ-WB")
    rows = []
    for r in results[:len(clean)]:
        if r.get("codec") == "mbe":
            rows.append((MBE_RATES[r["mbe"]][0], f"mbevocoder {r['mbe']}", r))
        elif r.get("codec") == "lyra":
            rows.append((r["bitrate"], "Lyra", r))
        else:
            rows.append((750 * r["codebooks"], f"vocoscodec {r['codebooks']} codebooks", r))
    for bits, label, r in sorted(rows, key=lambda row: row[0]):
        print(f"{label:32} {bits:6d}  {r['pesq_nb']:.2f}   {r['estoi']:.3f}  {r['pesq']:.2f}")

    def mean(keep, key):
        rows = [r for r in results[len(clean):] if keep(r)]
        if key == "lost":
            return 100 * np.mean([f["lost"] for r in rows for f in r["files"]])
        return np.mean([r[key] for r in rows])

    chains = [("AMBE+2 2450", lambda r: r.get("mbe") == "ambe2450"),
              ("vocoscodec 2.25k", lambda r: r.get("codebooks") == 3),
              ("vocoscodec 3.0k", lambda r: r.get("codebooks") == 4),
              ("Lyra 3.2k", lambda r: r.get("bitrate") == 3200)]
    print(f"\nlosses ({LOSS_SEEDS} patterns per file), narrowband PESQ, ESTOI, realised loss")
    for loss, rate in LOSS_CONDITIONS:
        print(f"{loss:6} {100 * rate:3.0f}%")
        for label, keep in chains:
            match = lambda r: keep(r) and r["loss"] == loss and r["rate"] == rate  # noqa: E731,B023
            print(f"    {label:18} {mean(match, 'pesq_nb'):.2f}  {mean(match, 'estoi'):.3f}  "
                  f"({mean(match, 'lost'):.1f}%)")


def osr(args):
    """Every codec on mbevocoder's Open Speech Repository recordings, scored at 8 kHz.

    These are 8 kHz recordings (testdata/corpus of mbevocoder, 15 s each, -26 dBov), so
    every codec gets the same band-limited speech, resampled to its own rate. There is
    nothing above 4 kHz to score, so no PESQ-WB. Speech samples from the Open Speech
    Repository."""
    from pesq import pesq
    from pystoi import stoi
    from scipy.stats import wilcoxon

    directory = OUT / "osr"
    directory.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.corpus).glob("OSR_*.wav"))
    if not files:
        raise SystemExit(f"no OSR_*.wav in {args.corpus}")

    def prepared(path):
        packets = directory / f"{path.stem}.packets"
        if not packets.exists():
            x8, rate = common.read_wav(path)
            assert rate == 8000
            wav24 = directory / f"{path.stem}_24k.wav"
            common.write_wav(wav24, resample_poly(x8, 3, 1).astype(np.float32), 24000)
            subprocess.run([str(TOOLS / "vocoscodec_encode.exe"), str(common.MODEL_DIR),
                            str(wav24), str(packets), "16"], check=True)
        return {"name": path.stem, "input8": str(path), "packets": str(packets)}

    def decoded(item, codec, value):
        """Output at 8 kHz, aligned with the input."""
        tag = f"{item['name']}_{threading.get_ident()}"
        x8, _ = common.read_wav(item["input8"])
        if codec == "mbe":
            return mbe_decode(item, value, None, str(work), tag)[MBE_DELAY[value]:]
        if codec == "lyra":
            return down2(lyra_decode(resample_poly(x8, 2, 1).astype(np.float32), value, None))
        out = work / f"{tag}.wav"
        subprocess.run([str(TOOLS / "vocoscodec_decode.exe"), str(common.MODEL_DIR),
                        item["packets"], str(out), "--codebooks", str(value)], check=True)
        y24, _ = common.read_wav(out)
        out.unlink()
        return down2(common.down_24k_to_16k(y24))

    def measure(task):
        item, codec, value = task
        x8, _ = common.read_wav(item["input8"])
        y8 = decoded(item, codec, value)
        n = min(len(x8), len(y8))
        return {"pesq_nb": float(pesq(8000, x8[:n], y8[:n], "nb")),
                "estoi": float(stoi(x8[:n], y8[:n], 8000, extended=True))}

    codecs = [("vocoscodec", n) for n in (2, 3, 4, 6, 8, 12, 16)] + \
             [("lyra", b) for b in (3200, 6000, 9200)] + [("mbe", r) for r in MBE_RATES]
    with ThreadPoolExecutor(args.workers) as pool:
        items = list(pool.map(prepared, files))
        tasks = [(item, *codec) for codec in codecs for item in items]
        scores = list(pool.map(measure, tasks))
    table = {}
    for i, codec in enumerate(codecs):
        table[codec] = scores[i * len(items):(i + 1) * len(items)]
    (OUT / "osr.json").write_text(json.dumps(
        [{"codec": c, "value": v, "files": [{"file": it["name"], **r}
                                            for it, r in zip(items, table[(c, v)])]}
         for c, v in codecs], indent=1))

    def rate_of(codec, value):
        return MBE_RATES[value][0] if codec == "mbe" else value if codec == "lyra" else 750 * value

    print(f"{len(items)} Open Speech Repository files, 8 kHz scoring")
    print(f"{'codec':24} {'bit/s':>6}  PESQ   ESTOI")
    for codec, value in sorted(codecs, key=lambda c: rate_of(*c)):
        rows = table[(codec, value)]
        print(f"{codec + ' ' + str(value):24} {rate_of(codec, value):6d}  "
              f"{np.mean([r['pesq_nb'] for r in rows]):.2f}   "
              f"{np.mean([r['estoi'] for r in rows]):.3f}")
    print("\npaired differences: mean +- standard error, files won of "
          f"{len(items)}, Wilcoxon p")
    pairs = [(("vocoscodec", 3), ("mbe", "ambe2450")), (("vocoscodec", 3), ("mbe", "dstar")),
             (("vocoscodec", 4), ("mbe", "ambe2450")), (("vocoscodec", 4), ("mbe", "dstar")),
             (("vocoscodec", 4), ("mbe", "imbe4400")), (("vocoscodec", 6), ("mbe", "imbe4400")),
             (("vocoscodec", 3), ("lyra", 3200)), (("vocoscodec", 4), ("lyra", 3200)),
             (("vocoscodec", 8), ("lyra", 6000)), (("vocoscodec", 12), ("lyra", 9200)),
             (("vocoscodec", 16), ("lyra", 9200))]
    for a, b in pairs:
        for key in ("pesq_nb", "estoi"):
            d = np.array([x[key] - y[key] for x, y in zip(table[a], table[b])])
            print(f"{a[0]} {a[1]} - {b[0]} {b[1]}: {key:7} {d.mean():+.3f} +- "
                  f"{d.std(ddof=1) / np.sqrt(len(d)):.3f}, {int((d > 0).sum())}, "
                  f"p={wilcoxon(d).pvalue:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("concealment")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--codebooks", type=int, nargs="+", default=[4])
    sub.add_parser("bandwidth")
    sub.add_parser("lookahead")
    sub.add_parser("quality")
    p = sub.add_parser("narrowband")
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("osr")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--corpus", default=r"C:\Users\jon\source\repos\ambe\testdata\corpus")
    args = parser.parse_args()
    {"concealment": concealment, "bandwidth": bandwidth, "lookahead": lookahead,
     "quality": quality, "narrowband": narrowband, "osr": osr}[args.command](args)


if __name__ == "__main__":
    main()
