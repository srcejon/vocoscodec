"""Build the speech sets used to design and validate vocoscodec.

Nothing is downloaded: the sources are archives already on disk (by default the
ones the AMBE project fetched into ambe/corpus/raw). Two disjoint sets are made:

  dev   decides design choices (decoder lookahead scheme, bandwidth embedding
        rule, concealment parameters).
  test  is held out until the final measurements (validation/README.md).

Each file is 16 kHz mono 16-bit PCM, at most 10 s, with its active speech level
set to -26 dBov. INDEX.json records the origin, speaker and SHA-256 of each file.
Needs ffmpeg on PATH, numpy and scipy.

Sources and licences: LibriSpeech (CC BY 4.0; Panayotov, Chen, Povey,
Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books",
ICASSP 2015) and CMU ARCTIC (Copyright Carnegie Mellon University, Language
Technologies Institute; free for any purpose with the notice kept).
"""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

DEFAULT_RAW = Path(r"C:\Users\jon\source\repos\ambe\corpus\raw")
ROOT = Path(__file__).resolve().parent.parent
MAX_SECONDS = 10.0
TARGET_DBOV = -26.0

# (archive relative to the raw directory, subset, speakers, utterances per speaker)
LIBRISPEECH = {
    "dev": [("librispeech_test_clean/test-clean.tar.gz", "test-clean",
             ["1089", "121", "237", "260", "672", "1221", "1320", "1580"], 2)],
    "test": [("librispeech_test_clean/test-clean.tar.gz", "test-clean",
              ["2830", "3729", "4446", "5105", "5639", "7729", "8463", "8555"], 2),
             ("librispeech_test_other/test-other.tar.gz", "test-other",
              ["1688", "1998", "3331", "4198"], 2)],
}
# (archive, speaker, utterance ids)
ARCTIC = {
    "dev": [("arctic_bdl/cmu_us_bdl_arctic-0.95-release.tar.bz2", "bdl", ["a0001", "a0002"]),
            ("arctic_slt/cmu_us_slt_arctic-0.95-release.tar.bz2", "slt", ["a0001", "a0002"])],
    "test": [(f"arctic_extra/cmu_us_{s}_arctic-0.95-release.tar.bz2", s, ["b0001", "b0002"])
             for s in ("awb", "clb", "jmk", "ksp")],
}
SEX = {"1089": "M", "121": "F", "237": "F", "260": "M", "672": "M", "1221": "F", "1320": "M",
       "1580": "F", "2830": "M", "3729": "F", "4446": "F", "5105": "M", "5639": "M", "7729": "M",
       "8463": "F", "8555": "F", "1688": "M", "1998": "F", "3331": "F", "4198": "M",
       "bdl": "M", "slt": "F", "awb": "M", "clb": "F", "jmk": "M", "ksp": "M"}


def decode_16k(data: bytes) -> np.ndarray:
    """Decode any format ffmpeg reads to 16 kHz mono float32."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", "16000", "-af", "aresample=resampler=soxr",
         "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def active_level_db(x: np.ndarray) -> float:
    """Active speech level: RMS over 20 ms frames within 30 dB of the loudest."""
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)
    energy = (frames ** 2).mean(axis=1) + 1e-12
    active = energy > energy.max() * 10 ** (-30 / 10)
    return 10 * np.log10(energy[active].mean())


def write(out_dir: Path, name: str, x: np.ndarray, origin: str, speaker: str, index: list):
    x = x[: int(MAX_SECONDS * 16000)]
    x = x * 10 ** ((TARGET_DBOV - active_level_db(x)) / 20)
    pcm = np.clip(np.round(x * 32768), -32768, 32767).astype("<i2")
    path = out_dir / f"{name}.wav"
    wavfile.write(path, 16000, pcm)
    index.append({"file": path.name, "origin": origin, "speaker": speaker,
                  "sex": SEX[speaker], "seconds": round(len(pcm) / 16000, 2),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "corpus")
    args = parser.parse_args()

    for set_name in ("dev", "test"):
        out_dir = args.out / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        index = []
        for archive, subset, speakers, count in LIBRISPEECH[set_name]:
            with tarfile.open(args.raw / archive) as tar:
                members = sorted((m for m in tar.getmembers() if m.name.endswith(".flac")),
                                 key=lambda m: m.name)
                for speaker in speakers:
                    chosen = [m for m in members if f"/{subset}/{speaker}/" in m.name][:count]
                    for member in chosen:
                        x = decode_16k(tar.extractfile(member).read())
                        write(out_dir, f"ls_{Path(member.name).stem}", x,
                              f"LibriSpeech {subset} {Path(member.name).name}", speaker, index)
        for archive, speaker, utterances in ARCTIC[set_name]:
            with tarfile.open(args.raw / archive) as tar:
                for utterance in utterances:
                    member = next(m for m in tar.getmembers()
                                  if m.name.endswith(f"/wav/arctic_{utterance}.wav"))
                    x = decode_16k(tar.extractfile(member).read())
                    write(out_dir, f"arctic_{speaker}_{utterance}", x,
                          f"CMU ARCTIC {speaker} arctic_{utterance}", speaker, index)
        (out_dir / "INDEX.json").write_text(json.dumps(index, indent=1))
        seconds = sum(e["seconds"] for e in index)
        print(f"{set_name}: {len(index)} files, {seconds:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
