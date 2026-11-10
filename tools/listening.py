"""Write listening comparisons to build/listening.

`listening.py` writes index.html, vocoscodec against Lyra. `listening.py --dstar`
writes dstar.html, vocoscodec 2.25k against AMBE (D-STAR) at 2.4 kbit/s, from
mbevocoder and with every version level-matched (see dstar_page).

For a few test-set utterances, writes 16 kHz WAVs of:
- the original;
- vocoscodec at several rates (lookahead 3, one packet per call);
- Lyra at its three rates;
- both at 3.0k / 3.2k under 5% burst and 10% radio-frame losses. The
  patterns are those of `evaluate.py quality`, seed 0.

Also writes index.html, with a player for each and its PESQ-WB. The audio is
embedded in the page, so it is one self-contained file (about 10 MB). Needs the
Release build, the corpus (tools/prepare_corpus.py) and the Lyra runtime
(evaluate.py LYRA_DLL).

Speech: LibriSpeech (CC BY 4.0; Panayotov et al., ICASSP 2015) and CMU ARCTIC
(Copyright Carnegie Mellon University, Language Technologies Institute).
"""
import base64
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from evaluate import common  # noqa: E402
from prepare_corpus import active_level_db  # noqa: E402

OUT = ev.ROOT / "build" / "listening"
CLIPS = ["ls_8463-287645-0000", "ls_5105-28233-0000", "ls_1998-15444-0001",
         "arctic_ksp_b0002"]
VOCOS_RATES = [(2, "1.5k"), (3, "2.25k"), (4, "3.0k"), (8, "6.0k"), (12, "9.0k")]
LYRA_RATES = [(3200, "3.2k"), (6000, "6.0k"), (9200, "9.2k")]
LOSSES = [("burst", 0.05, "5% burst losses"), ("radio", 0.10, "10% radio-frame losses")]


def vocos(item, codebooks, loss=None, rate=0.0):
    packets = ev.read_packets(Path(item["packets"]))
    source = Path(item["packets"])
    lost = np.zeros(len(packets), bool)
    if loss:
        lost = ev.loss_mask(len(packets), loss, rate, ev.loss_seed(item, {"seed": 0}))
        source = OUT / "tmp.packets"
        ev.write_packets(source, [b"" if l else p for p, l in zip(packets, lost)])
    wav = OUT / "tmp.wav"
    subprocess.run([str(ev.TOOLS / "vocoscodec_decode.exe"), str(common.MODEL_DIR), str(source),
                    str(wav), "--codebooks", str(codebooks)], check=True)
    y24, _ = common.read_wav(wav)
    return common.down_24k_to_16k(y24), lost.mean()


def lyra(item, bitrate, loss=None, rate=0.0):
    x, _ = common.read_wav(item["reference"])
    lost = np.zeros((len(x) + 319) // 320, bool)
    if loss:
        lost = ev.loss_mask(len(lost), loss, rate, ev.loss_seed(item, {"seed": 0}), frame_ms=20.0)
    return ev.lyra_decode(x, bitrate, lost), lost.mean()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    index = {e["file"][:-4]: e for e in json.loads((common.CORPUS / "test" / "INDEX.json").read_text())}
    items = {i["name"]: i for i in ev.prepare("test")}
    sections = []
    for name in CLIPS:
        item, meta = items[name], index[name]
        reference, _ = common.read_wav(item["reference"])
        rows = [("Original", "original", reference, 0.0)]
        for n, label in VOCOS_RATES:
            rows.append((f"vocoscodec {label}", f"vocoscodec_{n}cb", *vocos(item, n)))
        for bitrate, label in LYRA_RATES:
            rows.append((f"Lyra {label}", f"lyra_{bitrate}", *lyra(item, bitrate)))
        for loss, rate, label in LOSSES:
            tag = f"{loss}{round(100 * rate)}"
            rows.append((f"vocoscodec 3.0k, {label}", f"vocoscodec_4cb_{tag}",
                         *vocos(item, 4, loss, rate)))
            rows.append((f"Lyra 3.2k, {label}", f"lyra_3200_{tag}", *lyra(item, 3200, loss, rate)))
        entries = []
        for label, slug, audio, lost in rows:
            if "losses" in label:
                label += f" ({100 * lost:.0f}% of this clip lost)"
            filename = f"{name}__{slug}.wav"
            audio = audio[:len(reference)]
            common.write_wav(OUT / filename, audio, 16000)
            score = None if label == "Original" else common.pesq_wb(reference, audio)
            entries.append((label, filename, score))
            print(f"{name}  {label:44} {'' if score is None else f'{score:.2f}'}")
        sections.append((name, meta, entries))
    for tmp in ("tmp.wav", "tmp.packets"):
        (OUT / tmp).unlink(missing_ok=True)
    write_html(sections)
    print(f"-> {OUT / 'index.html'}")


def embed(file: str) -> str:
    """The WAV as a data URI, so the page plays wherever it is opened."""
    return "data:audio/wav;base64," + base64.b64encode((OUT / file).read_bytes()).decode()


def write_html(sections):
    parts = []
    for name, meta, entries in sections:
        rows = "\n".join(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td class=num>{'' if score is None else f'{score:.2f}'}</td>"
            f"<td><audio controls preload=none src=\"{embed(file)}\"></audio></td></tr>"
            for label, file, score in entries)
        parts.append(f"<h2>{html.escape(meta['origin'])} ({meta['sex']}, {meta['seconds']} s)</h2>\n"
                     f"<table><tr><th>Version</th><th>PESQ-WB</th><th></th></tr>\n{rows}</table>")
    page = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><title>vocoscodec and Lyra</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em auto; max-width: 60em; padding: 0 1em; }}
table {{ border-collapse: collapse; margin-bottom: 2em; }}
td, th {{ padding: 0.2em 0.8em; text-align: left; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
audio {{ height: 2em; }}
</style></head><body>
<h1>vocoscodec and Lyra</h1>
<p>Test-set utterances, 16 kHz. vocoscodec: lookahead 3 frames, one packet per decode
call, comfort noise on; encoded from the 24 kHz upsampled input and decoded back to 16 kHz.
Lyra: the Lyra fork's lyracodec (OpenCV backend). Losses use the same patterns as
validation/README.md; on short clips the realised loss differs from the nominal
rate, and is shown. Headphones recommended.</p>
<p>Speech: LibriSpeech (CC BY 4.0) and CMU ARCTIC (Copyright Carnegie Mellon University).</p>
{chr(10).join(parts)}
</body></html>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8", newline="\n")


# Chosen from the per-file scores of evaluate.py narrowband, by the gap in PESQ
# between vocoscodec 2.25k and D-STAR: near zero, then each codec's largest.
DSTAR_CLIPS = [("ls_5105-28233-0001", "Typical, male"),
               ("ls_4446-2271-0001", "Typical, female"),
               ("ls_4198-12259-0000", "D-STAR's largest win"),
               ("ls_1688-142285-0000", "D-STAR's largest win from another speaker"),
               ("ls_3331-159605-0001", "vocoscodec's largest win"),
               ("arctic_clb_b0001", "vocoscodec's largest win from another speaker")]


def up2(x):
    return resample_poly(x, 2, 1).astype(np.float32)


def dstar_page():
    """Six clips, each version at 16 kHz and at the original's active speech level.

    vocoscodec also appears limited to 4 kHz, so that its quality can be judged
    apart from its bandwidth. Scores are those of `evaluate.py narrowband`."""
    results = json.loads((ev.OUT / "narrowband.json").read_text())
    clean = [r for r in results if "loss" not in r]
    dstar = next(r for r in clean if r.get("mbe") == "dstar")["files"]
    voc = next(r for r in clean if r.get("codebooks") == 3)["files"]
    scores = {"dstar": {f["file"]: f for f in dstar}, "voc": {f["file"]: f for f in voc}}
    items = {i["name"]: i for i in ev.prepare("test")}
    index = {e["file"][:-4]: e
             for e in json.loads((common.CORPUS / "test" / "INDEX.json").read_text())}
    OUT.mkdir(parents=True, exist_ok=True)
    (ev.OUT / "work").mkdir(parents=True, exist_ok=True)
    sections = []
    for name, why in DSTAR_CLIPS:
        item = items[name]
        reference, _ = common.read_wav(item["reference"])
        y8 = ev.mbe_decode(item, "dstar", None, str(ev.OUT / "work"), name)
        y8 = y8[ev.MBE_DELAY["dstar"]:]
        voc16, _ = vocos(item, 3)
        target = active_level_db(reference)
        versions = [("Original", "original", reference),
                    ("Original, limited to 4 kHz", "original_4k", up2(ev.down2(reference))),
                    ("AMBE (D-STAR), 2400 bit/s", "dstar", up2(y8)),
                    ("vocoscodec 2.25k, limited to 4 kHz", "vocoscodec_4k", up2(ev.down2(voc16))),
                    ("vocoscodec 2.25k", "vocoscodec", voc16)]
        rows = []
        for label, slug, audio in versions:
            audio = np.pad(audio, (0, max(0, len(reference) - len(audio))))[:len(reference)]
            audio = audio * 10 ** ((target - active_level_db(audio)) / 20)
            filename = f"dstar_{name}__{slug}.wav"
            common.write_wav(OUT / filename, audio, 16000)
            rows.append(f"<tr><td>{html.escape(label)}</td><td><audio controls preload=none "
                        f"src=\"{embed(filename)}\"></audio></td></tr>")
            (OUT / filename).unlink()
        d, v = scores["dstar"][name], scores["voc"][name]
        sections.append(
            f"<h2>{html.escape(why)}</h2>\n<p>{html.escape(index[name]['origin'])}, "
            f"{index[name]['sex']}, {index[name]['seconds']} s. PESQ: D-STAR {d['pesq_nb']:.2f}, "
            f"vocoscodec {v['pesq_nb']:.2f}. ESTOI: D-STAR {d['estoi']:.3f}, vocoscodec "
            f"{v['estoi']:.3f}. PESQ-WB: D-STAR {d['pesq']:.2f}, vocoscodec {v['pesq']:.2f}.</p>\n"
            f"<table>{''.join(rows)}</table>")
        print(name, why)
    page = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><title>D-STAR and vocoscodec</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em auto; max-width: 52em; padding: 0 1em; }}
table {{ border-collapse: collapse; margin-bottom: 1.5em; }}
td {{ padding: 0.2em 0.8em; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
audio {{ height: 2em; }}
</style></head><body>
<h1>AMBE (D-STAR) and vocoscodec 2.25k</h1>
<p>Six test-set clips, chosen from the per-file PESQ gap, not by ear: two where the codecs
tie, and each codec's two largest wins. Across all 32 test files the mean scores are PESQ 2.95
D-STAR and 2.96 vocoscodec, and ESTOI 0.827 and 0.803.</p>
<p>D-STAR is mbevocoder's AMBE encoder and decoder at 2400 bit/s of voice (3600 on air, with
FEC), fed 8 kHz speech. vocoscodec is 2250 bit/s, lookahead 3, one packet per call, fed
wideband speech. There are no channel errors, and the speech is clean. Every version here is
at the original's active speech level, so that loudness doesn't decide the comparison.
The "limited to 4 kHz" versions let you compare quality apart from bandwidth.
Headphones recommended.</p>
<p>Speech: LibriSpeech (CC BY 4.0) and CMU ARCTIC (Copyright Carnegie Mellon University).</p>
{chr(10).join(sections)}
</body></html>
"""
    (OUT / "dstar.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"-> {OUT / 'dstar.html'}")


if __name__ == "__main__":
    if "--dstar" in sys.argv[1:]:
        sys.argv = sys.argv[:1]
        dstar_page()
    else:
        main()
