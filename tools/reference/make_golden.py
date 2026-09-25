"""Write the golden vectors the C++ tests compare against (testdata/golden.bin).

From the first 120 frames (1.6 s) of a LibriSpeech test-clean utterance
(CC BY 4.0), resampled to 24 kHz, the frame-by-frame reference gives:

  pcm            [120 * 320]  the input, int16 values / 32768
  latent         [120, 128]   streaming encoder output before quantisation
  indices        [120, 16]    quantiser output, all 16 codebooks
  packets_n{n}   [120, bytes] packed indices, n = 1..16 (byte values)
  ids_n{n}       [120]        bandwidth ids used by the decoder vectors
  decoded_n{n}_l{L}  [120 * 320 + delay]  streaming decode plus flush, for
                 (n, L) in DECODES, from the indices truncated to n codebooks,
                 one packet per call
  decoded_n4_l3_g9   the same with packets given nine per call
  decoded_n4_l3_loss the same, one per call, with frames in LOST missing
                     (comfort noise on)

The file uses the model file format (tools/modelfile.py); integers are
stored as float32, which holds them exactly.
"""
import numpy as np
import torch

import common
import sys
from pathlib import Path

import vocoscodec_ref as ref
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelfile  # noqa: E402

FRAMES = 120
CLIP = "ls_1089-134686-0000.wav"
DECODES = ((1, 3), (2, 3), (4, 3), (8, 3), (16, 3), (4, 0), (4, 27))
OUT = common.ROOT / "testdata" / "golden.bin"
# An isolated loss, a two-frame burst and a 40-frame (533 ms) outage.
LOST = [20, 40, 41] + list(range(60, 100))


def main():
    t = ref.load(common.MODEL_DIR)
    x16, _ = common.read_wav(common.CORPUS / "dev" / CLIP)
    x = common.up_16k_to_24k(x16)[:FRAMES * ref.FRAME]
    pcm = common.to_pcm16(x).astype(np.float32) / 32768

    enc = ref.StreamingEncoder(t, 16)
    latents, indices = [], []
    with torch.no_grad():
        for i in range(FRAMES):
            latent = enc.embedding(pcm[i * ref.FRAME:(i + 1) * ref.FRAME])
            latents.append(latent.numpy())
            indices.append(enc.quantize(latent))
    indices = np.array(indices)

    tensors = {"pcm": pcm, "latent": np.stack(latents), "indices": indices.astype(np.float32)}
    for n in range(1, 17):
        tensors[f"packets_n{n}"] = np.array(
            [list(ref.pack(row[:n])) for row in indices], dtype=np.float32)
        tensors[f"ids_n{n}"] = np.full(FRAMES, ref.bandwidth_id_for(n), dtype=np.float32)
    for n, lookahead in DECODES:
        dec = ref.StreamingDecoder(t, lookahead)
        with torch.no_grad():
            out = [dec.decode_indices(row[:n].tolist()) for row in indices] + [dec.flush()]
        tensors[f"decoded_n{n}_l{lookahead}"] = np.concatenate(out)
    dec = ref.StreamingDecoder(t, 3)
    with torch.no_grad():
        out = []
        for f in range(0, FRAMES, 9):
            group = indices[f:f + 9, :4].tolist()
            out.append(dec.decode_group([ref.features(t, c) for c in group],
                                        [ref.bandwidth_id_for(4)] * len(group)))
        out.append(dec.flush())
    tensors["decoded_n4_l3_g9"] = np.concatenate(out)
    dec = ref.StreamingDecoder(t, 3)
    with torch.no_grad():
        out = [dec.decode_packets([b"" if f in LOST else ref.pack(indices[f, :4])])
               for f in range(FRAMES)] + [dec.flush()]
    tensors["decoded_n4_l3_loss"] = np.concatenate(out)
    tensors["lost"] = np.array(LOST, dtype=np.float32)
    meta = {"clip": f"LibriSpeech test-clean {CLIP[3:-4]}.flac (CC BY 4.0), 16 -> 24 kHz",
            "frames": FRAMES, "bandwidth_rule": "measured", "half_weights": 1,
            "conceal_repeat": ref.REPEAT_FRAMES, "conceal_fade": ref.FADE_FRAMES,
            "decodes": ";".join(f"{n},{l}" for n, l in DECODES)}
    OUT.parent.mkdir(exist_ok=True)
    modelfile.write(OUT, meta, tensors)
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} kB, {len(tensors)} tensors")


if __name__ == "__main__":
    main()
