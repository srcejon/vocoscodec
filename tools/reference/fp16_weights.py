"""Would float16 weights (in memory; the files stay float32) be accurate enough?

Rounds the named weight groups to float16 and measures, on dev files:
encoder index agreement with float32 (per codebook), and decoder PESQ-WB
change (whole-signal decode, 4 codebooks). Usage: fp16_weights.py [files]
"""
import sys

import numpy as np
import torch
from encodec import EncodecModel
from encodec.modules.conv import SConv1d

import common
import vocoscodec_ref as ref

FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
GROUPS = {
    "encoder frame-rate layers (lstm, down3, conv_out)":
        lambda k: k.startswith(("lstm", "down3", "conv_out")),
    "all encoder convolutions and lstm":
        lambda k: k.endswith((".w", "w_ih", "w_hh")) and not k.startswith(("embed", "block", "head")),
    "decoder linear layers (embed, pw1, pw2, head)":
        lambda k: k in ("embed.w", "head.w") or k.endswith(("pw1.w", "pw2.w")),
}


def rounded(t, select):
    return {k: (v.half().float() if select(k) else v) for k, v in t.items()}


def encode(t, x):
    enc = ref.StreamingEncoder(t, 16)
    return np.array([enc.quantize(enc.embedding(x[i * 320:(i + 1) * 320]))
                     for i in range(len(x) // 320)])


def main():
    torch.set_num_threads(8)
    t = ref.load(common.MODEL_DIR)
    clips = common.corpus("dev")[:FILES]
    for label, select in GROUPS.items():
        t16 = rounded(t, select)
        agree, pesq32, pesq16 = [], [], []
        for clip in clips:
            x16, _ = common.read_wav(clip)
            x = common.up_16k_to_24k(x16)
            with torch.no_grad():
                c32 = encode(t, x)
                if "decoder" in label:
                    ids = [1] * len(c32)
                    y32 = ref.decode_batch(t, c32[:, :4].tolist(), ids)
                    y16 = ref.decode_batch(t16, c32[:, :4].tolist(), ids)
                    pesq32.append(common.pesq_wb(x16, common.down_24k_to_16k(y32)))
                    pesq16.append(common.pesq_wb(x16, common.down_24k_to_16k(y16)))
                else:
                    c16 = encode(t16, x)
                    agree.append((c16 == c32).mean(axis=0))
        if agree:
            a = 100 * np.mean(agree, axis=0)
            print(f"{label}: indices equal to float32, codebooks 1/2/4/8/16: "
                  f"{a[0]:.1f} {a[1]:.1f} {a[3]:.1f} {a[7]:.1f} {a[15]:.1f} %, all {a.mean():.1f} %")
        else:
            print(f"{label}: PESQ-WB float32 {np.mean(pesq32):.3f}, float16 {np.mean(pesq16):.3f}")
    print(f"({len(clips)} dev files)")


if __name__ == "__main__":
    main()
