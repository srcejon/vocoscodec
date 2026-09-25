"""Vocos decoding with limited lookahead, and encoder/decoder CPU split."""
import time
from pathlib import Path

import numpy as np
import torch

exec(open(Path(__file__).parent / "compare_vocos.py").read().split("results, timing = {}, {}")[0])

HOP_MS = 320 / 24000 * 1000


def decode_limited(features, bandwidth_id, lookahead, history=40):
    backbone, head = vocos_codec.backbone, vocos_codec.head
    frames = features.shape[-1]
    hidden = []
    for t in range(frames):
        start = max(0, t - history)
        end = min(frames, t + lookahead + 1)
        h = backbone(features[..., start:end], bandwidth_id=torch.tensor([bandwidth_id]))
        hidden.append(h[:, t - start:t - start + 1, :])
    return head(torch.cat(hidden, dim=1))


with torch.no_grad():
    scores = {}
    for clip in CLIPS:
        reference = read_wav(clip)
        seconds = len(reference) / 16000
        x = up(torch.from_numpy(reference).reshape(1, -1))
        for bandwidth, n_q, bid in ((1.5, 2, 0), (3.0, 4, 1)):
            encodec.set_target_bandwidth(bandwidth)
            t = time.perf_counter()
            codes = encodec.encode(x.reshape(1, 1, -1))[0][0]
            enc = time.perf_counter() - t
            features = vocos_codec.codes_to_features(codes[0].reshape(n_q, -1))
            t = time.perf_counter()
            vocos_codec.decode(features, bandwidth_id=torch.tensor([bid]))
            dec = time.perf_counter() - t
            print(f"{clip.stem} {bandwidth}k: EnCodec encoder {100 * enc / seconds:.0f}%, "
                  f"Vocos decoder {100 * dec / seconds:.1f}% of one core")
            for lookahead in (0, 3, 6, 12, 27):
                y = decode_limited(features, bid, lookahead)
                decoded = down(y.reshape(1, -1)).reshape(-1).numpy()
                scores.setdefault((bandwidth, lookahead), []).append(score(reference, decoded))
    print()
    for (bandwidth, lookahead), s in sorted(scores.items()):
        print(f"{bandwidth}k, lookahead {lookahead:2d} frames ({lookahead * HOP_MS:5.0f} ms): PESQ-WB {np.mean(s):.2f}")
