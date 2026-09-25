"""Decide the streaming decoder's lookahead scheme by PESQ-WB on the dev set.

Schemes:
  full      Vocos as released: every k=7 convolution sees 3 frames ahead (27 in all).
  window L  output t = backbone over frames [t-30, t+L] (zero beyond), as in
            lookahead.py. Exact for a bounded lookahead, but a streaming decoder
            recomputes about 70 ConvNeXt blocks per frame instead of 8.
  budget    each convolution k sees rights[k] frames ahead (sum = lookahead);
            taps beyond are zero ("zero") or repeat the last frame ("repeat").
            Streams at the cost of the unmodified network.

Codes come from the streaming encoder's whole-signal equivalent (upstream
EnCodec with zero start padding). Usage: decoder_lookahead.py [--quick]
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import torch
from encodec import EncodecModel
from encodec.modules.conv import SConv1d

import common
import vocoscodec_ref as ref

RATES = ((2, 0, "1.5k"), (4, 1, "3.0k"), (8, 2, "6.0k"))
BUDGETS = {
    0: [(0,) * 9],
    1: [(1,) + (0,) * 8, (0,) * 8 + (1,)],
    2: [(2,) + (0,) * 8, (1, 1) + (0,) * 7, (0,) * 7 + (1, 1)],
    3: [(3,) + (0,) * 8, (1, 1, 1) + (0,) * 6, (0,) * 6 + (1, 1, 1), (1, 0, 0, 1, 0, 0, 1, 0, 0),
        (2, 1) + (0,) * 7],
    4: [(3, 1) + (0,) * 7, (1,) * 4 + (0,) * 5, (2, 2) + (0,) * 7],
    6: [(3, 3) + (0,) * 7, (1,) * 6 + (0,) * 3, (2, 2, 2) + (0,) * 6],
    9: [(1,) * 9, (3, 3, 3) + (0,) * 6],
}
WINDOWS = (0, 1, 2, 3, 6)


def configs(quick: bool):
    out = [("full", None, None, 27)]
    for lookahead, rights_list in BUDGETS.items():
        for rights in rights_list:
            for edge in (("repeat",) if quick else ("zero", "repeat")):
                out.append(("budget", rights, edge, lookahead))
    if not quick:
        out += [("window", w, None, w) for w in WINDOWS]
    return out


def name(cfg):
    kind, arg, edge, lookahead = cfg
    if kind == "budget":
        return f"budget {''.join(map(str, arg))} {edge}"
    return "full" if kind == "full" else f"window {arg}"


def decode_window(t, codes, ids, lookahead, history=30):
    feats = torch.stack([ref.features(t, c) for c in codes])
    ids = torch.as_tensor(ids)
    hidden = []
    for i in range(len(codes)):
        lo, hi = max(0, i - history), min(len(codes), i + lookahead + 1)
        hidden.append(ref.backbone_batch(t, feats[lo:hi], ids[lo:hi])[i - lo])
    return ref.istft_batch(ref.head_spectrum(t, torch.stack(hidden))).numpy()


_state = {}


def work(args):
    path, cfgs = args
    torch.set_num_threads(1)
    if not _state:
        _state["t"] = ref.load(common.MODEL_DIR, half_weights=False)
        m = EncodecModel.encodec_model_24khz().eval()
        m.set_target_bandwidth(12.0)
        for mod in m.modules():
            if isinstance(mod, SConv1d):
                mod.pad_mode = "constant"
        _state["m"] = m
    t, m = _state["t"], _state["m"]
    x16, _ = common.read_wav(path)
    x = common.up_16k_to_24k(x16)
    with torch.no_grad():
        latent = m.encoder(torch.from_numpy(x)[None, None])
        codes = m.quantizer.encode(latent, 75, 12.0)[:, 0].T.numpy()
        scores = {}
        for n, bid, label in RATES:
            c = codes[:, :n].tolist()
            ids = [bid] * len(c)
            for cfg in cfgs:
                kind, arg, edge, _ = cfg
                if kind == "full":
                    y = ref.decode_batch(t, c, ids)
                elif kind == "budget":
                    y = ref.decode_batch(t, c, ids, arg, edge)
                else:
                    y = decode_window(t, c, ids, arg)
                scores[(label, name(cfg))] = common.pesq_wb(x16, common.down_24k_to_16k(y))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfgs = configs(args.quick)
    files = common.corpus("dev")
    with Pool(6) as pool:  # each worker loads torch; more exhausts the page file
        results = pool.map(work, [(f, cfgs) for f in files])
    table = {}
    for cfg in cfgs:
        row = {label: float(np.mean([r[(label, name(cfg))] for r in results]))
               for _, _, label in RATES}
        table[name(cfg)] = {"lookahead": cfg[3], **row}
    print(f"{'scheme':32} {'ahead':>5}  " + "  ".join(f"{l:>5}" for _, _, l in RATES) + "   mean")
    for key, row in table.items():
        values = [row[l] for _, _, l in RATES]
        print(f"{key:32} {row['lookahead']:5d}  " + "  ".join(f"{v:5.2f}" for v in values)
              + f"   {np.mean(values):.3f}")
    out = common.ROOT / "build" / "decoder_lookahead.json"
    out.write_text(json.dumps(table, indent=1))
    print(f"{len(files)} files, PESQ-WB means; written to {out}")


if __name__ == "__main__":
    main()
