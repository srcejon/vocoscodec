"""Convert the EnCodec 24 kHz and Vocos checkpoints to vocoscodec model files.

Writes three files to the model directory:

  encodec_encoder.bin    SEANet encoder and LSTM, weight norm folded
  encodec_codebooks.bin  the first 16 residual VQ codebooks (1024 x 128 each)
  vocos_decoder.bin      Vocos backbone and ISTFT head, layer scale folded

and SHA256SUMS listing them. Needs torch only (not the encodec or vocos
packages). Sources:

  EnCodec: encodec_24khz-d7cc33bc.th (Meta; code MIT, weights licence
           unresolved, see PLAN.md phase 0)
  Vocos:   charactr/vocos-encodec-24khz pytorch_model.bin (MIT)
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

import modelfile

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.path.expanduser("~"))
ENCODEC = HOME / ".cache/torch/hub/checkpoints/encodec_24khz-d7cc33bc.th"
VOCOS_DIR = HOME / ".cache/huggingface/hub/models--charactr--vocos-encodec-24khz/snapshots"
CODEBOOKS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_weight_norm(state, prefix):
    g = state[prefix + ".weight_g"].double()
    v = state[prefix + ".weight_v"].double()
    norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(g.shape)
    return (g * v / norm).float().numpy(), state[prefix + ".bias"].numpy()


def convert_encoder(state, out: Path, source: Path):
    tensors = {}

    def conv(name, index, suffix="conv.conv"):
        w, b = fold_weight_norm(state, f"encoder.model.{index}.{suffix}")
        tensors[name + ".w"], tensors[name + ".b"] = w, b

    conv("conv_in", 0)
    # Residual block at model index i, downsampling conv at i + 2.
    for stage, index in enumerate((1, 4, 7, 10)):
        conv(f"res{stage}.conv1", index, "block.1.conv.conv")
        conv(f"res{stage}.conv2", index, "block.3.conv.conv")
        conv(f"res{stage}.skip", index, "shortcut.conv.conv")
        conv(f"down{stage}", index + 2)
    for layer in range(2):
        prefix = "encoder.model.13.lstm."
        tensors[f"lstm{layer}.w_ih"] = state[prefix + f"weight_ih_l{layer}"].numpy()
        tensors[f"lstm{layer}.w_hh"] = state[prefix + f"weight_hh_l{layer}"].numpy()
        tensors[f"lstm{layer}.b"] = (state[prefix + f"bias_ih_l{layer}"].double()
                                     + state[prefix + f"bias_hh_l{layer}"].double()).float().numpy()
    conv("conv_out", 15)
    meta = {"model": "encodec_24khz_encoder", "sample_rate": 24000, "frame_samples": 320,
            "strides": "2,4,5,8", "activation": "elu", "lstm_layers": 2,
            "source": source.name, "source_sha256": sha256(source)}
    modelfile.write(out, meta, tensors)


def convert_codebooks(state, out: Path, source: Path):
    books = np.stack([state[f"quantizer.vq.layers.{i}._codebook.embed"].numpy()
                      for i in range(CODEBOOKS)])
    meta = {"model": "encodec_24khz_codebooks", "codebooks": CODEBOOKS, "entries": 1024,
            "dimension": 128, "source": source.name, "source_sha256": sha256(source)}
    modelfile.write(out, meta, {"codebooks": books})


def convert_decoder(state, out: Path, source: Path):
    tensors = {"embed.w": state["backbone.embed.weight"].numpy(),
               "embed.b": state["backbone.embed.bias"].numpy(),
               "norm.scale": state["backbone.norm.scale.weight"].numpy(),
               "norm.shift": state["backbone.norm.shift.weight"].numpy()}
    blocks = 0
    while f"backbone.convnext.{blocks}.gamma" in state:
        p = f"backbone.convnext.{blocks}."
        gamma = state[p + "gamma"].double()
        tensors[f"block{blocks}.dw.w"] = state[p + "dwconv.weight"][:, 0, :].numpy()
        tensors[f"block{blocks}.dw.b"] = state[p + "dwconv.bias"].numpy()
        tensors[f"block{blocks}.norm.scale"] = state[p + "norm.scale.weight"].numpy()
        tensors[f"block{blocks}.norm.shift"] = state[p + "norm.shift.weight"].numpy()
        tensors[f"block{blocks}.pw1.w"] = state[p + "pwconv1.weight"].numpy()
        tensors[f"block{blocks}.pw1.b"] = state[p + "pwconv1.bias"].numpy()
        # Layer scale folded: gamma * (W x + b) = (gamma W) x + gamma b.
        tensors[f"block{blocks}.pw2.w"] = (gamma[:, None]
                                           * state[p + "pwconv2.weight"].double()).float().numpy()
        tensors[f"block{blocks}.pw2.b"] = (gamma * state[p + "pwconv2.bias"].double()).float().numpy()
        blocks += 1
    tensors["final_norm.w"] = state["backbone.final_layer_norm.weight"].numpy()
    tensors["final_norm.b"] = state["backbone.final_layer_norm.bias"].numpy()
    tensors["head.w"] = state["head.out.weight"].numpy()
    tensors["head.b"] = state["head.out.bias"].numpy()
    meta = {"model": "vocos_encodec_24khz_decoder", "sample_rate": 24000, "hop": 320,
            "n_fft": 1280, "blocks": blocks, "bandwidths": "1.5,3,6,12",
            "layer_norm_eps": "1e-6", "source": "charactr/vocos-encodec-24khz " + source.parent.name,
            "source_sha256": sha256(source)}
    modelfile.write(out, meta, tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encodec", type=Path, default=ENCODEC)
    parser.add_argument("--vocos", type=Path,
                        default=next(VOCOS_DIR.glob("*/pytorch_model.bin"), None))
    parser.add_argument("--out", type=Path, default=ROOT / "model")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    encodec = torch.load(args.encodec, map_location="cpu")
    vocos = torch.load(args.vocos, map_location="cpu")
    # The Vocos embedding table is EnCodec's codebooks; one file serves both.
    table = vocos["feature_extractor.codebook_weights"].reshape(CODEBOOKS, 1024, 128)
    for i in range(CODEBOOKS):
        if not torch.equal(table[i], encodec[f"quantizer.vq.layers.{i}._codebook.embed"]):
            raise SystemExit(f"Vocos embedding {i} differs from EnCodec codebook {i}")

    files = {"encodec_encoder.bin": lambda p: convert_encoder(encodec, p, args.encodec),
             "encodec_codebooks.bin": lambda p: convert_codebooks(encodec, p, args.encodec),
             "vocos_decoder.bin": lambda p: convert_decoder(vocos, p, args.vocos)}
    lines = []
    for name, convert in files.items():
        convert(args.out / name)
        digest = sha256(args.out / name)
        lines.append(f"{digest}  {name}\n")
        print(f"{name}: {(args.out / name).stat().st_size / 1e6:.1f} MB  sha256 {digest}")
    (args.out / "SHA256SUMS").write_text("".join(lines), newline="\n")


if __name__ == "__main__":
    main()
