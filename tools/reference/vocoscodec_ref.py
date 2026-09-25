"""Frame-by-frame reference of the vocoscodec codec, computed as the C++ does.

Reads the converted model files (tools/convert_models.py), not the upstream
packages, so it also checks the converter. Everything is float32 on one thread,
with the weights the C++ holds in half precision rounded to it (load()).

Encoder: EnCodec 24 kHz SEANet encoder, streaming. Every causal convolution
keeps its last (kernel - stride) inputs and the LSTM its (h, c) between frames.
At reset the history is zero (silence before key down). Upstream pads the
first frame by reflection instead, which a streaming encoder cannot do, so the
first frames can differ from encoding a whole file at once.

Decoder: Vocos, streaming with bounded lookahead L. Frame t is what the
whole-signal decoder gives for frame t when the signal ends after the last
frame received (t + L when packets come one per call, later for frames early
in a group received together): every layer sees zero beyond that frame. The C++ caches the values that no
longer depend on the edge; this reference recomputes a window per frame. The
inverse STFT (n_fft 1280, hop 320, "same" padding) adds 480 samples, so the
decoder's delay is 480 + 320 L samples.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelfile  # noqa: E402

torch.set_num_threads(1)

FRAME = 320
N_FFT = 1280
HOP = 320
ISTFT_PAD = (N_FFT - HOP) // 2  # 480
MAX_CODEBOOKS = 16
BITS = 10


# Weights the C++ holds in half precision by default (VOCOSCODEC_HALF_WEIGHTS):
# those read once per frame. Rounded here the same way (to nearest, ties even).
HALF = ("lstm0.w_ih", "lstm0.w_hh", "lstm1.w_ih", "lstm1.w_hh", "down3.w", "conv_out.w",
        "embed.w", "head.w") + tuple(f"block{b}.pw{n}.w" for b in range(8) for n in (1, 2))


def load(model_dir: Path, half_weights: bool = True):
    tensors = {}
    for name in ("encodec_encoder.bin", "encodec_codebooks.bin", "vocos_decoder.bin"):
        _, t = modelfile.read(Path(model_dir) / name)
        tensors.update({k: torch.from_numpy(v.copy()) for k, v in t.items()})
    if half_weights:
        for k in HALF:
            tensors[k] = tensors[k].half().float()
    return tensors


# ---------------------------------------------------------------- packets

def packet_bytes(codebooks: int) -> int:
    return (BITS * codebooks + 7) // 8


def codebooks_in_packet(size: int) -> int:
    """Codebook count of a packet of `size` bytes, 0 if no count gives that size."""
    n = size * 8 // BITS
    return n if 1 <= n <= MAX_CODEBOOKS and packet_bytes(n) == size else 0


def pack(indices) -> bytes:
    """10-bit indices, coarse codebook first, MSB first, zero bits to the byte end."""
    value, bits = 0, 0
    out = bytearray()
    for index in indices:
        value = (value << BITS) | int(index)
        bits += BITS
        while bits >= 8:
            bits -= 8
            out.append((value >> bits) & 0xFF)
    if bits:
        out.append((value << (8 - bits)) & 0xFF)
    return bytes(out)


def unpack(packet: bytes) -> list[int]:
    n = codebooks_in_packet(len(packet))
    if n == 0:
        raise ValueError(f"no codebook count gives a {len(packet)}-byte packet")
    value = int.from_bytes(packet, "big") >> (8 * len(packet) - BITS * n)
    return [(value >> (BITS * (n - 1 - i))) & 0x3FF for i in range(n)]


# ---------------------------------------------------------------- encoder

class _CausalConv:
    def __init__(self, w, b, stride=1):
        self.w, self.b, self.stride = w, b, stride
        self.history = w.shape[2] - stride
        self.reset()

    def reset(self):
        self.state = torch.zeros(self.w.shape[1], self.history)

    def __call__(self, x):  # x: (channels, samples), samples a multiple of stride
        if self.history:
            x = torch.cat([self.state, x], dim=1)
            self.state = x[:, x.shape[1] - self.history:]
        return F.conv1d(x[None], self.w, self.b, stride=self.stride)[0]


class StreamingEncoder:
    def __init__(self, tensors, codebooks: int):
        if not 1 <= codebooks <= MAX_CODEBOOKS:
            raise ValueError("codebook count must be 1..16")
        t = tensors
        self.codebooks = codebooks
        self.conv_in = _CausalConv(t["conv_in.w"], t["conv_in.b"])
        self.stages = []
        for stage, stride in enumerate((2, 4, 5, 8)):
            p = f"res{stage}."
            self.stages.append((_CausalConv(t[p + "conv1.w"], t[p + "conv1.b"]),
                                _CausalConv(t[p + "conv2.w"], t[p + "conv2.b"]),
                                _CausalConv(t[p + "skip.w"], t[p + "skip.b"]),
                                _CausalConv(t[f"down{stage}.w"], t[f"down{stage}.b"], stride)))
        self.lstm = [(t[f"lstm{i}.w_ih"], t[f"lstm{i}.w_hh"], t[f"lstm{i}.b"]) for i in range(2)]
        self.conv_out = _CausalConv(t["conv_out.w"], t["conv_out.b"])
        self.book = t["codebooks"][:codebooks]
        self.book_norm = (self.book ** 2).sum(dim=2)
        self.reset()

    def reset(self):
        for conv in [self.conv_in, self.conv_out] + [c for s in self.stages for c in s]:
            conv.reset()
        self.h = [torch.zeros(512) for _ in self.lstm]
        self.c = [torch.zeros(512) for _ in self.lstm]

    def embedding(self, pcm: np.ndarray) -> torch.Tensor:
        """320 samples (float, [-1, 1)) -> the 128-dimensional latent of the frame."""
        x = self.conv_in(torch.as_tensor(pcm, dtype=torch.float32).reshape(1, FRAME))
        for conv1, conv2, skip, down in self.stages:
            x = skip(x) + conv2(F.elu(conv1(F.elu(x))))
            x = down(F.elu(x))
        x = x[:, 0]  # one frame, 512 channels
        y = x
        for i, (w_ih, w_hh, b) in enumerate(self.lstm):
            gates = w_ih @ y + w_hh @ self.h[i] + b
            gi, gf, gg, go = gates.chunk(4)
            self.c[i] = torch.sigmoid(gf) * self.c[i] + torch.sigmoid(gi) * torch.tanh(gg)
            self.h[i] = torch.sigmoid(go) * torch.tanh(self.c[i])
            y = self.h[i]
        x = (y + x)[:, None]
        return self.conv_out(F.elu(x))[:, 0]

    def quantize(self, latent: torch.Tensor) -> list[int]:
        residual = latent.clone()
        indices = []
        for q in range(self.codebooks):
            # argmin |r - e|^2 = argmin (|e|^2 - 2 r.e); first index on ties.
            score = self.book_norm[q] - 2 * (self.book[q] @ residual)
            index = int(torch.argmin(score))
            indices.append(index)
            residual = residual - self.book[q, index]
        return indices

    def encode(self, pcm: np.ndarray) -> bytes:
        return pack(self.quantize(self.embedding(pcm)))


# ---------------------------------------------------------------- decoder

def bandwidth_id_for(codebooks: int, rule: str = "measured") -> int:
    """Vocos conditioning id (0..3 for 1.5/3/6/12 kbit/s = 2/4/8/16 codebooks)."""
    if rule == "measured":  # best PESQ-WB per count on the dev set (evaluate.py bandwidth)
        return 0 if codebooks <= 3 else 1 if codebooks <= 5 else 2 if codebooks <= 10 else 3
    if rule == "nearest_down":  # largest trained rate not above the packet's
        return 0 if codebooks < 4 else 1 if codebooks < 8 else 2 if codebooks < 16 else 3
    if rule == "nearest_up":
        return 0 if codebooks <= 2 else 1 if codebooks <= 4 else 2 if codebooks <= 8 else 3
    if rule.startswith("fixed"):
        return int(rule[5:])
    raise ValueError(rule)


def features(tensors, indices) -> torch.Tensor:
    """Sum of the embeddings of the codebooks present."""
    book = tensors["codebooks"]
    return sum(book[q, i] for q, i in enumerate(indices))


def _layer_norm(x, scale, shift):  # x: (..., 384)
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6) * scale + shift


def masked_kernel(w: torch.Tensor, right: int, edge: str) -> torch.Tensor:
    """A k=7 kernel seeing `right` of its 3 future taps (batch equivalent of streaming)."""
    w = w.clone()
    centre = w.shape[-1] // 2
    last = centre + right
    if edge == "repeat":
        w[..., last] += w[..., last + 1:].sum(dim=-1)
    elif edge != "zero":
        raise ValueError(edge)
    w[..., last + 1:] = 0
    return w


def backbone_batch(t, feats: torch.Tensor, ids: torch.Tensor, rights=(3,) * 9,
                   edge: str = "zero") -> torch.Tensor:
    """Whole-signal backbone with truncated right context. feats (T, 128) -> (T, 384)."""
    x = F.conv1d(feats.T[None], masked_kernel(t["embed.w"], rights[0], edge), t["embed.b"],
                 padding=3)[0].T
    x = _layer_norm(x, t["norm.scale"][ids], t["norm.shift"][ids])
    for k in range(8):
        p = f"block{k}."
        w = masked_kernel(t[p + "dw.w"], rights[k + 1], edge)[:, None, :]
        y = F.conv1d(x.T[None], w, t[p + "dw.b"], padding=3, groups=x.shape[1])[0].T
        y = _layer_norm(y, t[p + "norm.scale"][ids], t[p + "norm.shift"][ids])
        y = F.gelu(y @ t[p + "pw1.w"].T + t[p + "pw1.b"])
        x = x + y @ t[p + "pw2.w"].T + t[p + "pw2.b"]
    return F.layer_norm(x, (384,), t["final_norm.w"], t["final_norm.b"], eps=1e-6)


def head_magnitude_phase(t, hidden: torch.Tensor):
    """(T, 384) -> magnitudes and phases (T, 641) each."""
    out = hidden @ t["head.w"].T + t["head.b"]
    return torch.clamp(torch.exp(out[:, :641]), max=100.0), out[:, 641:]


def head_spectrum(t, hidden: torch.Tensor) -> torch.Tensor:
    """(T, 384) -> complex spectra (T, 641)."""
    mag, phase = head_magnitude_phase(t, hidden)
    return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))


def window() -> torch.Tensor:
    return torch.hann_window(N_FFT)  # periodic


def istft_batch(spectra: torch.Tensor) -> torch.Tensor:
    """Vocos ISTFT with "same" padding: T frames -> 320 T samples."""
    frames = torch.fft.irfft(spectra, N_FFT, dim=1) * window()
    total = (len(spectra) - 1) * HOP + N_FFT
    y = torch.zeros(total)
    envelope = torch.zeros(total)
    for i, frame in enumerate(frames):
        y[i * HOP:i * HOP + N_FFT] += frame
        envelope[i * HOP:i * HOP + N_FFT] += window() ** 2
    return (y / envelope)[ISTFT_PAD:total - ISTFT_PAD]


def decode_batch(t, codes, ids, rights=(3,) * 9, edge="zero") -> np.ndarray:
    """Whole-signal decode; codes is a list of per-frame index lists."""
    feats = torch.stack([features(t, c) for c in codes])
    hidden = backbone_batch(t, feats, torch.as_tensor(ids), rights, edge)
    return istft_batch(head_spectrum(t, hidden)).numpy()


WINDOW_HISTORY = 30  # > the 27-frame receptive field: the left edge never matters


def backbone_window(t, feats: torch.Tensor, ids: torch.Tensor, frame: int, edge: int):
    """Hidden frame `frame` with the right edge at frame `edge`."""
    lo = max(0, frame - WINDOW_HISTORY)
    return backbone_batch(t, feats[lo:edge + 1], ids[lo:edge + 1])[frame - lo]


def decode_window_batch(t, codes, ids, lookahead: int) -> np.ndarray:
    """Whole-signal equivalent of streaming decoding plus flush."""
    feats = torch.stack([features(t, c) for c in codes])
    ids = torch.as_tensor(ids)
    last = len(codes) - 1
    hidden = torch.stack([backbone_window(t, feats, ids, f, min(f + lookahead, last))
                          for f in range(len(codes))])
    return istft_batch(head_spectrum(t, hidden)).numpy()


# ---------------------------------------------------------------- concealment

REPEAT_FRAMES = 3  # consecutive losses rendered from repeated features
FADE_FRAMES = 3    # then frames of crossfade to comfort noise (or silence)


class Concealer:
    """Chooses features, bandwidth id and comfort noise weight for lost frames."""

    def __init__(self, repeat=REPEAT_FRAMES, fade=FADE_FRAMES):
        self.repeat, self.fade = repeat, fade
        self.reset()

    def reset(self):
        self.last, self.last_id, self.lost = torch.zeros(128), 0, 0

    def received(self, feat, bid):
        self.last, self.last_id, self.lost = feat, bid, 0

    def conceal(self):
        self.lost += 1
        into_fade = self.lost - self.repeat
        if into_fade <= 0:
            weight = 0.0
        elif self.fade <= 0 or into_fade >= self.fade:
            weight = 1.0
        else:
            weight = np.float32(into_fade) / np.float32(self.fade)
        return self.last.clone(), self.last_id, float(weight)


class ComfortNoise:
    """Noise floor of the last second's received frames, with random phases."""
    HISTORY, QUIETEST, BINS = 75, 15, 641

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.mags = np.zeros((self.HISTORY, self.BINS), np.float32)
        self.energy = np.zeros(self.HISTORY, np.float32)
        self.count, self.next, self.random = 0, 0, 0x6D2B79F5

    def observe(self, magnitudes: np.ndarray):
        self.mags[self.next] = magnitudes
        energy = np.float32(0)
        for m in magnitudes.astype(np.float32):  # in order, as the C++ sums
            energy = np.float32(energy + m * m)
        self.energy[self.next] = energy
        self.next = (self.next + 1) % self.HISTORY
        self.count = min(self.count + 1, self.HISTORY)

    def floor(self) -> np.ndarray:
        out = np.zeros(self.BINS, np.float32)
        if self.count:
            quietest = max(1, self.count * self.QUIETEST // self.HISTORY)
            order = sorted(range(self.count), key=lambda i: (self.energy[i], i))[:quietest]
            for i in order:
                out = out + self.mags[i] / np.float32(quietest)
        return out

    def mix(self, spectrum: torch.Tensor, weight: float) -> torch.Tensor:
        if weight <= 0:
            return spectrum
        floor = self.floor() if self.enabled else np.zeros(self.BINS, np.float32)
        phases = np.zeros(self.BINS, np.float32)
        if self.enabled:
            r = self.random
            for k in range(self.BINS):  # xorshift32, top 24 bits
                r ^= (r << 13) & 0xFFFFFFFF
                r ^= r >> 17
                r ^= (r << 5) & 0xFFFFFFFF
                phases[k] = np.float32(r >> 8) * np.float32(6.28318530717958648 / 16777216.0)
            self.random = r
        noise = torch.polar(torch.from_numpy(floor), torch.from_numpy(phases))
        return (1 - weight) * spectrum + weight * noise


# ---------------------------------------------------------------- streaming decoder

class StreamingDecoder:
    def __init__(self, tensors, lookahead: int = 3, rule: str = "measured",
                 comfort_noise: bool = True, repeat=REPEAT_FRAMES, fade=FADE_FRAMES):
        self.t, self.lookahead, self.rule = tensors, lookahead, rule
        self.delay = ISTFT_PAD + HOP * lookahead
        self.window = window()
        self.comfort = ComfortNoise(comfort_noise)
        self.concealer = Concealer(repeat, fade)
        self.reset()

    def reset(self):
        self.comfort.reset()
        self.concealer.reset()
        self.noise, self.concealed = [], []
        self.feats, self.ids = [], []
        self.ola = torch.zeros(N_FFT)
        self.envelope = torch.zeros(N_FFT)
        self.emitted = -1  # last frame through the ISTFT
        self.samples_out = 0

    def _synthesize(self, frame: int) -> torch.Tensor:
        edge = len(self.feats) - 1
        hidden = backbone_window(self.t, torch.stack(self.feats), torch.as_tensor(self.ids),
                                 frame, edge)
        mag, phase = head_magnitude_phase(self.t, hidden[None])
        spectrum = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))[0]
        if not self.concealed[frame]:
            self.comfort.observe(mag[0].numpy())
        spectrum = self.comfort.mix(spectrum, self.noise[frame])
        self.ola += torch.fft.irfft(spectrum, N_FFT) * self.window
        self.envelope += self.window ** 2
        out = torch.where(self.envelope[:HOP] > 1e-11, self.ola[:HOP] / self.envelope[:HOP],
                          torch.zeros(HOP))
        self.ola = torch.cat([self.ola[HOP:], torch.zeros(HOP)])
        self.envelope = torch.cat([self.envelope[HOP:], torch.zeros(HOP)])
        self.emitted = frame
        return out

    def _emit(self, out: torch.Tensor) -> np.ndarray:
        """Samples before signal time 0 are silence (the "same" padding)."""
        out = out.clone()
        start = self.samples_out - self.delay
        if start < 0:
            out[:min(len(out), -start)] = 0
        self.samples_out += len(out)
        return out.numpy()

    def decode_group(self, feats, bids, noise=None, concealed=None) -> np.ndarray:
        """Frames received together -> 320 samples each, `delay` samples behind.

        Every frame synthesized here sees up to the last frame of the group,
        so frames early in a group look further ahead than `lookahead`."""
        self.feats.extend(feats)
        self.ids.extend(bids)
        self.noise.extend(noise or [0.0] * len(feats))
        self.concealed.extend(concealed or [False] * len(feats))
        last = len(self.feats) - 1 - self.lookahead
        first = self.emitted + 1
        ready = [self._synthesize(frame) for frame in range(first, last + 1)]
        waiting = [torch.zeros(HOP)] * (len(feats) - len(ready))
        return self._emit(torch.cat(waiting + ready))

    def decode_features(self, feat: torch.Tensor, bid: int) -> np.ndarray:
        """One frame's features -> 320 samples, `delay` samples behind the input."""
        return self.decode_group([feat], [bid])

    def decode_indices(self, indices) -> np.ndarray:
        return self.decode_features(features(self.t, indices),
                                    bandwidth_id_for(len(indices), self.rule))

    def decode(self, packet: bytes) -> np.ndarray:
        return self.decode_packets([packet])

    def decode_packets(self, packets) -> np.ndarray:
        """Packets received together; an empty one (or None) was lost."""
        feats, bids, noise, concealed = [], [], [], []
        for packet in packets:
            if packet:
                indices = unpack(packet)
                feat, bid = features(self.t, indices), bandwidth_id_for(len(indices), self.rule)
                self.concealer.received(feat, bid)
                weight = 0.0
            else:
                feat, bid, weight = self.concealer.conceal()
            feats.append(feat)
            bids.append(bid)
            noise.append(weight)
            concealed.append(not packet)
        return self.decode_group(feats, bids, noise, concealed)

    def flush(self) -> np.ndarray:
        """The last `delay` samples, as if the signal ended here; then reset."""
        parts = []
        first = self.emitted + 1
        gap = (first + self.lookahead) * HOP - self.samples_out
        if gap > 0:
            parts.append(self._emit(torch.zeros(gap)))
        for frame in range(first, len(self.feats)):
            parts.append(self._emit(self._synthesize(frame)))
        tail = torch.where(self.envelope[:ISTFT_PAD] > 1e-11,
                           self.ola[:ISTFT_PAD] / self.envelope[:ISTFT_PAD],
                           torch.zeros(ISTFT_PAD))
        parts.append(self._emit(tail))
        self.reset()
        return np.concatenate(parts)
