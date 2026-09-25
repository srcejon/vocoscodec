# vocoscodec validation

Measurements of the codec as built, made on 2026-09-25 on these conditions:
- **CPU:** Intel Core i9-14900K, Windows 11, MSVC 2022, AVX2 kernels, unless
  stated otherwise.
- **Speech:** two disjoint sets, taken from archives already on disk by
  `tools/prepare_corpus.py`. Each file is 16 kHz mono, at most 10 s, with
  its active speech level set to −26 dBov.
- **Scoring:** PESQ-WB (ITU-T P.862.2, `pesq` package) and STOI (`pystoi`),
  against the original 16 kHz file.
- **The codec path:** 16 kHz → 24 kHz (scipy polyphase), then
  `vocoscodec_encode`, then `vocoscodec_decode` (which trims the codec delay),
  then 24 kHz → 16 kHz.

| Set | Files | Speech | Speakers | Used for |
|---|---:|---:|---|---|
| dev | 20 | 144 s | 10 (5 F): LibriSpeech test-clean 1089, 121, 237, 260, 672, 1221, 1320, 1580; CMU ARCTIC bdl, slt | every design choice |
| test | 32 | 195 s | 16 (8 F): LibriSpeech test-clean 2830, 3729, 4446, 5105, 5639, 7729, 8463, 8555; test-other 1688, 1998, 3331, 4198; CMU ARCTIC awb, clb, jmk, ksp | this report only, after every choice was made |

`tools/evaluate.py quality` reproduces the test-set tables, with results in
`build/eval/quality.json`. The dev-set experiments are its `bandwidth`,
`lookahead` and `concealment` commands.

## Quality against Lyra (test set)

vocoscodec runs with its defaults: lookahead 3, one packet per decode call
(the minimum lookahead), and the bandwidth rule below. Lyra is the Lyra fork's
`lyracodec.dll` (C ABI 2, OpenCV 4.14 backend), through its C API.

| Codec | bit/s | PESQ-WB | STOI |
|---|---:|---:|---:|
| vocoscodec, 1 codebook | 750 | 1.23 | 0.781 |
| vocoscodec, 2 codebooks | 1500 | 1.88 | 0.877 |
| vocoscodec, 3 codebooks | 2250 | **2.33** | 0.900 |
| Lyra | 3200 | **2.33** | 0.897 |
| vocoscodec, 4 codebooks | 3000 | **2.59** | 0.922 |
| vocoscodec, 6 codebooks | 4500 | 2.82 | 0.936 |
| Lyra | 6000 | 2.93 | 0.929 |
| vocoscodec, 8 codebooks | 6000 | **3.08** | 0.945 |
| Lyra | 9200 | 3.17 | 0.940 |
| vocoscodec, 12 codebooks | 9000 | **3.23** | 0.953 |
| vocoscodec, 16 codebooks | 12000 | 3.39 | 0.958 |

- **2.25 kbit/s:** equals Lyra's 3.2 kbit/s (2.33 against 2.33), at 70% of
  the rate.
- **3.0 kbit/s:** beats Lyra's 3.2 kbit/s by 0.26.
- **6 kbit/s:** beats Lyra's 6 kbit/s by 0.15.
- **9 kbit/s:** beats Lyra's 9.2 kbit/s by 0.06.
- **STOI** follows the same order.
- **Against PLAN.md's two clips:** this larger set scores lower for both
  codecs, with smaller gaps.

### Streaming against the original models

This is what the streaming encoder, the bounded lookahead and the C++ port
cost, against PyTorch EnCodec and Vocos run on whole files with unbounded
lookahead:

| Codebooks | PyTorch, whole file | vocoscodec, 1 packet per call | vocoscodec, 9 per call |
|---:|---:|---:|---:|
| 2 | 1.96 | 1.88 | 1.92 |
| 4 | 2.65 | 2.59 | 2.61 |
| 8 | 3.10 | 3.08 | 3.09 |

The loss comes from the lookahead of 3 frames (60 ms of decoder delay).
Groups of packets, as Angel61 delivers them, recover about half of it (next
section).

## Decoder lookahead (dev set)

| Codebooks | L=0 | L=1 | L=2 | **L=3** | L=6 | L=27 (unbounded) | L=3, 9 per call | L=3, 18 per call |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.59 | 1.78 | 1.88 | **1.90** | 1.94 | 1.97 | 1.93 | 1.95 |
| 4 | 2.25 | 2.61 | 2.69 | **2.68** | 2.74 | 2.75 | 2.74 | 2.74 |
| 8 | 2.85 | 3.17 | 3.22 | **3.23** | 3.25 | 3.25 | 3.25 | 3.25 |

- **The default, L = 3:** it is within 0.07 of unbounded lookahead. L = 2 is
  as good at 4 and 8 codebooks, but 0.02 worse at 2 codebooks, which is
  Angel61's lowest HF tier.
- **Delay:** 480 + 320 L samples at 24 kHz, 60 ms at L = 3.
- **Packets per call:** when several arrive together, the frames early in
  the group see up to its end. Groups of 9 to 18 come within 0.02 of
  unbounded lookahead.
- **The rejected alternative:** masking each layer's future taps, so that
  every layer computes each frame once, costs 0.7 PESQ at the same total
  lookahead (`tools/reference/decoder_lookahead.py`, and PLAN.md phase 2).

## Bandwidth embedding (dev set)

Vocos was trained with conditioning embeddings for 1.5, 3, 6 and 12 kbit/s,
that is 2, 4, 8 and 16 codebooks. Every count was scored with every
embedding:

| Codebooks | 1.5k | 3k | 6k | 12k | Rule |
|---:|---:|---:|---:|---:|:---:|
| 1 | **1.21** | 1.15 | 1.12 | 1.11 | 1.5k |
| 2 | **1.90** | 1.70 | 1.44 | 1.31 | 1.5k |
| 3 | **2.35** | 2.33 | 2.00 | 1.66 | 1.5k |
| 4 | 2.59 | **2.68** | 2.50 | 2.08 | 3k |
| 5 | 2.68 | **2.87** | 2.80 | 2.46 | 3k |
| 6 | 2.74 | **2.97** | **2.97** | 2.74 | 6k |
| 7 | 2.78 | 3.08 | **3.13** | 2.93 | 6k |
| 8 | 2.84 | 3.12 | **3.22** | 3.09 | 6k |
| 9 | 2.87 | 3.17 | **3.28** | 3.20 | 6k |
| 10 | 2.88 | 3.19 | **3.33** | 3.30 | 6k |
| 11 | 2.88 | 3.21 | **3.37** | **3.37** | 12k |
| 12 | 2.89 | 3.22 | 3.41 | **3.43** | 12k |
| 13 | 2.89 | 3.24 | 3.43 | **3.47** | 12k |
| 14 | 2.90 | 3.25 | 3.46 | **3.50** | 12k |
| 15 | 2.91 | 3.27 | 3.47 | **3.54** | 12k |
| 16 | 2.91 | 3.26 | 3.49 | **3.56** | 12k |

The rule (`BandwidthId`):
- 1–3 codebooks: 1.5k;
- 4–5 codebooks: 3k;
- 6–10 codebooks: 6k;
- 11–16 codebooks: 12k.

Ties go to the higher embedding. Three codebooks (2.25k) do best with the
1.5k embedding, which confirms the earlier two-clip observation. Twelve
codebooks (9k, the VHF tier with no embedding of its own) do best with 12k.

## Lost packets

### Design (dev set, 3.0k)

A lost frame's features repeat the last good frame's for R frames. Then the
spectrum crossfades over F frames to comfort noise or silence. Mean PESQ-WB:

| Loss | R/F 0/0 | 1/2 | **3/3** | 5/3 | 8/4 | repeat only |
|---|---:|---:|---:|---:|---:|---:|
| random 1% | 2.542 | 2.561 | 2.561 | 2.561 | 2.561 | 2.561 |
| random 5% | 2.187 | 2.255 | 2.256 | 2.256 | 2.256 | 2.256 |
| random 10% | 1.790 | 1.905 | 1.899 | 1.899 | 1.899 | 1.899 |
| random 20% | 1.443 | 1.549 | 1.548 | 1.546 | 1.546 | 1.546 |
| burst 1% | 2.593 | 2.612 | 2.610 | 2.610 | 2.610 | 2.610 |
| burst 5% | 1.975 | 2.048 | **2.093** | 2.076 | 2.073 | 2.073 |
| burst 10% | 1.575 | 1.688 | **1.727** | 1.713 | 1.708 | 1.702 |
| burst 20% | 1.328 | 1.431 | **1.470** | 1.453 | 1.449 | 1.448 |

- **The choice, R = 3 and F = 3** (40 ms, then a 40 ms fade): it is best
  under burst losses and within 0.006 of the best under random losses.
- **Fading at once (0/0):** clearly worse.
- **Repeating without end:** scores almost as well on PESQ, but a long
  outage then becomes a repeating buzz. That is what the fade prevents.
- **After the fade:** the comfort noise has the magnitudes of the quietest
  fifth of the last second received, and random phases. It is synthesized
  by the same inverse STFT, so entering and leaving it overlap-add smoothly.
- **Test coverage:** `vocoscodec_packet_loss_test` checks a one-second
  outage. Time is kept, nothing clips, consecutive frames of comfort noise
  differ, the level returns within 0.1 s, and the silence option gives
  RMS < 1 LSB.
- **Caveat:** these dev patterns used the same random seed for every file,
  so their realised loss rates differ from the nominal ones. The ranking
  compares timings on identical patterns, so it stands. The test-set
  measurement below uses independent patterns per file.

### Measurement (test set, 3.0k, against Lyra 3.2k)

Three loss patterns per file, 32 files, and the realised loss in brackets.
Mean PESQ-WB; each codec is scored against the original speech:

| Loss | vocoscodec 3.0k, comfort noise | 3.0k, silence | Lyra 3.2k |
|---|---:|---:|---:|
| none | 2.59 | 2.59 | 2.33 |
| random 1% | **2.48** (1.0%) | 2.48 | 2.26 (1.1%) |
| random 5% | **2.11** (5.1%) | 2.11 | 2.03 (5.0%) |
| random 10% | 1.84 (10.1%) | 1.84 | 1.81 (10.1%) |
| random 20% | 1.52 (20.0%) | 1.52 | 1.52 (19.9%) |
| burst 1% | **2.45** (1.0%) | 2.44 | 2.25 (1.1%) |
| burst 5% | **2.07** (4.9%) | 2.05 | 1.97 (5.2%) |
| burst 10% | 1.71 (10.4%) | 1.68 | 1.73 (10.3%) |
| burst 20% | 1.38 (20.2%) | 1.36 | 1.43 (20.2%) |
| radio frame 5% | **2.18** (4.9%) | 2.14 | 2.01 (4.9%) |
| radio frame 10% | **1.86** (10.5%) | 1.81 | 1.76 (10.5%) |
| radio frame 20% | 1.51 (19.7%) | 1.46 | 1.47 (19.7%) |

Burst losses have a mean length of 53 ms. Radio-frame losses drop whole
240 ms groups: 18 vocoscodec packets or 12 Lyra packets, as when an Angel61
HF radio frame is lost. The patterns are in time, so both codecs lose
comparable audio.

- **Up to 5% loss, of any kind:** vocoscodec stays ahead of Lyra at its
  nearest higher rate.
- **Whole radio frames lost** (the Angel61 case): it stays ahead up to 20%.
- **Heavy loss:** at 10 to 20% random loss the two converge. Under heavy
  bursts, Lyra's concealment degrades more gently (by 0.02 and 0.05).
- **Comfort noise** is never worse than silence, and is better during long
  outages.

## Numerical agreement

**Against the Python reference** (`vocoscodec_golden_test`, 1.6 s of
LibriSpeech; reference and C++ both use half-precision weights):

| Build | Encoder indices equal | Decoder SNR against the reference |
|---|---|---|
| Windows AVX2 | 100.00% (all 16 codebooks) | 101–111 dB |
| Linux x64, GCC 13, AVX2 | 100.00% | 101–111 dB |
| Windows SSE4.2 (float weights) | 99.90% (codebook 16: 119 of 120) | 65 dB |
| Windows portable kernels (float weights) | 99.90% | 65 dB |

The SSE4.2 and portable kernels hold every weight as float, because they
have no hardware half conversion (see "x86 without AVX2" below). They
therefore differ from the half-precision reference by exactly the rounding
of those weights. A float32 build (`-DVOCOSCODEC_HALF_WEIGHTS=OFF`) gives
the same figures.

Lossy decoding (an isolated loss, a 2-frame burst, a 40-frame outage) matches
the reference's concealment, including its comfort noise, to 75.8 dB. That
is the limit of 16-bit output.

**Across builds, test set** (14,639 frames, 16 codebooks):

| Comparison | Indices equal |
|---|---|
| Windows MSVC AVX2 vs Linux GCC AVX2 | 100.00% (bit-identical) |
| AVX2 (half weights) vs SSE4.2 (float weights) | 99.97 / 99.92 / 99.85 / 99.65 / 99.06% (codebooks 1/2/4/8/16), 99.58% overall |
| AVX2 (half weights) vs portable (float weights) | 99.97 / 99.91 / 99.84 / 99.64 / 99.06%, 99.57% overall |
| SSE4.2 vs portable (both float weights) | 99.99% |

With the same weights, different kernels agree to 99.99%. The 0.4% between
AVX2 and the others is the half-precision rounding, almost all of it in the
finest codebooks. That costs no measurable quality (see "Half-precision
weights" below).

The arm64 NEON kernels were compiled with the Android NDK r30 (arm64-v8a,
no warnings) but not run: no arm64 machine was available. Running `ctest`
on a Raspberry Pi validates them against the same golden vectors.

**Against upstream:**
- **Encoder:** the streaming encoder matches PyTorch EnCodec with zero
  padding to a maximum latent difference of 6.5e-5. With upstream's reflect
  padding at the start of a file, the indices differ while its first frames
  settle.
- **Decoder:** the reference decoder matches the `vocos` package to 1e-5.
- **Half-precision weights** (the build default) change 0.6% of encoder
  indices against float32, almost all in the finest codebooks, and no
  decoder PESQ measurably (`tools/reference/fp16_weights.py`, 6 dev files:
  2.937 either way).

## CPU, memory and delay

Windows 11, i9-14900K, one performance core pinned (`benchmark.bat`, CPU 2),
half-precision weights. The input is a 10 s test-set utterance looped over
1500 frames; the figures are the median of three runs. A frame is 13.33 ms.

| Configuration | Encode ms/frame | Decode ms/frame | Decoder blocks/frame |
|---|---:|---:|---:|
| 4 codebooks, lookahead 3, 1 packet per decode call | 0.87 (6.5%) | 1.80 (13.5%) | 79.7 |
| the same, 9 packets per call | 0.92 (6.9%) | 0.41 (3.1%) | 18.3 |
| the same, 18 packets per call | 0.92 (6.9%) | 0.31 (2.3%) | 13.8 |
| lookahead 0, 1 per call | 0.91 | 1.87 (14.0%) | 68.8 |
| lookahead 27 (unbounded), 1 per call | 0.98 | 0.86 (6.5%) | 7.9 |
| 2 / 8 codebooks, 9 per call | 0.85 / 1.01 | 0.42 / 0.42 | 18.3 |

### x86 without AVX2

On x86-64 the kernels are chosen at run time: AVX2 with FMA and F16C if the
CPU has them, else SSE4.2, else portable C++. SSE4.2 covers CPUs before
Haswell (2013), and the Atom-class cores (up to Tremont) of low-power
PCs.

SSE4.2 has 4-wide vectors and no fused multiply-add. Without F16C, turning
half weights into floats costs more than the memory traffic it saves. Even
the lean integer conversion encoded at 3.0 ms per frame against 2.1 ms with
float weights. So the SSE4.2 and portable kernels hold all weights as float;
the models then take 70 MB in memory rather than 41 MB.

The same machine and method as above, forcing each kernel set with
`vocoscodec_benchmark --kernels`, median of three:

| Kernels | Encode | Decode, 1 per call | Decode, 9 per call | Decode, 18 per call |
|---|---:|---:|---:|---:|
| AVX2 + FMA + F16C (half weights) | 0.89 ms (6.7%) | 1.80 ms (13.5%) | 0.39 ms (2.9%) | 0.30 ms (2.2%) |
| SSE4.2 (float weights) | 1.96 ms (14.7%) | 4.92 ms (36.9%) | 0.99 ms (7.4%) | 0.67 ms (5.1%) |
| portable C++ (float weights, compiler-vectorised SSE2) | 4.05 ms (30.4%) | 5.83 ms (43.7%) | 1.26 ms (9.5%) | 0.99 ms (7.4%) |

- **SSE4.2 against AVX2:** encoding takes 2.2 times as long, decoding 2.2 to
  2.7 times.
- **The cause:** a quarter of the arithmetic per instruction (4-wide, with
  separate multiply and add), and twice the weight traffic of half
  precision.
- **Older hardware:** an SSE4.2-only CPU of 2010 to 2012 would also clock
  lower and have less memory bandwidth, so expect roughly 25 to 35% of a
  core to encode and 10 to 15% to decode in radio-frame groups. That is an
  estimate, not a measurement.
- **Its first version** tiled 8 outputs, half a cache line of each weight
  row. It decoded more slowly than the compiler-vectorised portable code
  (7.8 ms against 5.8 ms at one packet per call), so its tiles now span 16
  outputs, as AVX2's do.

How the costs arise:
- **Decoder blocks.** A Vocos decoder without lookahead limits evaluates 8
  ConvNeXt blocks per frame. The bounded lookahead recomputes the values
  near the newest frame on every call. That costs 80 blocks per frame with
  one packet per call, and 18 or 14 per frame when a radio frame of 9 or 18
  packets arrives at once. `tools/cost_model.py` gives the arithmetic: 71,
  17 and 13 M multiply-accumulates per frame respectively.
- **The encoder** is 19.9 M MAC per frame. It is bound by memory traffic:
  16.2 MB of weights are read per frame, mostly by the LSTM.
- **Silence** costs the same as speech: there is no denormal slowdown.

For comparison:
- **Lyra** (OpenCV backend, 20 ms frames) takes 1.22 ms encode and 1.71 ms
  decode per frame, that is 6.1% and 8.6% (`lyra_codec/validation`).
- **PyTorch on one core, whole files:** EnCodec encoding about 5%, Vocos
  decoding about 1% (PLAN.md).

**Memory**, from `vocoscodec_benchmark` after loading one encoder and one
decoder:

| Item | Size |
|---|---:|
| encoder weights (float, with LSTM and frame-rate convolutions in half) | 16.2 MB |
| codebooks (shared by encoder and decoder) | 8.5 MB |
| decoder weights (float, linear layers in half) | 16.1 MB |
| decoder instance state (lookahead cache, buffers) | about 3.6 MB |
| encoder instance state | about 0.3 MB |
| process resident after loading both | 50.0 MB |
| peak while loading (the float32 files are read, then converted) | 80.0 MB |
| model files on disk | 70.0 MB |

Models are loaded in about 130 ms and shared by every instance in a process.

**Delay** at the default lookahead of 3:
- **Decoder:** 1440 samples, 60 ms.
- **Encoder:** one frame of buffering, 13.3 ms.
- **In total:** 73 ms of algorithmic delay, plus any buffering in the
  protocol's radio frames.

## Raspberry Pi (estimate)

No Raspberry Pi was available, so these are estimates.

**Method.** The per-frame arithmetic and weight traffic come from
`tools/cost_model.py`. The efficiency comes from this PC: decoding at 9 per
call reaches 42 GMAC/s, 46% of the core's 91 GMAC/s FMA peak. The same
efficiency was assumed for the NEON kernels, and the memory bandwidths are
typical single-core figures.

| | Pi 4 (Cortex-A72, 1.8 GHz) | Pi 5 (Cortex-A76, 2.4 GHz) |
|---|---|---|
| NEON FMA peak, and at 46% | 7.2 GMAC/s, 3.3 | 19.2 GMAC/s, 8.8 |
| single-core memory bandwidth | about 3.5 GB/s | about 10 GB/s |
| encoder (19.9 M MAC + 16.2 MB per frame) | 6 to 11 ms: **45 to 80%** of a core | 2.3 to 3.9 ms: **17 to 29%** |
| decoder, 18 per call (12.9 M MAC, 0.9 MB) | about 4.2 ms: **31%** | about 1.6 ms: **12%** |
| decoder, 9 per call (17.2 M MAC, 1.8 MB) | about 5.7 ms: **43%** | about 2.1 ms: **16%** |
| decoder, 1 per call (71 M MAC, 16 MB) | about 26 ms: **not real time** | about 9.7 ms: **73%** |

The encoder range spans memory access overlapping with arithmetic fully (the
low figure) or not at all (the high figure).

- **Pi 5:** comfortable.
- **Pi 4:** possible when packets are decoded in radio-frame groups, as
  Angel61 delivers them. Transmit and receive don't run at once, and the
  other three cores stay free. The encoder takes most of one core, though,
  and decoding one packet per call is too slow.
- **If a Pi 4 falls short,** the next steps are 8-bit weights for the LSTM
  and linear layers (a quarter of today's traffic), or an encoder called
  with several frames at once. Measure first: build with
  `-DVOCOSCODEC_BUILD_TOOLS=ON -DVOCOSCODEC_BUILD_TESTS=ON` on the Pi, then
  run `ctest` and `vocoscodec_benchmark <models> --group 18`.
