# vocoscodec: a standalone EnCodec + Vocos speech codec for Angel61

## Why

Angel61 (the SDRangel digital voice protocol, spec in
`srcejon_sdrangel_bugfix/sdrbase/dsp/dv/ANGEL61.md`) uses Lyra (codec family
L1) through `lyracodec`, a runtime-loaded shared library built from the Lyra
fork in `C:\Users\jon\source\repos\lyra`. While designing an HF profile, EnCodec's
encoder with the Vocos decoder measured better than Lyra at every bit rate, on
the same two clips (`lyra/testdata`, PESQ-WB, mean of the two):

| Codec | Rate | PESQ-WB | Lyra at a similar rate |
|---|---|---|---|
| EnCodec + Vocos | 1.5k (2 codebooks) | 2.28 | Lyra 2.0k 1.96, 2.4k 2.17 |
| EnCodec + Vocos | 2.25k (3 codebooks) | 2.66 | Lyra 3.2k 2.50 |
| EnCodec + Vocos | 3.0k (4 codebooks) | 2.87 | Lyra 3.2k 2.50 |
| EnCodec + Vocos | 6.0k (8 codebooks) | 3.40 | Lyra 6k 3.17, 9.2k 3.47 |
| EnCodec's own decoder | 1.5k / 3.0k | 1.83 / 2.47 | |

PyTorch on one core: EnCodec encoder about 5%, Vocos decoder about 1% (Lyra through
OpenCV: 6% + 8.5%). Vocos as released looks 27 frames (360 ms) ahead; limiting
it to 3 frames (40 ms) cost nothing measurable (3.0k: 2.89 against 2.87), 0
frames cost about 0.25. The scripts that produced these numbers are in
`tools/reference/` (they read the Lyra test clips and the pip packages
`vocos` 0.1.0 and `encodec` 0.1.1, already installed with torch 2.7.1).

The goal is a library SDRangel can load exactly as it loads `lyracodec`, so
that Angel61 can adopt it as a new codec family.

## What the codec is

- **EnCodec 24 kHz** (Meta, `encodec_model_24khz`): causal SEANet encoder
  (convolutions with weight norm, ELU, strided downsampling by 320, a 2 layer
  LSTM), 75 frames/s (13.33 ms), residual vector quantiser of up to 32
  codebooks of 1024 entries (10 bits each, 128 dimensions). Each codebook adds
  750 bit/s; the codebooks are progressive, so a stream truncates to any lower
  codebook count, like Lyra's stages. `audio_normalize=False` for this model.
- **Vocos** (`charactr/vocos-encodec-24khz`, MIT): features = the sum of the
  EnCodec codebook embeddings of the codebooks present; backbone = conv embed
  (k 7) + 8 ConvNeXt blocks (dim 384, intermediate 1152, depthwise k 7,
  AdaLayerNorm conditioned on a bandwidth id 0..3 for 1.5/3/6/12 kbit/s);
  head = linear to 641 magnitudes + 641 phases, inverse STFT n_fft 1280,
  hop 320, "same" padding. Configs are in the Hugging Face cache
  (`%USERPROFILE%\.cache\huggingface\hub\models--charactr--vocos-*`); the
  EnCodec checkpoint is `%USERPROFILE%\.cache\torch\hub\checkpoints\encodec_24khz-d7cc33bc.th`.
- Measured once, on two clips: with 3 codebooks the 1.5k embedding (id 0) beat
  the 3k one (2.66 against 2.59). The rule mapping a codebook count to an
  embedding id is a decoder choice to settle by measurement.

## Requirements

1. **Standalone repository and CMake build** with no Python, PyTorch or ML
   runtime needed at run time. It builds on Windows x64 (MSVC 2022, Ninja), and
   must remain portable to Linux x64, Linux arm64 (Raspberry Pi 4/5), macOS and
   Android.
2. **A stable C ABI in a shared library** (`vocoscodec.dll` /
   `libvocoscodec.so` / `.dylib`) with an ABI version, following
   `lyra/lyra_codec/c_api.h`:
   version, last error, built model directory, encoder and decoder create,
   encode, decode, decode lost, reset and destroy. SDRangel loads it with
   `QLibrary` (see `srcejon_sdrangel_bugfix/sdrbase/dsp/dv/lyracodec.h`); if the
   library or its models are missing, only that load fails.
3. **Audio**: 24 kHz mono 16-bit PCM, one EnCodec frame = 320 samples (13.33 ms)
   per packet. The caller resamples (SDRangel runs at 48 kHz; 2:1).
4. **Packets**: a codebook count set at encoder creation (1..16 at least,
   covering 0.75k to 12k); the packet is the 10-bit indices, coarse codebook
   first, MSB first, so truncating the bit string to n x 10 bits gives the
   packet at n codebooks. Specify the layout normatively in the README. The
   decoder accepts any codebook count up to what it was built for.
5. **Streaming encoder**: frame by frame, the causal convolutions' and LSTM's
   state carried between calls, output identical (within float tolerance) to
   encoding the whole signal at once.
6. **Streaming decoder with bounded lookahead**: Vocos is not causal. Provide
   decoding with a configurable lookahead (default chosen by measurement; 3
   frames = 40 ms looked free) and enough history for the receptive field. Allow
   several packets per call, since Angel61 receives a radio frame of packets at
   once. Report the decoder's delay in samples. State what cost per frame the
   chosen scheme has.
7. **Lost packets**: `decode_lost` must advance time by one frame
   (Angel61's rule: lost audio keeps its time). Neither model has concealment:
   design one (e.g. repeat the last features and fade, then comfort noise or
   silence after a few frames, with a flag for comfort noise like Lyra's),
   and measure it with PESQ at 1, 5, 10 and 20% random and burst losses.
8. **Resets**: encoder reset at key down, decoder reset at a new transmission.
9. **Models as files** in a model directory, in a simple versioned binary
   format of our own (header, tensor table, float32 data), generated by a
   Python converter in `tools/` from the PyTorch checkpoints. Weight norm is
   folded at conversion. Record the SHA-256 of each model file (Angel61 names a
   codec family by its exact model set).
10. **CPU and memory**: measure encode and decode per frame on this PC, and
    estimate or measure a Raspberry Pi 4/5 (SDRangel's DSD work has a Pi-class
    CPU requirement). Report memory and model size.
11. **Tools**: WAV encode/decode command-line tools and a benchmark (off by
    default), as in the Lyra fork.

## Phases

0. **Licence gate (do first).** Confirm the licence of the EnCodec 24 kHz
   *weights* (the code is MIT; no weights licence was found on a first look at
   the GitHub README and the Hugging Face card). Check the repository's LICENSE
   files, README history and Meta's release notes. Vocos code and weights are
   MIT. SDRangel is GPL v3: a non-commercial weights licence (e.g. CC-BY-NC)
   would make this unusable, so stop and report if that is the answer.
1. **Reference and test vectors.** A Python reference (from `tools/reference/`)
   that runs frame by frame exactly as the C++ will, including the chosen
   lookahead and loss handling. Golden vectors: PCM in, indices out; indices
   in, PCM out; for regression tests of the C++ port.
2. **Inference engine.** Recommended: hand-written C++ with no ML runtime. The
   networks are only 1-D convolutions, ELU/GELU, LayerNorm, an LSTM, linear
   layers, a nearest-neighbour search and an inverse FFT, which keeps
   deployment trivial on the Pi and Android. Use SIMD where it pays (AVX2 and
   NEON, with a plain C++ fallback). ONNX Runtime is the fallback if a native
   port proves impractical; OpenCV dnn is not recommended (it misdecoded Lyra's
   quantised graphs and the Windows OpenCV package is a CUDA build). Decide
   early and write the reasons down.
3. **Encoder port + converter**, verified against the golden vectors: index
   agreement (expect near 100%, not exact: float differences can flip nearest
   neighbours, as with Lyra) and decoded-audio quality equal to the reference.
4. **Decoder port** (features, backbone, head, inverse STFT, streaming with
   lookahead), verified the same way.
5. **Loss concealment**, designed and measured (requirement 7).
6. **C ABI runtime, CMake install and package config**, tools, tests
   (ctest), README with the packet format, delay, and rates.
7. **Measurements and report** in `validation/README.md`: PESQ-WB (and STOI if
   useful) against Lyra at matching rates on a larger speech set than two
   clips (ask before downloading any dataset), CPU, memory, delay, the
   bandwidth-embedding rule, cross-platform index agreement.

## Integration notes for later (not this task)

- **Frame timing**: Angel61 NB/WB frames are 60 ms = 4.5 EnCodec frames. The
  HF profile under design uses 240 ms frames = 18 EnCodec frames. NB/WB would
  need a new frame layout (e.g. 120 ms pairs) to carry this codec; Angel61
  decides that, but the library must not assume any grouping.
- **Rates for Angel61**: HF 2.7 kHz channel tiers of about 1.5k, 2.25k and 3.0k; 6 kHz
  channel 2.25k/3.0k/6.0k; VHF equivalents of Lyra 3.2k/6k/9.2k would be 3.0k,
  6.0k and 9.0k (12 codebooks, which has no Vocos embedding of its own:
  measure).
- **Gateways** relay by truncating packets to fewer codebooks, as with Lyra.
- SDRangel side: a loader like `LyraCodec` in `sdrbase/dsp/dv`, and a new codec
  family in ANGEL61.md 3.1.

## Working rules

- The user (Jon Beniston) builds on Windows with MSVC 2022
  (`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`)
  and Ninja; Python 3.12 is at `C:\Applications\Python312` with torch, vocos,
  encodec, pesq, scipy and onnxruntime installed.
- Don't commit or push unless the user asks; ask before downloading datasets
  or large files.
- Measure rather than assume, and report numbers with what they were measured
  on.
