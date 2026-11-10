# vocoscodec

A speech codec made of the EnCodec 24 kHz encoder and the Vocos decoder, as a
standalone C++ library with a C ABI, for SDRangel's Angel61 digital
voice protocol. It needs no Python, PyTorch or ML runtime: the networks run
in hand-written C++ (AVX2/FMA/F16C or else SSE4.2 on x86-64, NEON on arm64, portable C++
elsewhere).

## Comparison with mbevocoder and Lyra

[mbevocoder](https://github.com/srcejon/mbevocoder) implements AMBE, AMBE+2 and
IMBE, the vocoders of D-STAR, DMR, P25 and other digital voice systems. They run
at an 8 kHz sample rate, so they carry speech up to 4 kHz.
[Lyra](https://github.com/google/lyra) is Google's neural speech codec. On a
held-out set of 32 16 kHz recordings from 16 speakers, every codec was scored
against the original speech: PESQ and ESTOI at an 8 kHz sample rate, and PESQ-WB
and WARP-Q at 16 kHz:

| Codec            | Voice bit/s | PESQ     | ESTOI     | PESQ-WB  | WARP-Q    |
|------------------|-------------|----------|-----------|----------|-----------|
| vocoscodec 1.5k  | 1500        | 2.46     | 0.757     | 1.88     | 0.324     |
| vocoscodec 2.25k | 2250        | **2.96** | 0.803     | **2.33** | 0.396     |
| AMBE (D-STAR)    | 2400        | **2.95** | **0.827** | 2.05     | 0.393     |
| AMBE+2 (DMR, P25p2, YSF DN, NXDN) | 2450 | 2.90 | 0.818 | 2.05 | **0.421** |
| vocoscodec 3.0k  | 3000        | **3.16** | **0.835** | **2.59** | **0.427** |
| Lyra 3.2k        | 3200        | 2.90     | 0.797     | 2.33     | 0.417     |
| IMBE (P25p1, YSF VW, NXDN) | 4400 | 3.03  | 0.833     | 1.99     | 0.400     |
| vocoscodec 4.5k  | 4500        | **3.36** | **0.863** | **2.82** | **0.462** |
| Lyra 6k          | 6000        | 3.41     | 0.853     | 2.93     | 0.481     |
| vocoscodec 6k    | 6000        | **3.56** | **0.881** | **3.08** | **0.492** |
| vocoscodec 9k    | 9000        | **3.68** | **0.898** | **3.23** | **0.521** |
| Lyra 9.2k        | 9200        | 3.59     | 0.874     | 3.17     | 0.500     |
| vocoscodec 12k   | 12000       | 3.78     | 0.909     | 3.39     | 0.542     |

Higher is better on all four measures:

- **PESQ** (ITU-T P.862) predicts how listeners would rate the decoded speech
  against the original, from 1 (bad) to 4.5 (excellent). This is its narrowband
  form, which covers speech up to 4 kHz.
- **PESQ-WB** (ITU-T P.862.2) is the wideband form, covering speech up to 8 kHz,
  so it marks down a codec that stops at 4 kHz. Its scores can't be compared
  directly with narrowband PESQ.
- **ESTOI** (extended short-time objective intelligibility) predicts how
  intelligible the speech is, from 0 to 1, by measuring how much of the
  original's short-time spectral structure survives. It only analyses up to
  about 4.3 kHz, whatever the sample rate, so it gives almost no credit for
  bandwidth above that.
- **[WARP-Q](https://github.com/wjassim/WARP-Q)** predicts quality from how
  closely the decoded speech's MFCCs can be time-aligned to the original's. It
  was designed for low-rate neural codecs, whose output can sound right without
  matching the waveform. Scores here are its normalised form, in which the
  original against itself scores 0.83, not 1. It analyses up to 5 kHz by
  default, so it gives no credit for bandwidth above that, but marks down a
  codec that stops at 4 kHz: the original limited to 4 kHz scores only 0.63,
  which counts against the MBE codecs. With both signals limited to 4 kHz, AMBE
  scores 0.438 and AMBE+2 0.466, against 0.422 for vocoscodec 2.25k and 0.453
  for 3.0k.

What the results show:

- **At about 2.4 kbit/s, MBE holds its own.** vocoscodec 2.25k ties AMBE and
  AMBE+2 on PESQ, and is significantly behind them on ESTOI, which tracks
  intelligibility. Its advantage there is bandwidth: PESQ-WB 2.33 against 2.05,
  because MBE stops at 4 kHz.
- **At 3.0 kbit/s, vocoscodec leads.** It beats AMBE+2 at 2.45 kbit/s and IMBE
  at 4.4 kbit/s on PESQ, by 0.26 and 0.13, and matches or beats them on ESTOI.
- **At 4.5 kbit/s, vocoscodec is clearly ahead of IMBE** on PESQ and ESTOI, by
  0.33 and 0.030.
- **Against Lyra, vocoscodec reaches the same quality at a lower rate.** Its
  2.25k ties Lyra 3.2k, and from 3.0k it is ahead at every rate. It leads Lyra
  3.2k by 0.26 on PESQ at 3.0k, Lyra 6k by 0.15 at 6k, and Lyra 9.2k by 0.10 at
  9k, and it wins on ESTOI on all 32 files at each of these rates.
- **Under packet loss,** vocoscodec 3.0k stays ahead of AMBE+2 2450 at up to 10%
  random loss and at every level of radio-frame loss. At 20% bursts they tie. It
  also leads Lyra 3.2k, except at 10% and 20% bursts, where Lyra is ahead by
  0.02 and 0.06.
- **MBE's voice rates exclude its FEC:** AMBE and AMBE+2 are 3600 bit/s on air,
  IMBE 7200. vocoscodec and Lyra carry no FEC of their own, so the protocol's
  FEC adds to their rates in the same way. Bit errors were not compared.
- **MBE is far cheaper.** It needs no models (vocoscodec holds 41 MB in memory)
  and about 1.6% of a core to encode and 0.05% to decode, against about 7% and
  3%. Its delay is 85 ms (91 ms for AMBE at D-STAR), against 73 ms.

`validation/README.md` has the measurements: quality against Lyra and
mbevocoder, losses, agreement, CPU, memory and delay. `PLAN.md` records why the
codec was built and the decisions made.

## Licences and provenance

- **The EnCodec code** (Meta, `facebookresearch/encodec`) has been MIT since
  April 2023.
- **The EnCodec 24 kHz weights** (`encodec_24khz-d7cc33bc.th`) were published
  in October 2022, when the whole repository was CC-BY-NC 4.0.
  - The 2023 relicensing says "the code". Meta has not answered the public
    questions about whether the weights are MIT too (encodec issues #1, #49,
    #69).
- **Vocos** (`charactr/vocos-encodec-24khz`), code and weights: MIT. Its
  embedding table is EnCodec's first 16 codebooks, bit for bit.
- **The test and validation speech** is not stored here:
  - LibriSpeech (CC BY 4.0; Panayotov et al., ICASSP 2015);
  - CMU ARCTIC (Copyright Carnegie Mellon University).
  - `testdata/golden.bin` contains 1.6 s of LibriSpeech test-clean 1089-134686-0000.
- **This library** (its code, build files, tools and documentation): MIT, see
  `LICENSE`. The MIT licence does not extend to the model files in `model/`:
  they are converted from the EnCodec and Vocos weights and carry those
  weights' terms, as above.

## Building

Windows, MSVC 2022 and Ninja:

```bat
build.bat            :: Release build in build\Release, with tests and tools
test.bat             :: ctest
benchmark.bat --group 9
```

Elsewhere, plain CMake (3.20+, C++17):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVOCOSCODEC_BUILD_TESTS=ON
cmake --build build && ctest --test-dir build
```

| Option | Default | |
|---|---|---|
| `VOCOSCODEC_BUILD_RUNTIME` | ON | the `vocoscodec` shared library (C ABI) |
| `VOCOSCODEC_BUILD_TESTS` | OFF | ctest suite |
| `VOCOSCODEC_BUILD_TOOLS` | OFF | `vocoscodec_encode`, `vocoscodec_decode`, `vocoscodec_benchmark` |
| `VOCOSCODEC_HALF_WEIGHTS` | ON | hold the per-frame layers' weights in half precision (see below) |
| `VOCOSCODEC_RUNTIME_MODEL_DIR` | `<source>/model` | reported by `vocoscodec_built_model_directory()` |

`cmake --install` installs:
- the static library `vocoscodec_codec`, the runtime `vocoscodec.dll`,
  `libvocoscodec.so` or `.dylib`, and the headers `codec.h` and `c_api.h`;
- the three model files in `share/vocoscodec/model`;
- a CMake package, `find_package(Vocoscodec)` with targets `vocoscodec::codec`
  and `vocoscodec::runtime` and the variable `Vocoscodec_MODEL_DIR`.

## Models

`tools/convert_models.py` makes the model directory from the PyTorch
checkpoints. It needs only torch; weight norm and layer scale are folded. The
checkpoints come from `pip install encodec vocos` followed by
`EncodecModel.encodec_model_24khz()` and
`Vocos.from_pretrained("charactr/vocos-encodec-24khz")`, which download them.
The model files are in `model/` (70 MB); rerunning the converter reproduces
them bit for bit (`model/SHA256SUMS`).

| File | Contents | Size | SHA-256 |
|---|---|---|---|
| `encodec_encoder.bin` | SEANet encoder and 2-layer LSTM | 29.7 MB | `7960bf7cf1af0b002bd20f6712b9bff0f2a57bd1bbd4b4d3bc758987f784e4ae` |
| `encodec_codebooks.bin` | residual VQ codebooks 1..16, 1024 x 128 each | 8.4 MB | `4adfcc7a3b5670310dc9d2c227bfa6773b4b6de36bf4709507b0a1c82926b57d` |
| `vocos_decoder.bin` | Vocos backbone and ISTFT head | 31.9 MB | `b373ace31e114cfb7f95b2d96da374c987fbfb9168a13905b3a58dbd99556e5b` |

The conversion is deterministic, and `model/SHA256SUMS` lists the hashes.
An Angel61 codec family is named by exactly this set.

The model file format, version 1: all integers little-endian.

| Offset | Type | Field |
|---|---|---|
| 0 | char[8] | magic `VOCOSCM\0` |
| 8 | u32 | format version, 1 |
| 12 | u32 | tensor count |
| 16 | u32 | metadata length in bytes |
| 20 | u32 | reserved, 0 |
| 24 | | metadata: UTF-8, one `key=value` per line |
| | | tensor table: per tensor, u16 name length, name, u8 dtype (0 = float32), u8 rank, u32 dims[rank], u64 offset of its data |
| | | data: each tensor row-major float32, on a 64-byte boundary |

With `VOCOSCODEC_HALF_WEIGHTS`, the weights read once per frame are rounded
to IEEE half precision when loaded. These are the encoder's LSTM, last
downsampling and output convolutions, and the decoder's linear layers. That
halves their memory and memory traffic, which is what bounds their speed.
It changed 0.6% of encoder indices and no decoder PESQ measurably.

This applies only where the kernels convert half precision in hardware (AVX2
with F16C, NEON). With the SSE4.2 or portable kernels, the weights stay
float: converting in software cost more than the traffic it saved.

## Audio and frames

- **Audio:** 24 kHz mono, 16-bit PCM (or float in [-1, 1) through the C++ API).
- **Frames:** 320 samples, 13.33 ms, 75 frames/s. The caller resamples;
  SDRangel's 48 kHz is 2:1.
- **Encoder:**
  - Streaming: its convolution histories and LSTM state carry across calls.
  - A reset (key down) sets that history to silence.
  - Its output equals encoding the whole signal at once with zero padding
    before the start. Upstream EnCodec pads the start by reflection instead,
    which no streaming encoder can do. That changes the first few hundred
    milliseconds of indices, not the audio quality.

## Packets

This section is normative.

- A packet carries one frame as n codebook indices, 1 <= n <= 16, each an
  unsigned 10-bit number.
- The indices are written coarse codebook first, each most significant bit
  first, with no gaps. The last byte is padded with zero bits.
- A packet of n codebooks is ceil(10 n / 8) bytes, and its size alone gives n.
- The first 10 m bits of an n-codebook packet are the m-codebook packet of the
  same frame. A relay lowers the rate by keeping ceil(10 m / 8) bytes and
  clearing the bits after the first 10 m (`vocoscodec_packet_truncate`).
- The decoder accepts any n in each packet; streams may change rate from
  packet to packet.

| Codebooks | Payload bit/s | Packet bytes | Bytes/s | Vocos bandwidth embedding |
|---:|---:|---:|---:|---|
| 1 | 750 | 2 | 150 | 1.5k |
| 2 | 1500 | 3 | 225 | 1.5k |
| 3 | 2250 | 4 | 300 | 1.5k |
| 4 | 3000 | 5 | 375 | 3k |
| 6 | 4500 | 8 | 600 | 6k |
| 8 | 6000 | 10 | 750 | 6k |
| 12 | 9000 | 15 | 1125 | 12k |
| 16 | 12000 | 20 | 1500 | 12k |

Only these sizes are packets: 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17,
18, 19 and 20 bytes. Protocols that pack frames bit-contiguously carry the
10 n payload bits and may drop the padding.

## Decoder delay and lookahead

Vocos is not causal: each of its nine convolutions of width 7 looks 3 frames
ahead, 27 in all. The decoder bounds this with a lookahead of L frames (0..27,
default 3). Each frame is decoded as the whole-signal decoder would decode
the packets received so far, as if the signal ended there. At L = 3 this
costs 0.05 PESQ against unbounded Vocos. Masking the future taps in every
layer instead costs about 0.7.

- **Delay:** 480 + 320 L samples at 24 kHz. At the default this is 1440
  samples (60 ms), and `vocoscodec_decoder_delay_samples()` returns it.
  Output sample n corresponds to input sample n - delay.
  - The 480 samples come from the inverse STFT (n_fft 1280, hop 320).
  - The encoder adds one frame of buffering (13.3 ms), so the end-to-end
    algorithmic delay is 800 + 320 L samples: 73 ms at L = 3.
- **Packets per call:**
  - One packet per call gives every frame exactly L frames of lookahead.
  - Several packets per call (an Angel61 radio frame) let frames early in
    the group look up to its end. That only helps quality, and it is much
    cheaper: the decoder recomputes values near the newest frame once per
    call, not once per frame.
- **End of transmission:** `vocoscodec_decoder_flush()` returns the last
  `delay` samples (the audio of the last frames), then resets.
  `vocoscodec_decoder_reset()` starts a new transmission.

## Lost packets

`vocoscodec_decode_lost()`, or a packet of size 0 in
`vocoscodec_decode_packets()`, conceals one missing frame and advances time
by exactly one frame.

- **Short losses** are decoded from the last good frame's features. The
  lookahead gives Vocos the real frames after an isolated loss.
- **Longer losses:** after R frames the spectrum crossfades, over F frames,
  to comfort noise. Its magnitudes are the noise floor of the last second
  received, and its phases are random. With `VOCOSCODEC_DECODER_SILENCE`
  the crossfade goes to silence.
- **Recovery:** the decoder recovers by itself when packets resume.
- **Status:** `vocoscodec_decoder_is_comfort_noise()` reports the fully
  faded state, as Lyra's does.

R = 3 and F = 3 (40 ms each) were chosen on the dev set.
`validation/README.md` has the measurements against Lyra at 1–20% random,
burst and radio-frame losses.

## C API

`vocoscodec/c_api.h`, ABI version 1, follows `lyra_codec/c_api.h`. A program
can load the runtime at run time (SDRangel uses QLibrary). If the library or
its models are missing, only that load fails.

```c
int vocoscodec_abi_version(void);                 /* VOCOSCODEC_ABI_VERSION, 1 */
const char* vocoscodec_version(void);
const char* vocoscodec_built_model_directory(void);
int vocoscodec_has_models(const char* dir);
const char* vocoscodec_last_error(void);          /* per thread */
int vocoscodec_packet_bytes(int codebooks);
int vocoscodec_packet_codebooks(int packet_bytes);
int vocoscodec_packet_truncate(uint8_t* packet, int bytes, int codebooks);

vocoscodec_encoder* vocoscodec_encoder_create(const char* dir, int codebooks);
int  vocoscodec_encode(vocoscodec_encoder*, const int16_t pcm[320], uint8_t* packet, int capacity);
void vocoscodec_encoder_reset(vocoscodec_encoder*);          /* key down */
void vocoscodec_encoder_destroy(vocoscodec_encoder*);

vocoscodec_decoder* vocoscodec_decoder_create(const char* dir, int lookahead /* -1: 3 */, int flags);
int  vocoscodec_decoder_delay_samples(vocoscodec_decoder*);
int  vocoscodec_decode(vocoscodec_decoder*, const uint8_t* packet, int bytes, int16_t* pcm, int capacity);
int  vocoscodec_decode_packets(vocoscodec_decoder*, const uint8_t* const* packets, const int* bytes,
                               int count, int16_t* pcm, int capacity);   /* bytes[i] == 0: lost */
int  vocoscodec_decode_lost(vocoscodec_decoder*, int16_t* pcm, int capacity);
int  vocoscodec_decoder_flush(vocoscodec_decoder*, int16_t* pcm, int capacity);
int  vocoscodec_decoder_is_comfort_noise(vocoscodec_decoder*);
void vocoscodec_decoder_reset(vocoscodec_decoder*);          /* new transmission */
void vocoscodec_decoder_destroy(vocoscodec_decoder*);
```

- **Returns:** functions return a sample or byte count, or 0 on failure with
  `vocoscodec_last_error()` saying why.
- **Handles:** a null handle is harmless.
- **Model sharing:** models are loaded once per process and shared by all
  encoders and decoders that use the same files.
- **Threads:** each handle belongs to one thread at a time.

## Tools

With `VOCOSCODEC_BUILD_TOOLS`:

- `vocoscodec_encode <models> in.wav out.packets [codebooks]` takes 24 kHz
  mono input.
- `vocoscodec_decode <models> in.packets out.wav` takes these options:
  `--lookahead`, `--codebooks` (truncate), `--group`, `--silence` and
  `--keep-delay`.
  - The packet file is a sequence of [size byte][packet], where size 0
    marks a lost packet.
  - By default the output is trimmed by the codec delay, so it lines up
    with the input.
- `vocoscodec_benchmark <models>` reports encode and decode time per frame,
  model load time and memory. Its options are `--codebooks`, `--group`,
  `--lookahead`, `--wav` and `--kernels avx2|sse4.2|neon|generic`.
  `benchmark.bat` pins it to one
  performance core.

## Tests and reference

`ctest` runs these tests:

| Test | What it checks |
|---|---|
| `vocoscodec_kernels_test` | SIMD kernels against the portable ones; FFT against a DFT |
| `vocoscodec_golden_test` | encoder, packets and decoder against the Python reference's golden vectors (`testdata/golden.bin`) |
| `vocoscodec_golden_generic_test` | the same with the portable kernels |
| `vocoscodec_golden_sse42_test` | the same with the SSE4.2 kernels (x86-64) |
| `vocoscodec_codec_test` | the C++ API: errors, grouping, truncation, reset, invalid packets, rate changes |
| `vocoscodec_packet_loss_test` | a one-second loss: time kept, comfort noise that does not repeat, no clipping, recovery |
| `vocoscodec_c_api_test` | the C ABI |
| `vocoscodec_runtime_load_test` | loads the shared library at run time and resolves every symbol |

`tools/reference/` holds the Python side:

| Script | What it does |
|---|---|
| `vocoscodec_ref.py` | the frame-by-frame reference: reads the model files and computes as the C++ does |
| `check_reference.py` | checks the reference against the `encodec` and `vocos` packages |
| `make_golden.py` | writes the golden vectors |
| `decoder_lookahead.py` | the lookahead study |
| `fp16_weights.py` | the half-precision check |

`tools/evaluate.py` produces the numbers in `validation/README.md`.
