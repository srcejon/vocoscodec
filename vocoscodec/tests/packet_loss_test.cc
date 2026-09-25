// Lost packets: time advances, a long loss fades to comfort noise (or
// silence) that does not repeat, nothing clips, and speech resumes.
//
// Usage: vocoscodec_packet_loss_test <model directory> <golden.bin>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "vocoscodec/codec.h"
#include "vocoscodec/model_file.h"
#include "vocoscodec/tests/test_util.h"

using namespace vocoscodec;

namespace {

double Rms(const std::vector<std::int16_t>& x, std::size_t from, std::size_t to) {
  double sum = 0.0;
  for (std::size_t i = from; i < to; ++i) sum += double(x[i]) * x[i];
  return std::sqrt(sum / std::max<std::size_t>(1, to - from));
}

struct Run {
  std::vector<std::int16_t> audio;
  std::vector<bool> comfort_noise;  // after each frame
};

Run DecodeWithLosses(Decoder& decoder, const std::vector<std::vector<std::uint8_t>>& packets,
                     int lost_from, int lost_to) {
  Run run;
  std::string error;
  for (int f = 0; f < static_cast<int>(packets.size()); ++f) {
    const bool lost = f >= lost_from && f < lost_to;
    const bool ok = lost ? decoder.DecodeLostFrame(&run.audio, &error)
                         : decoder.Decode(packets[f].data(), packets[f].size(), &run.audio, &error);
    CHECK(ok);
    run.comfort_noise.push_back(decoder.is_comfort_noise());
  }
  decoder.Flush(&run.audio, &error);
  return run;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <model directory> <golden.bin>\n", argv[0]);
    return 2;
  }
  try {
    const std::filesystem::path models = argv[1];
    ModelFile golden(argv[2]);
    const int golden_frames = std::stoi(golden.Meta("frames"));
    const float* pcm = golden.Get("pcm", {static_cast<std::uint32_t>(golden_frames * 320)}).data;

    // 3.2 s of speech (the golden clip twice) at 4 codebooks.
    std::string error;
    auto encoder = Encoder::Create(models, 4, &error);
    CHECK(encoder != nullptr);
    if (!encoder) return test::Finish("vocoscodec_packet_loss_test");
    std::vector<std::vector<std::uint8_t>> packets;
    for (int f = 0; f < 2 * golden_frames; ++f) {
      std::int16_t samples[320];
      const float* frame = pcm + (f % golden_frames) * 320;
      for (int i = 0; i < 320; ++i) samples[i] = static_cast<std::int16_t>(frame[i] * 32768.0f);
      std::vector<std::uint8_t> packet;
      encoder->Encode(samples, 320, &packet, &error);
      packets.push_back(packet);
    }
    const int frames = static_cast<int>(packets.size());

    for (const bool comfort_noise : {true, false}) {
      DecoderOptions options;
      options.comfort_noise = comfort_noise;
      auto decoder = Decoder::Create(models, options, &error);
      CHECK(decoder != nullptr);
      if (!decoder) break;
      const int delay = decoder->delay_samples();
      const int lookahead = decoder->lookahead();
      const Run clean = DecodeWithLosses(*decoder, packets, 0, 0);
      // A one-second loss from frame 100.
      const int from = 100, to = 175;
      const Run lossy = DecodeWithLosses(*decoder, packets, from, to);
      CHECK(lossy.audio.size() == clean.audio.size());  // lost audio keeps its time

      // Before the loss the output is untouched, up to the lookahead: frames
      // from - lookahead .. from - 1 already saw the concealed frame.
      const std::size_t untouched = static_cast<std::size_t>(from - lookahead) * 320 + delay - 480;
      bool same = true;
      for (std::size_t i = 0; i < untouched; ++i) same = same && lossy.audio[i] == clean.audio[i];
      CHECK(same);

      // Deep in the loss: comfort noise (or silence), flagged, never clipped.
      const std::size_t deep_from = static_cast<std::size_t>(from + 20) * 320 + delay;
      const std::size_t deep_to = static_cast<std::size_t>(to) * 320 + delay - 320;
      CHECK(lossy.comfort_noise[from + 20]);
      CHECK(!lossy.comfort_noise[from - 1]);
      int clipped = 0;
      for (auto s : lossy.audio) clipped += s == 32767 || s == -32768;
      CHECK(clipped == 0);
      const double noise_rms = Rms(lossy.audio, deep_from, deep_to);
      const double speech_rms = Rms(clean.audio, deep_from, deep_to);
      std::printf("%s: RMS in a long loss %.1f (speech there %.1f)\n",
                  comfort_noise ? "comfort noise" : "silence", noise_rms, speech_rms);
      if (comfort_noise) {
        CHECK(noise_rms > 0.5 && noise_rms < 0.5 * speech_rms);
        // Not a repeating waveform: consecutive frames differ.
        int identical = 0;
        for (std::size_t f = deep_from; f + 640 <= deep_to; f += 320) {
          bool equal = true;
          for (int i = 0; i < 320; ++i) equal = equal && lossy.audio[f + i] == lossy.audio[f + 320 + i];
          identical += equal;
        }
        CHECK(identical == 0);
      } else {
        CHECK(noise_rms < 1.0);
      }

      // Speech resumes: the flag clears once received frames are synthesized
      // and the level returns within 100 ms.
      CHECK(!lossy.comfort_noise[to + lookahead + 1]);
      const std::size_t after_from = static_cast<std::size_t>(to + 8) * 320 + delay;
      const std::size_t after_to = after_from + 320 * 30;
      const double resumed = Rms(lossy.audio, after_from, after_to);
      const double reference = Rms(clean.audio, after_from, after_to);
      std::printf("  resumed RMS %.1f against %.1f without loss\n", resumed, reference);
      CHECK(resumed > 0.7 * reference && resumed < 1.3 * reference);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return test::Finish("vocoscodec_packet_loss_test");
}
