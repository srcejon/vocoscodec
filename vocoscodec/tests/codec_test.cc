// The C++ codec API: creation errors, encoding and decoding against the
// golden vectors, grouping, truncation, resets and invalid packets.
//
// Usage: vocoscodec_codec_test <model directory> <golden.bin>

#include <cstdio>
#include <string>
#include <vector>

#include "vocoscodec/codec.h"
#include "vocoscodec/model_file.h"
#include "vocoscodec/packet.h"
#include "vocoscodec/tests/test_util.h"

using namespace vocoscodec;

namespace {

using Packets = std::vector<std::vector<std::uint8_t>>;

Packets GoldenPackets(ModelFile& golden, int n, int frames) {
  const int bytes = PacketBytes(n);
  const float* data = golden.Get("packets_n" + std::to_string(n),
                                 {static_cast<std::uint32_t>(frames),
                                  static_cast<std::uint32_t>(bytes)}).data;
  Packets packets(frames);
  for (int f = 0; f < frames; ++f)
    for (int i = 0; i < bytes; ++i) packets[f].push_back(static_cast<std::uint8_t>(data[f * bytes + i]));
  return packets;
}

// Decodes packets `group` per call (an empty packet is lost), then flushes.
std::vector<std::int16_t> DecodeAll(Decoder& decoder, const Packets& packets, int group = 1) {
  std::vector<std::int16_t> out;
  std::string error;
  for (std::size_t f = 0; f < packets.size(); f += group) {
    std::vector<const std::uint8_t*> pointers;
    std::vector<std::size_t> sizes;
    for (std::size_t j = f; j < std::min(packets.size(), f + group); ++j) {
      pointers.push_back(packets[j].data());
      sizes.push_back(packets[j].size());
    }
    CHECK(decoder.Decode(pointers.data(), sizes.data(), static_cast<int>(sizes.size()), &out,
                         &error));
  }
  CHECK(decoder.Flush(&out, &error));
  return out;
}

void TestCreation(const std::filesystem::path& models) {
  std::string error;
  CHECK(HasModels(models));
  CHECK(!HasModels(models / "nonexistent"));
  CHECK(Encoder::Create(models, 0, &error) == nullptr && !error.empty());
  CHECK(Encoder::Create(models, 17, &error) == nullptr);
  CHECK(Encoder::Create(models / "nonexistent", 4, &error) == nullptr);
  CHECK(error.find("not found") != std::string::npos);
  DecoderOptions options;
  options.lookahead = 28;
  CHECK(Decoder::Create(models, options, &error) == nullptr);
  options.lookahead = 3;
  auto decoder = Decoder::Create(models, options, &error);
  CHECK(decoder && decoder->delay_samples() == 480 + 3 * 320);
}

void TestEncoder(const std::filesystem::path& models, ModelFile& golden, int frames) {
  const float* pcm = golden.Get("pcm", {static_cast<std::uint32_t>(frames * 320)}).data;
  const Packets expected = GoldenPackets(golden, 16, frames);
  std::string error;
  auto encoder = Encoder::Create(models, 16, &error);
  CHECK(encoder != nullptr);
  if (!encoder) return;
  Packets first;
  int equal = 0;
  for (int pass = 0; pass < 2; ++pass) {
    for (int f = 0; f < frames; ++f) {
      std::int16_t samples[320];
      for (int i = 0; i < 320; ++i) samples[i] = static_cast<std::int16_t>(pcm[f * 320 + i] * 32768.0f);
      std::vector<std::uint8_t> packet;
      CHECK(encoder->Encode(samples, 320, &packet, &error));
      CHECK(packet.size() == 20);
      if (pass == 0) {
        first.push_back(packet);
        int a[16], b[16];
        Unpack(packet.data(), 20, a);
        Unpack(expected[f].data(), 20, b);
        for (int q = 0; q < 16; ++q) equal += a[q] == b[q];
      } else {
        CHECK(packet == first[f]);  // after Reset the same input gives the same packets
      }
    }
    encoder->Reset();
  }
  std::printf("codec encoder: %.2f%% of indices equal the reference\n",
              100.0 * equal / (16.0 * frames));
  CHECK(equal > 0.98 * 16 * frames);
  std::vector<std::uint8_t> packet;
  std::int16_t short_frame[100] = {};
  CHECK(!encoder->Encode(short_frame, 100, &packet, &error));
}

void TestDecoder(const std::filesystem::path& models, ModelFile& golden, int frames) {
  std::string error;
  DecoderOptions options;
  auto decoder = Decoder::Create(models, options, &error);
  CHECK(decoder != nullptr);
  if (!decoder) return;
  const int delay = decoder->delay_samples();
  const int total = frames * 320 + delay;
  for (int n : {1, 2, 4, 8, 16}) {
    const Packets packets = GoldenPackets(golden, n, frames);
    const float* expected = golden.Get("decoded_n" + std::to_string(n) + "_l3",
                                       {static_cast<std::uint32_t>(total)}).data;
    const auto out = DecodeAll(*decoder, packets);
    CHECK(static_cast<int>(out.size()) == total);
    std::vector<float> as_float(out.begin(), out.end());
    std::vector<float> reference(total);
    for (int i = 0; i < total; ++i) reference[i] = expected[i] * 32768.0f;
    const double snr = test::SnrDb(reference.data(), as_float.data(), total);
    const float diff = test::MaxAbsDiff(reference.data(), as_float.data(), total);
    std::printf("codec decoder %2d codebooks: SNR against the reference %.1f dB, max diff %.1f LSB\n",
                n, snr, diff);
    CHECK(snr > 60.0);
    CHECK(diff <= 8.0f);

    // A decoder reused after Flush gives the same again.
    CHECK(DecodeAll(*decoder, packets) == out);

    if (n == 4) {
      // Nine packets per call, against the reference doing the same.
      const float* grouped = golden.Get("decoded_n4_l3_g9",
                                        {static_cast<std::uint32_t>(total)}).data;
      const auto out9 = DecodeAll(*decoder, packets, 9);
      std::vector<float> out9_float(out9.begin(), out9.end()), grouped_pcm(total);
      for (int i = 0; i < total; ++i) grouped_pcm[i] = grouped[i] * 32768.0f;
      CHECK(test::SnrDb(grouped_pcm.data(), out9_float.data(), total) > 60.0);
      CHECK(DecodeAll(*decoder, packets, 120).size() == out.size());

      // Packets of 16 codebooks truncated to 4 are the packets of 4.
      Packets truncated = GoldenPackets(golden, 16, frames);
      for (auto& p : truncated) p.resize(Truncate(p.data(), 20, 4));
      CHECK(DecodeAll(*decoder, truncated) == out);

      // Reset mid-stream gives the output of a fresh decoder.
      std::vector<std::int16_t> ignored;
      for (int f = 0; f < 50; ++f) decoder->Decode(packets[f].data(), packets[f].size(), &ignored, &error);
      decoder->Reset();
      CHECK(DecodeAll(*decoder, packets) == out);

      // An invalid packet is refused and changes nothing.
      std::vector<std::int16_t> with_bad;
      for (int f = 0; f < frames; ++f) {
        if (f == 60) {
          const std::uint8_t bad[6] = {};
          CHECK(!decoder->Decode(bad, 6, &with_bad, &error));
          CHECK(!error.empty());
        }
        CHECK(decoder->Decode(packets[f].data(), packets[f].size(), &with_bad, &error));
      }
      CHECK(decoder->Flush(&with_bad, &error));
      CHECK(with_bad == out);
    }
  }

  // Lost packets, concealed as the reference conceals them.
  {
    const std::vector<float> lost_list = [&] {
      const auto& t = golden.Get("lost", {43});
      return std::vector<float>(t.data, t.data + t.size);
    }();
    Packets lossy = GoldenPackets(golden, 4, frames);
    for (float f : lost_list) lossy[static_cast<int>(f)].clear();
    const float* expected = golden.Get("decoded_n4_l3_loss",
                                       {static_cast<std::uint32_t>(total)}).data;
    const auto out = DecodeAll(*decoder, lossy);
    std::vector<float> as_float(out.begin(), out.end()), reference(total);
    for (int i = 0; i < total; ++i) reference[i] = expected[i] * 32768.0f;
    const double snr = test::SnrDb(reference.data(), as_float.data(), total);
    std::printf("codec decoder with %zu lost packets: SNR against the reference %.1f dB\n",
                lost_list.size(), snr);
    CHECK(snr > 60.0);
  }

  // A stream whose codebook count changes every frame (a relay changing rate).
  Packets mixed = GoldenPackets(golden, 16, frames);
  for (int f = 0; f < frames; ++f) mixed[f].resize(Truncate(mixed[f].data(), 20, 1 + f % 16));
  const auto out = DecodeAll(*decoder, mixed);
  CHECK(static_cast<int>(out.size()) == total);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <model directory> <golden.bin>\n", argv[0]);
    return 2;
  }
  try {
    ModelFile golden(argv[2]);
    const int frames = std::stoi(golden.Meta("frames"));
    TestCreation(argv[1]);
    TestEncoder(argv[1], golden, frames);
    TestDecoder(argv[1], golden, frames);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return test::Finish("vocoscodec_codec_test");
}
