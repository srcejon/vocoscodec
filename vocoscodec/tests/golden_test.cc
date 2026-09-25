// The C++ encoder, packets and decoder against the Python reference's golden
// vectors (tools/reference/make_golden.py).
//
// Usage: vocoscodec_golden_test <model directory> <golden.bin> [--kernels NAME]
// --kernels checks named kernels (avx2, sse4.2, neon, generic) instead of the
// best for this CPU; other than the best, only the 4-codebook decodes run.
// --generic is short for --kernels generic.

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "vocoscodec/decoder.h"
#include "vocoscodec/encoder.h"
#include "vocoscodec/kernels.h"
#include "vocoscodec/model_file.h"
#include "vocoscodec/packet.h"
#include "vocoscodec/tests/test_util.h"

using namespace vocoscodec;

namespace {

bool generic = false;

void TestEncoder(ModelFile& golden, const std::shared_ptr<const EncoderModel>& model,
                 const std::shared_ptr<const Codebooks>& books, int frames) {
  const float* pcm = golden.Get("pcm", {static_cast<std::uint32_t>(frames * kFrameSamples)}).data;
  const float* latent = golden.Get("latent", {static_cast<std::uint32_t>(frames), kLatentDim}).data;
  const float* indices = golden.Get("indices", {static_cast<std::uint32_t>(frames), 16}).data;

  StreamingEncoder encoder(model, books, 16);
  float worst = 0.0f, peak = 0.0f;
  int equal[16] = {}, equal_given_latent = 0;
  for (int f = 0; f < frames; ++f) {
    float z[kLatentDim];
    int ids[16], from_golden_latent[16];
    encoder.Embed(pcm + f * kFrameSamples, z);
    encoder.Quantize(z, ids);
    encoder.Quantize(latent + f * kLatentDim, from_golden_latent);
    worst = std::fmax(worst, test::MaxAbsDiff(z, latent + f * kLatentDim, kLatentDim));
    for (int i = 0; i < kLatentDim; ++i) peak = std::fmax(peak, std::fabs(latent[f * kLatentDim + i]));
    for (int q = 0; q < 16; ++q) {
      equal[q] += ids[q] == static_cast<int>(indices[f * 16 + q]);
      equal_given_latent += from_golden_latent[q] == static_cast<int>(indices[f * 16 + q]);
    }
  }
  int all = 0;
  std::ostringstream per_book;
  for (int q = 0; q < 16; ++q) {
    all += equal[q];
    if (q == 0 || q == 1 || q == 3 || q == 7 || q == 15) per_book << " " << q + 1 << ":" << equal[q];
  }
  const double agreement = 100.0 * all / (16.0 * frames);
  std::printf("encoder (%s): latent max |diff| %.2e (peak %.1f); indices equal %.2f%% "
              "(per codebook of %d:%s); quantiser alone %.2f%%\n",
              kernels::InstructionSet(), worst, peak, agreement, frames,
              per_book.str().c_str(), 100.0 * equal_given_latent / (16.0 * frames));
  CHECK(worst < 2e-3f);
  CHECK(agreement > 98.0);
  CHECK(equal[0] == frames);
  CHECK(equal_given_latent == 16 * frames);
}

void TestPackets(ModelFile& golden, int frames) {
  const float* indices = golden.Get("indices", {static_cast<std::uint32_t>(frames), 16}).data;
  for (int n = 1; n <= 16; ++n) {
    const int bytes = PacketBytes(n);
    CHECK(PacketCodebooks(bytes) == n);
    const float* expected = golden.Get("packets_n" + std::to_string(n),
                                       {static_cast<std::uint32_t>(frames),
                                        static_cast<std::uint32_t>(bytes)}).data;
    for (int f = 0; f < frames; ++f) {
      int ids[16], back[16];
      for (int q = 0; q < 16; ++q) ids[q] = static_cast<int>(indices[f * 16 + q]);
      std::uint8_t packet[20], full[20];
      Pack(ids, n, packet);
      for (int i = 0; i < bytes; ++i) {
        if (packet[i] != static_cast<int>(expected[f * bytes + i])) {
          CHECK_MSG(false, "packet byte differs at n=" + std::to_string(n));
          return;
        }
      }
      CHECK(Unpack(packet, bytes, back) == n);
      for (int q = 0; q < n; ++q) CHECK(back[q] == ids[q]);
      Pack(ids, 16, full);
      CHECK(Truncate(full, PacketBytes(16), n) == bytes);
      for (int i = 0; i < bytes; ++i) CHECK(full[i] == packet[i]);
    }
  }
  int invalid = 0;
  for (int bytes = 0; bytes <= 21; ++bytes) invalid += PacketCodebooks(bytes) == 0;
  CHECK(invalid == 6);  // 0, 1, 6, 11, 16 and 21 bytes
}

void TestDecoder(ModelFile& golden, const std::shared_ptr<const DecoderModel>& model,
                 const Codebooks& books, int frames) {
  const float* indices = golden.Get("indices", {static_cast<std::uint32_t>(frames), 16}).data;
  std::istringstream decodes(golden.Meta("decodes"));
  for (std::string item; std::getline(decodes, item, ';');) {
    const int n = std::stoi(item.substr(0, item.find(',')));
    const int lookahead = std::stoi(item.substr(item.find(',') + 1));
    if (generic && n != 4) continue;
    const std::string name = "decoded_n" + std::to_string(n) + "_l" + std::to_string(lookahead);
    const int delay = kIstftPad + kFrameSamples * lookahead;
    const int total = frames * kFrameSamples + delay;
    const float* expected = golden.Get(name, {static_cast<std::uint32_t>(total)}).data;
    const float* ids_golden = golden.Get("ids_n" + std::to_string(n),
                                         {static_cast<std::uint32_t>(frames)}).data;

    std::vector<float> features(static_cast<std::size_t>(frames) * kLatentDim, 0.0f);
    std::vector<int> ids(frames);
    for (int f = 0; f < frames; ++f) {
      for (int q = 0; q < n; ++q) {
        const float* e = books.Entry(q, static_cast<int>(indices[f * 16 + q]));
        for (int i = 0; i < kLatentDim; ++i) features[f * kLatentDim + i] += e[i];
      }
      ids[f] = static_cast<int>(ids_golden[f]);
    }

    // One frame per call gives every frame exactly the lookahead.
    StreamingDecoder decoder(model, lookahead);
    CHECK(decoder.delay() == delay);
    std::vector<float> out(total);
    for (int f = 0; f < frames; ++f) {
      decoder.Run(features.data() + f * kLatentDim, ids.data() + f, 1,
                  out.data() + f * kFrameSamples);
    }
    const long long blocks = decoder.block_evaluations();
    decoder.Flush(out.data() + frames * kFrameSamples);
    const float diff = test::MaxAbsDiff(out.data(), expected, total);
    const double snr = test::SnrDb(expected, out.data(), total);
    std::printf("decoder %2d codebooks, lookahead %2d: max |diff| %.2e, SNR %.1f dB, "
                "%.1f blocks/frame one frame per call\n",
                n, lookahead, diff, snr, static_cast<double>(blocks) / frames);
    CHECK_MSG(diff < 1e-3f, name + " differs from the reference");

    if (n == 4 && lookahead == 3) {
      // Nine frames per call: frames early in a group see up to its end.
      const float* grouped = golden.Get("decoded_n4_l3_g9", {static_cast<std::uint32_t>(total)}).data;
      StreamingDecoder batch(model, lookahead);
      std::vector<float> out9(total);
      for (int f = 0; f < frames; f += 9) {
        const int count = std::min(9, frames - f);
        batch.Run(features.data() + f * kLatentDim, ids.data() + f, count,
                  out9.data() + f * kFrameSamples);
      }
      const long long blocks9 = batch.block_evaluations();
      batch.Flush(out9.data() + frames * kFrameSamples);
      const float diff9 = test::MaxAbsDiff(out9.data(), grouped, total);
      std::printf("decoder  4 codebooks, lookahead  3, nine per call: max |diff| %.2e, "
                  "%.1f blocks/frame; %.1f dB from one per call\n",
                  diff9, static_cast<double>(blocks9) / frames,
                  test::SnrDb(out.data(), out9.data(), total));
      CHECK_MSG(diff9 < 1e-3f, "decoded_n4_l3_g9 differs from the reference");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string chosen;
  if (argc == 4 && std::string(argv[3]) == "--generic") chosen = "generic";
  if (argc == 5 && std::string(argv[3]) == "--kernels") chosen = argv[4];
  if (argc < 3 || argc > 5 || (argc > 3 && chosen.empty())) {
    std::fprintf(stderr, "usage: %s <model directory> <golden.bin> [--kernels NAME]\n",
                 argv[0]);
    return 2;
  }
  if (!chosen.empty()) {
    if (!kernels::Select(chosen)) {
      std::printf("kernels %s are not available here: skipped\n", chosen.c_str());
      return 0;
    }
    generic = true;  // a partial run
  }
  try {
    const std::filesystem::path models = argv[1];
    ModelFile golden(argv[2]);
    const int frames = std::stoi(golden.Meta("frames"));
    auto encoder_model = std::make_shared<const EncoderModel>(models / "encodec_encoder.bin");
    auto books = std::make_shared<const Codebooks>(models / "encodec_codebooks.bin");
    auto decoder_model = std::make_shared<const DecoderModel>(models / "vocos_decoder.bin");
    TestEncoder(golden, encoder_model, books, frames);
    TestPackets(golden, frames);
    TestDecoder(golden, decoder_model, *books, frames);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return test::Finish("vocoscodec_golden_test");
}
