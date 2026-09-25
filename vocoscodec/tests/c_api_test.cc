// The C ABI, linked against the runtime library.
//
// Usage: vocoscodec_c_api_test <model directory> <golden.bin>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vocoscodec/c_api.h"
#include "vocoscodec/model_file.h"
#include "vocoscodec/tests/test_util.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <model directory> <golden.bin>\n", argv[0]);
    return 2;
  }
  const char* models = argv[1];
  vocoscodec::ModelFile golden(argv[2]);
  const int frames = std::stoi(golden.Meta("frames"));
  const float* pcm = golden.Get("pcm", {static_cast<std::uint32_t>(frames * 320)}).data;

  CHECK(vocoscodec_abi_version() == VOCOSCODEC_ABI_VERSION);
  CHECK(std::strncmp(vocoscodec_version(), "vocoscodec ", 11) == 0);
  CHECK(vocoscodec_built_model_directory() != nullptr);
  CHECK(vocoscodec_has_models(models) == 1);
  CHECK(vocoscodec_has_models("no such directory") == 0);
  CHECK(vocoscodec_has_models(nullptr) == 0);
  CHECK(vocoscodec_packet_bytes(2) == 3 && vocoscodec_packet_bytes(16) == 20);
  CHECK(vocoscodec_packet_bytes(0) == 0 && vocoscodec_packet_bytes(17) == 0);
  CHECK(vocoscodec_packet_codebooks(5) == 4 && vocoscodec_packet_codebooks(6) == 0);

  // Failures set the last error; null handles are harmless.
  CHECK(vocoscodec_encoder_create("no such directory", 4) == nullptr);
  CHECK(std::strlen(vocoscodec_last_error()) > 0);
  CHECK(vocoscodec_encoder_create(models, 0) == nullptr);
  CHECK(vocoscodec_decoder_create(models, 28, 0) == nullptr);
  vocoscodec_encoder_reset(nullptr);
  vocoscodec_encoder_destroy(nullptr);
  vocoscodec_decoder_reset(nullptr);
  vocoscodec_decoder_destroy(nullptr);
  CHECK(vocoscodec_decoder_is_comfort_noise(nullptr) == 0);
  CHECK(vocoscodec_decoder_delay_samples(nullptr) == 0);
  std::int16_t buffer[320 * 4];
  CHECK(vocoscodec_decode_lost(nullptr, buffer, 320) == 0);

  vocoscodec_encoder* encoder = vocoscodec_encoder_create(models, 4);
  vocoscodec_decoder* decoder = vocoscodec_decoder_create(models, -1, 0);
  CHECK(encoder != nullptr && decoder != nullptr);
  if (!encoder || !decoder) return test::Finish("vocoscodec_c_api_test");
  CHECK(std::strlen(vocoscodec_last_error()) == 0);
  CHECK(vocoscodec_decoder_delay_samples(decoder) == 480 + 320 * VOCOSCODEC_DEFAULT_LOOKAHEAD);

  // Encode everything, then decode in groups of four with every tenth lost.
  std::vector<std::vector<std::uint8_t>> packets;
  for (int f = 0; f < frames; ++f) {
    std::int16_t samples[320];
    for (int i = 0; i < 320; ++i) samples[i] = static_cast<std::int16_t>(pcm[f * 320 + i] * 32768.0f);
    std::uint8_t packet[VOCOSCODEC_MAX_PACKET_BYTES];
    CHECK(vocoscodec_encode(encoder, samples, packet, 4) == 0);  // buffer too small
    const int size = vocoscodec_encode(encoder, samples, packet, sizeof packet);
    CHECK(size == 5);
    packets.emplace_back(packet, packet + size);
  }
  std::vector<std::int16_t> audio;
  for (int f = 0; f < frames; f += 4) {
    const std::uint8_t* pointers[4];
    int sizes[4];
    for (int j = 0; j < 4; ++j) {
      const bool lost = (f + j) % 10 == 9;
      pointers[j] = lost ? nullptr : packets[f + j].data();
      sizes[j] = lost ? 0 : static_cast<int>(packets[f + j].size());
    }
    CHECK(vocoscodec_decode_packets(decoder, pointers, sizes, 4, buffer, 320 * 3) == 0);
    const int n = vocoscodec_decode_packets(decoder, pointers, sizes, 4, buffer, 320 * 4);
    CHECK(n == 1280);
    audio.insert(audio.end(), buffer, buffer + n);
  }
  std::vector<std::int16_t> tail(vocoscodec_decoder_delay_samples(decoder));
  CHECK(vocoscodec_decoder_flush(decoder, tail.data(), static_cast<int>(tail.size())) ==
        static_cast<int>(tail.size()));
  double energy = 0.0;
  for (auto s : audio) energy += double(s) * s;
  CHECK(energy > 0.0);

  // Single packets, a lost one, a bad one.
  CHECK(vocoscodec_decode(decoder, packets[0].data(), 5, buffer, 320) == 320);
  CHECK(vocoscodec_decode_lost(decoder, buffer, 320) == 320);
  CHECK(vocoscodec_decode(decoder, packets[0].data(), 6, buffer, 320) == 0);
  CHECK(std::strlen(vocoscodec_last_error()) > 0);
  CHECK(vocoscodec_decode(decoder, packets[0].data(), 5, buffer, 100) == 0);

  // Relays truncate.
  std::uint8_t packet[5];
  std::memcpy(packet, packets[1].data(), 5);
  CHECK(vocoscodec_packet_truncate(packet, 5, 2) == 3);
  CHECK((packet[2] & 0x0F) == 0);
  CHECK(vocoscodec_packet_truncate(packet, 3, 4) == 0);
  CHECK(vocoscodec_decode(decoder, packet, 3, buffer, 320) == 320);

  vocoscodec_decoder* silent = vocoscodec_decoder_create(models, 0, VOCOSCODEC_DECODER_SILENCE);
  CHECK(silent && vocoscodec_decoder_delay_samples(silent) == 480);
  vocoscodec_decoder_destroy(silent);

  vocoscodec_encoder_reset(encoder);
  vocoscodec_decoder_reset(decoder);
  vocoscodec_encoder_destroy(encoder);
  vocoscodec_decoder_destroy(decoder);
  return test::Finish("vocoscodec_c_api_test");
}
