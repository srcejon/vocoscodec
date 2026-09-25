// Encodes a 24 kHz mono 16-bit WAV file to a packet file.
//
// Usage: vocoscodec_encode <model directory> <input.wav> <output.packets> [codebooks]
//                          [--kernels NAME | --generic]
// --kernels chooses avx2, sse4.2, neon or generic instead of the best here.

#include <iostream>
#include <string>
#include <vector>

#include "vocoscodec/codec.h"
#include "vocoscodec/kernels.h"
#include "vocoscodec/tools/wav_io.h"

int main(int argc, char** argv) {
  if (argc > 4 && std::string(argv[argc - 1]) == "--generic") {
    vocoscodec::kernels::UseGeneric(true);
    --argc;
  } else if (argc > 5 && std::string(argv[argc - 2]) == "--kernels") {
    if (!vocoscodec::kernels::Select(argv[argc - 1])) {
      std::cerr << "kernels " << argv[argc - 1] << " are not available here\n";
      return 2;
    }
    argc -= 2;
  }
  if (argc < 4 || argc > 5) {
    std::cerr << "usage: vocoscodec_encode <model directory> <input.wav> <output.packets> "
                 "[codebooks 1..16, default 4] [--generic]\n";
    return 2;
  }
  int codebooks = 4;
  try {
    if (argc == 5) codebooks = std::stoi(argv[4]);
  } catch (const std::exception&) {
    std::cerr << "invalid codebook count\n";
    return 2;
  }
  std::string error;
  std::vector<std::int16_t> samples;
  int rate = 0;
  if (!vocoscodec::tools::ReadMonoWav(argv[2], &samples, &rate, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (rate != vocoscodec::kCodecSampleRate) {
    std::cerr << argv[2] << ": " << rate << " Hz; the codec takes 24000 Hz\n";
    return 1;
  }
  auto encoder = vocoscodec::Encoder::Create(argv[1], codebooks, &error);
  if (!encoder) {
    std::cerr << error << '\n';
    return 1;
  }
  const std::size_t frame = vocoscodec::kCodecFrameSamples;
  samples.resize((samples.size() + frame - 1) / frame * frame, 0);  // pad the last frame
  std::vector<std::vector<std::uint8_t>> packets;
  for (std::size_t i = 0; i < samples.size(); i += frame) {
    std::vector<std::uint8_t> packet;
    if (!encoder->Encode(samples.data() + i, frame, &packet, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    packets.push_back(std::move(packet));
  }
  if (!vocoscodec::tools::WritePackets(argv[3], packets, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  return 0;
}
