// Decodes a packet file (size-0 records are lost packets) to a 24 kHz WAV.
//
// Usage: vocoscodec_decode <model directory> <input.packets> <output.wav> [options]
//   --lookahead N     frames of decoder lookahead (default 3)
//   --silence         long losses fade to silence instead of comfort noise
//   --codebooks N     truncate every packet to N codebooks first
//   --group N         packets per decode call (default 1)
//   --keep-delay      keep the codec delay at the start (default: trim it,
//                     so the output lines up with the encoder's input)
//   --repeat N --fade N   concealment timing (frames)
//   --bandwidth-id N  force the Vocos bandwidth embedding (0..3)

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "vocoscodec/codec.h"
#include "vocoscodec/packet.h"
#include "vocoscodec/tools/wav_io.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: vocoscodec_decode <model directory> <input.packets> <output.wav> "
                 "[--lookahead N] [--silence] [--codebooks N] [--group N] [--keep-delay] "
                 "[--repeat N] [--fade N] [--bandwidth-id N]\n";
    return 2;
  }
  vocoscodec::DecoderOptions options;
  int codebooks = 0, group = 1;
  bool keep_delay = false;
  try {
    for (int i = 4; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() {
        if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
        return std::stoi(argv[++i]);
      };
      if (arg == "--lookahead") options.lookahead = value();
      else if (arg == "--silence") options.comfort_noise = false;
      else if (arg == "--codebooks") codebooks = value();
      else if (arg == "--group") group = value();
      else if (arg == "--keep-delay") keep_delay = true;
      else if (arg == "--repeat") options.conceal_repeat_frames = value();
      else if (arg == "--fade") options.conceal_fade_frames = value();
      else if (arg == "--bandwidth-id") options.bandwidth_id = value();
      else throw std::invalid_argument("unknown option " + arg);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
  if (group < 1) {
    std::cerr << "--group must be at least 1\n";
    return 2;
  }

  std::string error;
  std::vector<std::vector<std::uint8_t>> packets;
  if (!vocoscodec::tools::ReadPackets(argv[2], &packets, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (codebooks > 0) {
    for (auto& packet : packets) {
      if (packet.empty()) continue;
      const int size = vocoscodec::Truncate(packet.data(), static_cast<int>(packet.size()), codebooks);
      if (size == 0) {
        std::cerr << "cannot truncate a " << packet.size() << "-byte packet to " << codebooks
                  << " codebooks\n";
        return 1;
      }
      packet.resize(size);
    }
  }
  auto decoder = vocoscodec::Decoder::Create(argv[1], options, &error);
  if (!decoder) {
    std::cerr << error << '\n';
    return 1;
  }
  std::vector<std::int16_t> audio;
  for (std::size_t f = 0; f < packets.size(); f += group) {
    std::vector<const std::uint8_t*> pointers;
    std::vector<std::size_t> sizes;
    for (std::size_t j = f; j < std::min(packets.size(), f + group); ++j) {
      pointers.push_back(packets[j].data());
      sizes.push_back(packets[j].size());
    }
    if (!decoder->Decode(pointers.data(), sizes.data(), static_cast<int>(sizes.size()), &audio,
                         &error)) {
      std::cerr << "packet " << f << ": " << error << '\n';
      return 1;
    }
  }
  const int delay = decoder->delay_samples();
  decoder->Flush(&audio, &error);
  if (!keep_delay) audio.erase(audio.begin(), audio.begin() + delay);
  if (!vocoscodec::tools::WriteMonoWav(argv[3], audio, vocoscodec::kCodecSampleRate, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  return 0;
}
