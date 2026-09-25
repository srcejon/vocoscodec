// Encode and decode time per frame, model load time and memory.
//
// Usage: vocoscodec_benchmark <model directory> [options]
//   --codebooks N   (default 4)
//   --frames N      frames timed (default 1000)
//   --group N       packets per decode call (default 1)
//   --lookahead N   (default 3)
//   --wav FILE      24 kHz mono input (default: a synthetic vowel-like signal)
//   --kernels NAME  avx2, sse4.2, neon or generic instead of the best here
//   --generic       short for --kernels generic

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include "vocoscodec/codec.h"
#include "vocoscodec/kernels.h"
#include "vocoscodec/tools/wav_io.h"

namespace {

using Clock = std::chrono::steady_clock;

double PeakMemoryMb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters))
    return counters.PeakWorkingSetSize / 1e6;
  return 0.0;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return usage.ru_maxrss / 1e6;
#else
  return usage.ru_maxrss / 1e3;
#endif
#endif
}

// Resident memory now (0 where unknown).
double CurrentMemoryMb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters))
    return counters.WorkingSetSize / 1e6;
  return 0.0;
#elif defined(__linux__)
  long pages = 0, resident = 0;
  if (FILE* f = std::fopen("/proc/self/statm", "r")) {
    if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
    std::fclose(f);
  }
  return resident * 4096 / 1e6;
#else
  return 0.0;
#endif
}

void PrintStats(const char* label, std::vector<double> ms_per_frame) {
  std::sort(ms_per_frame.begin(), ms_per_frame.end());
  const double mean =
      std::accumulate(ms_per_frame.begin(), ms_per_frame.end(), 0.0) / ms_per_frame.size();
  const double p50 = ms_per_frame[ms_per_frame.size() / 2];
  const double p95 = ms_per_frame[static_cast<std::size_t>(0.95 * (ms_per_frame.size() - 1))];
  const double frame_ms = 1000.0 * vocoscodec::kCodecFrameSamples / vocoscodec::kCodecSampleRate;
  std::cout << std::left << std::setw(7) << label << " mean " << std::setw(7) << mean
            << " p50 " << std::setw(7) << p50 << " p95 " << std::setw(7) << p95 << " max "
            << std::setw(7) << ms_per_frame.back() << " ms/frame, " << std::setw(6)
            << 100.0 * mean / frame_ms << "% of one core, " << frame_ms / mean
            << "x realtime\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: vocoscodec_benchmark <model directory> [--codebooks N] [--frames N] "
                 "[--group N] [--lookahead N] [--wav FILE] [--kernels NAME]\n";
    return 2;
  }
  int codebooks = 4, frames = 1000, group = 1;
  vocoscodec::DecoderOptions options;
  std::string wav;
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
        return argv[++i];
      };
      if (arg == "--codebooks") codebooks = std::stoi(next());
      else if (arg == "--frames") frames = std::stoi(next());
      else if (arg == "--group") group = std::stoi(next());
      else if (arg == "--lookahead") options.lookahead = std::stoi(next());
      else if (arg == "--wav") wav = next();
      else if (arg == "--generic") vocoscodec::kernels::UseGeneric(true);
      else if (arg == "--kernels") {
        const std::string name = next();
        if (!vocoscodec::kernels::Select(name))
          throw std::invalid_argument("kernels " + name + " are not available here");
      }
      else throw std::invalid_argument("unknown option " + arg);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
  if (frames <= 0 || group <= 0) {
    std::cerr << "frames and group must be positive\n";
    return 2;
  }

  std::string error;
  std::vector<std::int16_t> input;
  if (!wav.empty()) {
    int rate = 0;
    if (!vocoscodec::tools::ReadMonoWav(wav, &input, &rate, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
  } else {
    // Harmonics of 140 Hz with a slow amplitude modulation.
    input.resize(24000 * 4);
    for (std::size_t i = 0; i < input.size(); ++i) {
      const double t = i / 24000.0;
      double v = 0.0;
      for (int h = 1; h <= 20; ++h) v += std::sin(2.0 * 3.14159265358979 * 140.0 * h * t) / h;
      input[i] = static_cast<std::int16_t>(4000.0 * v * (0.6 + 0.4 * std::sin(6.28 * 3.0 * t)));
    }
  }
  const std::size_t frame = vocoscodec::kCodecFrameSamples;
  input.resize(std::max<std::size_t>(frame, input.size() / frame * frame));

  const auto load_start = Clock::now();
  auto encoder = vocoscodec::Encoder::Create(argv[1], codebooks, &error);
  auto decoder = encoder ? vocoscodec::Decoder::Create(argv[1], options, &error) : nullptr;
  const double load_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
  const double loaded_mb = CurrentMemoryMb();
  if (!encoder || !decoder) {
    std::cerr << error << '\n';
    return 1;
  }

  std::vector<std::vector<std::uint8_t>> packets(frames);
  std::vector<double> encode_ms, decode_ms;
  for (int f = 0; f < frames; ++f) {
    const std::int16_t* samples = input.data() + (f * frame) % input.size();
    const auto start = Clock::now();
    encoder->Encode(samples, frame, &packets[f], &error);
    encode_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  std::vector<std::int16_t> audio;
  for (int f = 0; f < frames; f += group) {
    const int count = std::min(group, frames - f);
    std::vector<const std::uint8_t*> pointers;
    std::vector<std::size_t> sizes;
    for (int j = f; j < f + count; ++j) {
      pointers.push_back(packets[j].data());
      sizes.push_back(packets[j].size());
    }
    const auto start = Clock::now();
    decoder->Decode(pointers.data(), sizes.data(), count, &audio, &error);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (int j = 0; j < count; ++j) decode_ms.push_back(ms / count);
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "kernels " << vocoscodec::kernels::InstructionSet() << ", " << codebooks
            << " codebooks, lookahead " << decoder->lookahead() << ", " << group
            << " packet(s) per decode call, " << frames << " frames\n";
  std::cout << "model load " << load_ms << " ms; resident after load " << loaded_mb
            << " MB, peak " << PeakMemoryMb() << " MB\n";
  PrintStats("encode", std::move(encode_ms));
  PrintStats("decode", std::move(decode_ms));
  std::cout << "decoder: " << static_cast<double>(decoder->block_evaluations()) / frames
            << " ConvNeXt blocks per frame (8 without lookahead limits)\n";
  return 0;
}
