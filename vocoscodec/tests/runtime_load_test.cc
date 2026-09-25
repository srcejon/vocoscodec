// Loads the runtime library at run time, as SDRangel does with QLibrary,
// resolves every C ABI symbol and round-trips a frame.
//
// Usage: vocoscodec_runtime_load_test <runtime library> <model directory>

#include <cstdint>
#include <iostream>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#if defined(_WIN32)
using Library = HMODULE;
Library Open(const char* path) { return LoadLibraryA(path); }
void* Symbol(Library library, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(library, name));
}
void Close(Library library) { FreeLibrary(library); }
#else
using Library = void*;
Library Open(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void* Symbol(Library library, const char* name) { return dlsym(library, name); }
void Close(Library library) { dlclose(library); }
#endif

const char* const kSymbols[] = {
    "vocoscodec_abi_version", "vocoscodec_version", "vocoscodec_built_model_directory",
    "vocoscodec_has_models", "vocoscodec_last_error", "vocoscodec_packet_bytes",
    "vocoscodec_packet_codebooks", "vocoscodec_packet_truncate", "vocoscodec_encoder_create",
    "vocoscodec_encode", "vocoscodec_encoder_reset", "vocoscodec_encoder_destroy",
    "vocoscodec_decoder_create", "vocoscodec_decoder_delay_samples", "vocoscodec_decode",
    "vocoscodec_decode_packets", "vocoscodec_decode_lost", "vocoscodec_decoder_flush",
    "vocoscodec_decoder_is_comfort_noise", "vocoscodec_decoder_reset",
    "vocoscodec_decoder_destroy"};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: vocoscodec_runtime_load_test <runtime library> <model directory>\n";
    return 2;
  }
  Library library = Open(argv[1]);
  if (!library) {
    std::cerr << "cannot load " << argv[1] << '\n';
    return 1;
  }
  int failures = 0;
  for (const char* name : kSymbols) {
    if (!Symbol(library, name)) {
      std::cerr << "missing symbol " << name << '\n';
      ++failures;
    }
  }
  if (failures) {
    Close(library);
    return 1;
  }

  auto abi_version = reinterpret_cast<int (*)()>(Symbol(library, "vocoscodec_abi_version"));
  auto last_error = reinterpret_cast<const char* (*)()>(Symbol(library, "vocoscodec_last_error"));
  auto encoder_create =
      reinterpret_cast<void* (*)(const char*, int)>(Symbol(library, "vocoscodec_encoder_create"));
  auto encode = reinterpret_cast<int (*)(void*, const std::int16_t*, std::uint8_t*, int)>(
      Symbol(library, "vocoscodec_encode"));
  auto encoder_destroy =
      reinterpret_cast<void (*)(void*)>(Symbol(library, "vocoscodec_encoder_destroy"));
  auto decoder_create = reinterpret_cast<void* (*)(const char*, int, int)>(
      Symbol(library, "vocoscodec_decoder_create"));
  auto decode = reinterpret_cast<int (*)(void*, const std::uint8_t*, int, std::int16_t*, int)>(
      Symbol(library, "vocoscodec_decode"));
  auto decoder_destroy =
      reinterpret_cast<void (*)(void*)>(Symbol(library, "vocoscodec_decoder_destroy"));

  if (abi_version() != 1) {
    std::cerr << "ABI version " << abi_version() << ", expected 1\n";
    Close(library);
    return 1;
  }
  void* encoder = encoder_create(argv[2], 2);
  void* decoder = decoder_create(argv[2], -1, 0);
  if (!encoder || !decoder) {
    std::cerr << "create failed: " << last_error() << '\n';
    ++failures;
  } else {
    std::int16_t pcm[320] = {};
    std::uint8_t packet[20];
    const int size = encode(encoder, pcm, packet, sizeof packet);
    if (size != 3 || decode(decoder, packet, size, pcm, 320) != 320) {
      std::cerr << "round trip failed: " << last_error() << '\n';
      ++failures;
    }
  }
  encoder_destroy(encoder);
  decoder_destroy(decoder);
  Close(library);
  if (!failures) std::cout << "vocoscodec_runtime_load_test: passed\n";
  return failures ? 1 : 0;
}
