#include "vocoscodec/c_api.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vocoscodec/codec.h"
#include "vocoscodec/packet.h"

#ifndef VOCOSCODEC_VERSION
#define VOCOSCODEC_VERSION "unknown"
#endif

#ifndef VOCOSCODEC_RUNTIME_MODEL_DIR
#define VOCOSCODEC_RUNTIME_MODEL_DIR ""
#endif

namespace {

thread_local std::string last_error;

void SetExceptionError() {
  try {
    throw;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown C++ exception";
  }
}

std::filesystem::path PathOf(const char* directory) {
  return std::filesystem::u8path(directory ? directory : "");
}

// Copies audio out, or fails if it does not fit.
int CopyOut(const std::vector<int16_t>& audio, int16_t* pcm, int capacity) {
  if (static_cast<int>(audio.size()) > capacity) {
    last_error = "audio buffer too small";
    return 0;
  }
  std::copy(audio.begin(), audio.end(), pcm);
  return static_cast<int>(audio.size());
}

}  // namespace

struct vocoscodec_encoder {
  std::unique_ptr<vocoscodec::Encoder> encoder;
  std::vector<uint8_t> packet;
};

struct vocoscodec_decoder {
  std::unique_ptr<vocoscodec::Decoder> decoder;
  std::vector<int16_t> audio;
};

int vocoscodec_abi_version(void) { return VOCOSCODEC_ABI_VERSION; }

const char* vocoscodec_version(void) { return "vocoscodec " VOCOSCODEC_VERSION; }

const char* vocoscodec_built_model_directory(void) { return VOCOSCODEC_RUNTIME_MODEL_DIR; }

int vocoscodec_has_models(const char* model_directory) {
  try {
    return model_directory && vocoscodec::HasModels(PathOf(model_directory)) ? 1 : 0;
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

const char* vocoscodec_last_error(void) { return last_error.c_str(); }

int vocoscodec_packet_bytes(int codebooks) { return vocoscodec::PacketBytes(codebooks); }

int vocoscodec_packet_codebooks(int packet_bytes) {
  return vocoscodec::PacketCodebooks(packet_bytes);
}

int vocoscodec_packet_truncate(uint8_t* packet, int packet_bytes, int codebooks) {
  last_error.clear();
  if (!packet) {
    last_error = "invalid packet";
    return 0;
  }
  const int size = vocoscodec::Truncate(packet, packet_bytes, codebooks);
  if (size == 0) last_error = "invalid packet size or codebook count";
  return size;
}

vocoscodec_encoder* vocoscodec_encoder_create(const char* model_directory, int codebooks) {
  last_error.clear();
  try {
    std::string error;
    auto encoder = vocoscodec::Encoder::Create(PathOf(model_directory), codebooks, &error);
    if (!encoder) {
      last_error = error;
      return nullptr;
    }
    auto result = std::make_unique<vocoscodec_encoder>();
    result->encoder = std::move(encoder);
    return result.release();
  } catch (...) {
    SetExceptionError();
    return nullptr;
  }
}

int vocoscodec_encode(vocoscodec_encoder* encoder, const int16_t* pcm, uint8_t* packet,
                      int packet_capacity) {
  last_error.clear();
  if (!encoder || !pcm || !packet) {
    last_error = "invalid encoder or buffer";
    return 0;
  }
  try {
    std::string error;
    if (!encoder->encoder->Encode(pcm, VOCOSCODEC_SAMPLES_PER_FRAME, &encoder->packet,
                                  &error)) {
      last_error = error;
      return 0;
    }
    if (static_cast<int>(encoder->packet.size()) > packet_capacity) {
      last_error = "packet buffer too small";
      return 0;
    }
    std::copy(encoder->packet.begin(), encoder->packet.end(), packet);
    return static_cast<int>(encoder->packet.size());
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

void vocoscodec_encoder_reset(vocoscodec_encoder* encoder) {
  if (!encoder) return;
  try {
    encoder->encoder->Reset();
  } catch (...) {
    SetExceptionError();
  }
}

void vocoscodec_encoder_destroy(vocoscodec_encoder* encoder) {
  try {
    delete encoder;
  } catch (...) {
    SetExceptionError();
  }
}

vocoscodec_decoder* vocoscodec_decoder_create(const char* model_directory, int lookahead,
                                              int flags) {
  last_error.clear();
  try {
    vocoscodec::DecoderOptions options;
    if (lookahead >= 0) options.lookahead = lookahead;
    options.comfort_noise = (flags & VOCOSCODEC_DECODER_SILENCE) == 0;
    std::string error;
    auto decoder = vocoscodec::Decoder::Create(PathOf(model_directory), options, &error);
    if (!decoder) {
      last_error = error;
      return nullptr;
    }
    auto result = std::make_unique<vocoscodec_decoder>();
    result->decoder = std::move(decoder);
    return result.release();
  } catch (...) {
    SetExceptionError();
    return nullptr;
  }
}

int vocoscodec_decoder_delay_samples(vocoscodec_decoder* decoder) {
  return decoder ? decoder->decoder->delay_samples() : 0;
}

int vocoscodec_decode_packets(vocoscodec_decoder* decoder, const uint8_t* const* packets,
                              const int* packet_bytes, int count, int16_t* pcm,
                              int pcm_capacity) {
  last_error.clear();
  if (!decoder || !pcm || count < 0 || (count > 0 && (!packets || !packet_bytes))) {
    last_error = "invalid decoder, packets or buffer";
    return 0;
  }
  if (static_cast<long long>(count) * VOCOSCODEC_SAMPLES_PER_FRAME > pcm_capacity) {
    last_error = "audio buffer too small";
    return 0;
  }
  try {
    std::vector<std::size_t> sizes(count);
    for (int i = 0; i < count; ++i) {
      if (packet_bytes[i] < 0) {
        last_error = "negative packet size";
        return 0;
      }
      sizes[i] = static_cast<std::size_t>(packet_bytes[i]);
    }
    decoder->audio.clear();
    std::string error;
    if (!decoder->decoder->Decode(packets, sizes.data(), count, &decoder->audio, &error)) {
      last_error = error;
      return 0;
    }
    return CopyOut(decoder->audio, pcm, pcm_capacity);
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

int vocoscodec_decode(vocoscodec_decoder* decoder, const uint8_t* packet, int packet_bytes,
                      int16_t* pcm, int pcm_capacity) {
  if (!packet || packet_bytes <= 0) {
    last_error = "invalid packet (use vocoscodec_decode_lost for a lost one)";
    return 0;
  }
  return vocoscodec_decode_packets(decoder, &packet, &packet_bytes, 1, pcm, pcm_capacity);
}

int vocoscodec_decode_lost(vocoscodec_decoder* decoder, int16_t* pcm, int pcm_capacity) {
  const uint8_t* none = nullptr;
  const int zero = 0;
  return vocoscodec_decode_packets(decoder, &none, &zero, 1, pcm, pcm_capacity);
}

int vocoscodec_decoder_flush(vocoscodec_decoder* decoder, int16_t* pcm, int pcm_capacity) {
  last_error.clear();
  if (!decoder || !pcm) {
    last_error = "invalid decoder or audio buffer";
    return 0;
  }
  if (decoder->decoder->delay_samples() > pcm_capacity) {
    last_error = "audio buffer too small";
    return 0;
  }
  try {
    decoder->audio.clear();
    std::string error;
    if (!decoder->decoder->Flush(&decoder->audio, &error)) {
      last_error = error;
      return 0;
    }
    return CopyOut(decoder->audio, pcm, pcm_capacity);
  } catch (...) {
    SetExceptionError();
    return 0;
  }
}

int vocoscodec_decoder_is_comfort_noise(vocoscodec_decoder* decoder) {
  return decoder && decoder->decoder->is_comfort_noise() ? 1 : 0;
}

void vocoscodec_decoder_reset(vocoscodec_decoder* decoder) {
  if (!decoder) return;
  try {
    decoder->decoder->Reset();
  } catch (...) {
    SetExceptionError();
  }
}

void vocoscodec_decoder_destroy(vocoscodec_decoder* decoder) {
  try {
    delete decoder;
  } catch (...) {
    SetExceptionError();
  }
}
