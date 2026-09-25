#include "vocoscodec/codec.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <mutex>

#include "vocoscodec/concealment.h"
#include "vocoscodec/decoder.h"
#include "vocoscodec/encoder.h"
#include "vocoscodec/packet.h"

namespace vocoscodec {

const char* const kModelFiles[3] = {"encodec_encoder.bin", "encodec_codebooks.bin",
                                    "vocos_decoder.bin"};

namespace {

// Models are read once per process and shared by every encoder and decoder
// using the same file; they are freed with the last user.
template <typename Model>
std::shared_ptr<const Model> LoadShared(const std::filesystem::path& path) {
  static std::mutex mutex;
  static std::map<std::string, std::weak_ptr<const Model>> cache;
  std::error_code ignored;
  const std::string key = std::filesystem::weakly_canonical(path, ignored).string();
  std::lock_guard<std::mutex> lock(mutex);
  if (auto model = cache[key].lock()) return model;
  auto model = std::make_shared<const Model>(path);
  cache[key] = model;
  return model;
}

std::string MissingModels(const std::filesystem::path& directory) {
  if (directory.empty()) return "no model directory given";
  for (const char* name : kModelFiles) {
    std::error_code ignored;
    if (!std::filesystem::is_regular_file(directory / name, ignored)) {
      return "model file " + (directory / name).string() + " not found";
    }
  }
  return {};
}

void ToPcm16(const float* in, std::size_t n, std::vector<std::int16_t>* out) {
  const std::size_t start = out->size();
  out->resize(start + n);
  for (std::size_t i = 0; i < n; ++i) {
    const float scaled = std::nearbyint(in[i] * 32768.0f);
    (*out)[start + i] = static_cast<std::int16_t>(std::clamp(scaled, -32768.0f, 32767.0f));
  }
}

}  // namespace

bool HasModels(const std::filesystem::path& model_directory) {
  return MissingModels(model_directory).empty();
}

int BandwidthId(int codebooks) {
  // Vocos has embeddings for 1.5, 3, 6 and 12 kbit/s (2, 4, 8, 16 codebooks).
  // Best by PESQ-WB for each count on the dev set (validation/README.md).
  if (codebooks <= 3) return 0;
  if (codebooks <= 5) return 1;
  if (codebooks <= 10) return 2;
  return 3;
}

// ---------------------------------------------------------------- encoder

Encoder::Encoder(std::unique_ptr<StreamingEncoder> encoder, int codebooks)
    : encoder_(std::move(encoder)), codebooks_(codebooks), indices_(codebooks),
      samples_(kCodecFrameSamples) {}

Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::Create(const std::filesystem::path& model_directory,
                                         int codebooks, std::string* error) {
  try {
    if (codebooks < 1 || codebooks > kCodecMaxCodebooks) {
      if (error) *error = "codebook count must be 1..16";
      return nullptr;
    }
    const std::string missing = MissingModels(model_directory);
    if (!missing.empty()) {
      if (error) *error = missing;
      return nullptr;
    }
    auto model = LoadShared<EncoderModel>(model_directory / kModelFiles[0]);
    auto books = LoadShared<Codebooks>(model_directory / kModelFiles[1]);
    auto streaming = std::make_unique<StreamingEncoder>(model, books, codebooks);
    return std::unique_ptr<Encoder>(new Encoder(std::move(streaming), codebooks));
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
}

bool Encoder::Encode(const float* audio, std::size_t sample_count,
                     std::vector<std::uint8_t>* packet, std::string* error) {
  if (sample_count != kCodecFrameSamples) {
    if (error) *error = "a frame is 320 samples";
    return false;
  }
  encoder_->Encode(audio, indices_.data());
  packet->resize(PacketBytes(codebooks_));
  Pack(indices_.data(), codebooks_, packet->data());
  return true;
}

bool Encoder::Encode(const std::int16_t* audio, std::size_t sample_count,
                     std::vector<std::uint8_t>* packet, std::string* error) {
  if (sample_count != kCodecFrameSamples) {
    if (error) *error = "a frame is 320 samples";
    return false;
  }
  for (int i = 0; i < kCodecFrameSamples; ++i) samples_[i] = audio[i] / 32768.0f;
  return Encode(samples_.data(), sample_count, packet, error);
}

void Encoder::Reset() { encoder_->Reset(); }

// ---------------------------------------------------------------- decoder

Decoder::Decoder(std::unique_ptr<StreamingDecoder> decoder,
                 std::shared_ptr<const Codebooks> books, const DecoderOptions& options)
    : decoder_(std::move(decoder)),
      books_(std::move(books)),
      concealer_(std::make_unique<Concealer>(ConcealmentParameters{
          options.conceal_repeat_frames, options.conceal_fade_frames})),
      bandwidth_id_(options.bandwidth_id) {}

Decoder::~Decoder() = default;

std::unique_ptr<Decoder> Decoder::Create(const std::filesystem::path& model_directory,
                                         const DecoderOptions& options, std::string* error) {
  try {
    if (options.lookahead < 0 || options.lookahead > kReceptiveField) {
      if (error) *error = "lookahead must be 0..27 frames";
      return nullptr;
    }
    if (options.bandwidth_id < -1 || options.bandwidth_id >= kBandwidths ||
        options.conceal_repeat_frames < 0 || options.conceal_fade_frames < 0) {
      if (error) *error = "invalid decoder option";
      return nullptr;
    }
    const std::string missing = MissingModels(model_directory);
    if (!missing.empty()) {
      if (error) *error = missing;
      return nullptr;
    }
    auto model = LoadShared<DecoderModel>(model_directory / kModelFiles[2]);
    auto books = LoadShared<Codebooks>(model_directory / kModelFiles[1]);
    auto streaming =
        std::make_unique<StreamingDecoder>(model, options.lookahead, options.comfort_noise);
    return std::unique_ptr<Decoder>(
        new Decoder(std::move(streaming), std::move(books), options));
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
}

bool Decoder::DecodeFloat(const std::uint8_t* const* packets, const std::size_t* sizes,
                          int count, std::string* error) {
  // Every packet is checked before any changes the decoder's state.
  for (int j = 0; j < count; ++j) {
    if (sizes[j] != 0 &&
        (!packets || !packets[j] || PacketCodebooks(static_cast<int>(sizes[j])) == 0)) {
      if (error) {
        *error = packets && packets[j]
                     ? "no codebook count gives a " + std::to_string(sizes[j]) + "-byte packet"
                     : "missing packet data";
      }
      return false;
    }
  }
  features_.assign(static_cast<std::size_t>(count) * kLatentDim, 0.0f);
  ids_.resize(count);
  noise_weights_.assign(count, 0.0f);
  concealed_.assign(count, 0);
  for (int j = 0; j < count; ++j) {
    float* features = features_.data() + static_cast<std::size_t>(j) * kLatentDim;
    if (sizes[j] == 0) {
      concealer_->Lost(features, &ids_[j], &noise_weights_[j]);
      concealed_[j] = 1;
      continue;
    }
    int indices[kMaxCodebooks];
    const int codebooks = Unpack(packets[j], static_cast<int>(sizes[j]), indices);
    for (int q = 0; q < codebooks; ++q) {
      const float* entry = books_->Entry(q, indices[q]);
      for (int i = 0; i < kLatentDim; ++i) features[i] += entry[i];
    }
    ids_[j] = bandwidth_id_ >= 0 ? bandwidth_id_ : BandwidthId(codebooks);
    concealer_->Received(features, ids_[j]);
  }
  output_.resize(static_cast<std::size_t>(count) * kFrameSamples);
  decoder_->Run(features_.data(), ids_.data(), count, output_.data(), noise_weights_.data(),
                concealed_.data());
  return true;
}

bool Decoder::Decode(const std::uint8_t* const* packets, const std::size_t* sizes, int count,
                     std::vector<float>* audio, std::string* error) {
  if (count < 0) {
    if (error) *error = "negative packet count";
    return false;
  }
  if (!DecodeFloat(packets, sizes, count, error)) return false;
  audio->insert(audio->end(), output_.begin(), output_.end());
  return true;
}

bool Decoder::Decode(const std::uint8_t* const* packets, const std::size_t* sizes, int count,
                     std::vector<std::int16_t>* audio, std::string* error) {
  if (count < 0) {
    if (error) *error = "negative packet count";
    return false;
  }
  if (!DecodeFloat(packets, sizes, count, error)) return false;
  ToPcm16(output_.data(), output_.size(), audio);
  return true;
}

bool Decoder::Decode(const std::uint8_t* packet, std::size_t packet_size,
                     std::vector<std::int16_t>* audio, std::string* error) {
  if (packet_size == 0) {
    if (error) *error = "empty packet (use DecodeLostFrame for a lost one)";
    return false;
  }
  return Decode(&packet, &packet_size, 1, audio, error);
}

bool Decoder::DecodeLostFrame(std::vector<std::int16_t>* audio, std::string* error) {
  const std::uint8_t* none = nullptr;
  const std::size_t zero = 0;
  return Decode(&none, &zero, 1, audio, error);
}

bool Decoder::Flush(std::vector<float>* audio, std::string*) {
  const std::size_t start = audio->size();
  audio->resize(start + decoder_->delay());
  decoder_->Flush(audio->data() + start);
  concealer_->Reset();
  return true;
}

bool Decoder::Flush(std::vector<std::int16_t>* audio, std::string* error) {
  std::vector<float> tail;
  if (!Flush(&tail, error)) return false;
  ToPcm16(tail.data(), tail.size(), audio);
  return true;
}

bool Decoder::is_comfort_noise() const { return decoder_->is_comfort_noise(); }

void Decoder::Reset() {
  decoder_->Reset();
  concealer_->Reset();
}

int Decoder::delay_samples() const { return decoder_->delay(); }
int Decoder::lookahead() const { return decoder_->lookahead(); }
long long Decoder::block_evaluations() const { return decoder_->block_evaluations(); }

}  // namespace vocoscodec
