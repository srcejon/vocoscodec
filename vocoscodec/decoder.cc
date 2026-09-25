#include "vocoscodec/decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vocoscodec/kernels.h"
#include "vocoscodec/model_file.h"

namespace vocoscodec {
namespace {

constexpr int kTaps = 7;
constexpr int kCentre = 3;
constexpr int kHeadOut = 2 * kBins;
constexpr float kEps = 1e-6f;

std::vector<float> Copy(const Tensor& t) { return std::vector<float>(t.data, t.data + t.size); }

// [rows][cols] -> [cols][rows]
std::vector<float> Transposed(const Tensor& t, int rows, int cols) {
  std::vector<float> out(t.size);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      out[static_cast<std::size_t>(c) * rows + r] = t.data[static_cast<std::size_t>(r) * cols + c];
  return out;
}

void LayerNorm(float* x, const float* scale, const float* shift) {
  float mean = 0.0f;
  for (int i = 0; i < kDim; ++i) mean += x[i];
  mean /= kDim;
  float variance = 0.0f;
  for (int i = 0; i < kDim; ++i) variance += (x[i] - mean) * (x[i] - mean);
  variance /= kDim;
  const float inv = 1.0f / std::sqrt(variance + kEps);
  for (int i = 0; i < kDim; ++i) x[i] = (x[i] - mean) * inv * scale[i] + shift[i];
}

}  // namespace

DecoderModel::DecoderModel(const std::filesystem::path& path) {
  ModelFile file(path);
  const Tensor& embed = file.Get("embed.w", {kDim, kLatentDim, kTaps});
  std::vector<float> embed_t(embed.size);  // [out][in][tap] -> [tap][in][out]
  for (int o = 0; o < kDim; ++o)
    for (int i = 0; i < kLatentDim; ++i)
      for (int k = 0; k < kTaps; ++k)
        embed_t[(static_cast<std::size_t>(k) * kLatentDim + i) * kDim + o] =
            embed.data[(static_cast<std::size_t>(o) * kLatentDim + i) * kTaps + k];
  embed_w.Assign(std::move(embed_t), UseHalfWeights());
  embed_b = Copy(file.Get("embed.b", {kDim}));
  norm_scale = Copy(file.Get("norm.scale", {kBandwidths, kDim}));
  norm_shift = Copy(file.Get("norm.shift", {kBandwidths, kDim}));
  for (int b = 0; b < kBlocks; ++b) {
    const std::string p = "block" + std::to_string(b) + ".";
    blocks[b].dw_w = Transposed(file.Get(p + "dw.w", {kDim, kTaps}), kDim, kTaps);
    blocks[b].dw_b = Copy(file.Get(p + "dw.b", {kDim}));
    blocks[b].norm_scale = Copy(file.Get(p + "norm.scale", {kBandwidths, kDim}));
    blocks[b].norm_shift = Copy(file.Get(p + "norm.shift", {kBandwidths, kDim}));
    blocks[b].pw1_w.Assign(
        Transposed(file.Get(p + "pw1.w", {kIntermediate, kDim}), kIntermediate, kDim),
        UseHalfWeights());
    blocks[b].pw1_b = Copy(file.Get(p + "pw1.b", {kIntermediate}));
    blocks[b].pw2_w.Assign(
        Transposed(file.Get(p + "pw2.w", {kDim, kIntermediate}), kDim, kIntermediate),
        UseHalfWeights());
    blocks[b].pw2_b = Copy(file.Get(p + "pw2.b", {kDim}));
  }
  if (file.Has("block8.pw1.w")) throw std::runtime_error("decoder has more than 8 blocks");
  final_w = Copy(file.Get("final_norm.w", {kDim}));
  final_b = Copy(file.Get("final_norm.b", {kDim}));
  head_w.Assign(Transposed(file.Get("head.w", {kHeadOut, kDim}), kHeadOut, kDim), UseHalfWeights());
  head_b = Copy(file.Get("head.b", {kHeadOut}));
}

std::size_t DecoderModel::bytes() const {
  std::size_t total = embed_w.bytes() + head_w.bytes();
  const auto floats = [](const std::vector<float>& v) { return v.size() * sizeof(float); };
  total += floats(embed_b) + floats(norm_scale) + floats(norm_shift) + floats(final_w) +
           floats(final_b) + floats(head_b);
  for (const Block& b : blocks) {
    total += b.pw1_w.bytes() + b.pw2_w.bytes() + floats(b.dw_w) + floats(b.dw_b) +
             floats(b.norm_scale) + floats(b.norm_shift) + floats(b.pw1_b) + floats(b.pw2_b);
  }
  return total;
}

// ---------------------------------------------------------------- decoder

StreamingDecoder::StreamingDecoder(std::shared_ptr<const DecoderModel> model, int lookahead,
                                   bool comfort_noise)
    : model_(std::move(model)),
      lookahead_(lookahead),
      fft_(kFftSize),
      comfort_noise_(comfort_noise) {
  if (lookahead < 0 || lookahead > kReceptiveField) {
    throw std::invalid_argument("lookahead must be 0.." + std::to_string(kReceptiveField) +
                                " frames");
  }
  delay_ = kIstftPad + kFrameSamples * lookahead_;
  features_.resize(static_cast<std::size_t>(kCapacity) * kLatentDim);
  for (auto& layer : layers_) layer.resize(static_cast<std::size_t>(kCapacity) * kDim);
  ids_.resize(kCapacity);
  noise_weights_.resize(kCapacity);
  concealed_.resize(kCapacity);
  y_.resize(static_cast<std::size_t>(kCapacity) * kDim);
  z_.resize(static_cast<std::size_t>(kCapacity) * kIntermediate);
  residual_.resize(static_cast<std::size_t>(kCapacity) * kDim);
  hidden_.resize(static_cast<std::size_t>(kCapacity) * kDim);
  head_.resize(static_cast<std::size_t>(kCapacity) * kHeadOut);

  window_.resize(kFftSize);
  window_sq_.resize(kFftSize);
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < kFftSize; ++i) {  // periodic Hann, as torch.hann_window
    window_[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * i / kFftSize));
    window_sq_[i] = window_[i] * window_[i];
  }
  frame_.resize(kFftSize);
  ola_.resize(kFftSize);
  envelope_.resize(kFftSize);
  spectrum_.resize(kBins);
  magnitudes_.resize(kBins);
  Reset();
}

void StreamingDecoder::Reset() {
  // Rows before position 0 stay zero: the whole-signal decoder's padding.
  base_ = -(kReceptiveField + 8);
  std::fill(features_.begin(), features_.end(), 0.0f);
  for (auto& layer : layers_) std::fill(layer.begin(), layer.end(), 0.0f);
  std::fill(ids_.begin(), ids_.end(), 0);
  std::fill(noise_weights_.begin(), noise_weights_.end(), 0.0f);
  std::fill(concealed_.begin(), concealed_.end(), std::uint8_t{0});
  edge_ = -1;
  final_.fill(-1);
  emitted_ = -1;
  std::fill(ola_.begin(), ola_.end(), 0.0f);
  std::fill(envelope_.begin(), envelope_.end(), 0.0f);
  comfort_noise_.Reset();
  comfort_noise_only_ = false;
  samples_out_ = 0;
}

float* StreamingDecoder::Row(int layer, long long position) {
  const auto row = static_cast<std::size_t>(position - base_);
  return layer < 0 ? features_.data() + row * kLatentDim
                   : layers_[layer].data() + row * kDim;
}

// Drops rows no later call reads: the next output frame's receptive field
// starts 27 frames before it, and the embedding reads 3 more.
void StreamingDecoder::Compact() {
  const long long new_base = emitted_ + 1 - kReceptiveField - 8;
  const long long shift = new_base - base_;
  if (shift <= 0) return;
  const auto move = [shift](std::vector<float>& rows, int channels) {
    const std::size_t offset = static_cast<std::size_t>(shift) * channels;
    std::move(rows.begin() + offset, rows.end(), rows.begin());
    std::fill(rows.end() - offset, rows.end(), 0.0f);
  };
  move(features_, kLatentDim);
  for (auto& layer : layers_) move(layer, kDim);
  std::move(ids_.begin() + shift, ids_.end(), ids_.begin());
  std::fill(ids_.end() - shift, ids_.end(), 0);
  std::move(noise_weights_.begin() + shift, noise_weights_.end(), noise_weights_.begin());
  std::fill(noise_weights_.end() - shift, noise_weights_.end(), 0.0f);
  std::move(concealed_.begin() + shift, concealed_.end(), concealed_.begin());
  std::fill(concealed_.end() - shift, concealed_.end(), std::uint8_t{0});
  base_ = new_base;
}

void StreamingDecoder::Advance(const float* features, const int* ids,
                               const float* noise_weights, const std::uint8_t* concealed, int count) {
  // Rows up to edge + count + 3 (the zero padding beyond the edge) must fit.
  if (edge_ + count + 4 - base_ > kCapacity) Compact();
  if (edge_ + count + 4 - base_ > kCapacity) throw std::logic_error("decoder rows exhausted");
  for (int j = 0; j < count; ++j) {
    const long long p = edge_ + 1 + j;
    std::copy(features + j * kLatentDim, features + (j + 1) * kLatentDim, Row(-1, p));
    ids_[p - base_] = ids[j];
    noise_weights_[p - base_] = noise_weights ? noise_weights[j] : 0.0f;
    concealed_[p - base_] = concealed && concealed[j];
  }
  edge_ += count;
}

void StreamingDecoder::ComputeEmbed(Range range) {
  if (range.hi < range.lo) return;
  const DecoderModel& m = *model_;
  const int count = static_cast<int>(range.hi - range.lo + 1);
  constexpr int depth = kTaps * kLatentDim;
  m.embed_w.DenseT(m.embed_b.data(), kDim, depth, Row(-1, range.lo - kCentre), kLatentDim,
                   count, Row(0, range.lo), kDim, false);
  for (long long p = range.lo; p <= range.hi; ++p) {
    const int id = ids_[p - base_];
    LayerNorm(Row(0, p), m.norm_scale.data() + id * kDim, m.norm_shift.data() + id * kDim);
  }
}

void StreamingDecoder::ComputeBlock(int b, Range first, Range second) {
  const auto& block = model_->blocks[b];
  // The positions of both ranges become consecutive rows of y_ and residual_.
  std::vector<long long>& positions = positions_;
  positions.clear();
  for (const Range& range : {first, second})
    for (long long p = range.lo; p <= range.hi; ++p) positions.push_back(p);
  const int count = static_cast<int>(positions.size());
  if (count == 0) return;
  for (int j = 0; j < count; ++j) {
    const long long p = positions[j];
    float* y = y_.data() + static_cast<std::size_t>(j) * kDim;
    std::copy(block.dw_b.begin(), block.dw_b.end(), y);
    for (int k = 0; k < kTaps; ++k) {
      const float* w = block.dw_w.data() + k * kDim;
      const float* x = Row(b, p - kCentre + k);
      for (int c = 0; c < kDim; ++c) y[c] += w[c] * x[c];
    }
    const int id = ids_[p - base_];
    LayerNorm(y, block.norm_scale.data() + id * kDim, block.norm_shift.data() + id * kDim);
    std::copy(Row(b, p), Row(b, p) + kDim, residual_.data() + static_cast<std::size_t>(j) * kDim);
  }
  block.pw1_w.DenseT(block.pw1_b.data(), kIntermediate, kDim, y_.data(), kDim, count, z_.data(),
                     kIntermediate, false);
  kernels::Gelu(z_.data(), static_cast<std::size_t>(count) * kIntermediate);
  block.pw2_w.DenseT(block.pw2_b.data(), kDim, kIntermediate, z_.data(), kIntermediate, count,
                     residual_.data(), kDim, true);
  for (int j = 0; j < count; ++j) {
    const float* r = residual_.data() + static_cast<std::size_t>(j) * kDim;
    std::copy(r, r + kDim, Row(b + 1, positions[j]));
  }
  block_evaluations_ += count;
}

// Computes what output frames first..last need with the right edge at
// edge_, and every value that has become final.
void StreamingDecoder::Compute(long long first, long long last) {
  // The positions each layer must provide, from the top down. Beyond the
  // edge a layer's input is zero, so nothing past it is ever needed.
  std::array<Range, kLayers> need;
  need[kLayers - 1] = {first, last};
  for (int k = kLayers - 2; k >= 0; --k) {
    const Range& above = need[k + 1];
    need[k] = above.hi < above.lo
                  ? above
                  : Range{above.lo - kCentre, std::min(above.hi + kCentre, edge_)};
  }
  for (int k = 0; k < kLayers; ++k) {
    // Layer k is final up to edge - 3 (k + 1). The top layer's values feed
    // nothing else, so only the needed ones are computed there.
    std::array<Range, 2> ranges{Range{0, -1}, Range{0, -1}};
    if (k < kLayers - 1) {
      const long long now_final = std::max(final_[k], edge_ - kCentre * (k + 1));
      const Range finals{final_[k] + 1, now_final};
      const Range rest{std::max(need[k].lo, now_final + 1), need[k].hi};
      if (rest.hi >= rest.lo && rest.lo <= finals.hi + 1) {
        ranges[0] = {finals.lo, std::max(finals.hi, rest.hi)};
      } else {
        ranges[0] = finals;
        ranges[1] = rest;
      }
      final_[k] = now_final;
    } else {
      ranges[0] = need[k];
    }
    if (k == 0) {
      ComputeEmbed(ranges[0]);
      ComputeEmbed(ranges[1]);
    } else {
      ComputeBlock(k - 1, ranges[0], ranges[1]);
    }
  }
}

// Final layer norm, head and inverse STFT for frames first..last.
void StreamingDecoder::Synthesize(long long first, long long last, float* out) {
  const int frames = static_cast<int>(last - first + 1);
  if (frames <= 0) return;
  for (int j = 0; j < frames; ++j) {
    float* h = hidden_.data() + static_cast<std::size_t>(j) * kDim;
    const float* top = Row(kLayers - 1, first + j);
    std::copy(top, top + kDim, h);
    LayerNorm(h, model_->final_w.data(), model_->final_b.data());
  }
  model_->head_w.DenseT(model_->head_b.data(), kHeadOut, kDim, hidden_.data(), kDim, frames,
                        head_.data(), kHeadOut, false);
  for (int f = 0; f < frames; ++f) {
    const float* h = head_.data() + static_cast<std::size_t>(f) * kHeadOut;
    for (int k = 0; k < kBins; ++k) {
      const float magnitude = std::min(std::exp(h[k]), 100.0f);
      const float phase = h[kBins + k];
      magnitudes_[k] = magnitude;
      spectrum_[k] = {magnitude * std::cos(phase), magnitude * std::sin(phase)};
    }
    const std::size_t row = static_cast<std::size_t>(first + f - base_);
    if (!concealed_[row]) comfort_noise_.Observe(magnitudes_.data());
    comfort_noise_.Mix(spectrum_.data(), noise_weights_[row]);
    comfort_noise_only_ = noise_weights_[row] >= 1.0f;
    fft_.Run(spectrum_.data(), frame_.data());
    for (int i = 0; i < kFftSize; ++i) {
      ola_[i] += frame_[i] * window_[i];
      envelope_[i] += window_sq_[i];
    }
    float* block = out + static_cast<std::size_t>(f) * kFrameSamples;
    for (int i = 0; i < kFrameSamples; ++i) {
      block[i] = envelope_[i] > 1e-11f ? ola_[i] / envelope_[i] : 0.0f;
    }
    std::move(ola_.begin() + kFrameSamples, ola_.end(), ola_.begin());
    std::move(envelope_.begin() + kFrameSamples, envelope_.end(), envelope_.begin());
    std::fill(ola_.end() - kFrameSamples, ola_.end(), 0.0f);
    std::fill(envelope_.end() - kFrameSamples, envelope_.end(), 0.0f);
  }
  emitted_ = last;
}

// Advances the output clock; samples before signal time 0 are silence.
void StreamingDecoder::Emit(float* out, int samples) {
  const long long start = samples_out_ - delay_;
  if (start < 0) std::fill(out, out + std::min<long long>(samples, -start), 0.0f);
  samples_out_ += samples;
}

void StreamingDecoder::Run(const float* features, const int* bandwidth_ids, int count,
                           float* out, const float* noise_weights, const std::uint8_t* concealed) {
  while (count > 0) {
    const int batch = std::min(count, kMaxBatch);
    Advance(features, bandwidth_ids, noise_weights, concealed, batch);
    const long long first = emitted_ + 1;
    const long long last = edge_ - lookahead_;
    Compute(first, last);
    // Until the lookahead has filled at the start, a frame in gives no frame out.
    const int frames = static_cast<int>(std::max<long long>(0, last - first + 1));
    const int waiting = batch - frames;
    std::fill(out, out + static_cast<std::size_t>(waiting) * kFrameSamples, 0.0f);
    Synthesize(first, last, out + static_cast<std::size_t>(waiting) * kFrameSamples);
    Emit(out, batch * kFrameSamples);
    features += static_cast<std::size_t>(batch) * kLatentDim;
    bandwidth_ids += batch;
    if (noise_weights) noise_weights += batch;
    if (concealed) concealed += batch;
    out += static_cast<std::size_t>(batch) * kFrameSamples;
    count -= batch;
  }
}

void StreamingDecoder::Flush(float* out) {
  const long long first = emitted_ + 1;
  const long long last = edge_;
  Compute(first, last);
  // Frame f leaves the ISTFT at stream sample 320 (f + L). A transmission
  // shorter than the lookahead leaves a silent gap before that.
  const long long gap = (first + lookahead_) * kFrameSamples - samples_out_;
  std::fill(out, out + gap, 0.0f);
  Emit(out, static_cast<int>(gap));
  out += gap;
  const int frames = static_cast<int>(std::max<long long>(0, last - first + 1));
  Synthesize(first, last, out);
  Emit(out, frames * kFrameSamples);
  out += static_cast<std::size_t>(frames) * kFrameSamples;
  // The tail the last frames left in the overlap-add buffer.
  for (int i = 0; i < kIstftPad; ++i) {
    out[i] = envelope_[i] > 1e-11f ? ola_[i] / envelope_[i] : 0.0f;
  }
  Emit(out, kIstftPad);
  Reset();
}

}  // namespace vocoscodec
