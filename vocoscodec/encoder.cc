#include "vocoscodec/encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vocoscodec/kernels.h"
#include "vocoscodec/model_file.h"

namespace vocoscodec {
namespace {

constexpr int kStrides[4] = {2, 4, 5, 8};
constexpr int kFilters = 32;
constexpr int kLstmDim = 512;

std::vector<float> Copy(const Tensor& t) { return std::vector<float>(t.data, t.data + t.size); }

// Loads a convolution [out][in][taps], reordered to [tap][in][out]
// (transposed, float) or [out][tap][in] (frame rate, half if configured).
void LoadConv(ModelFile& file, const std::string& name, std::uint32_t out, std::uint32_t in,
              std::uint32_t taps, ConvWeights* conv, int stride = 1, bool frame_rate = false) {
  const Tensor& w = file.Get(name + ".w", {out, in, taps});
  std::vector<float> reordered(w.size);
  for (std::uint32_t o = 0; o < out; ++o)
    for (std::uint32_t i = 0; i < in; ++i)
      for (std::uint32_t k = 0; k < taps; ++k) {
        const std::size_t depth = static_cast<std::size_t>(k) * in + i;
        reordered[frame_rate ? o * taps * in + depth : depth * out + o] =
            w.data[(static_cast<std::size_t>(o) * in + i) * taps + k];
      }
  conv->w.Assign(std::move(reordered), frame_rate && UseHalfWeights());
  conv->b = Copy(file.Get(name + ".b", {out}));
  conv->in = static_cast<int>(in);
  conv->out = static_cast<int>(out);
  conv->taps = static_cast<int>(taps);
  conv->stride = stride;
  conv->transposed = !frame_rate;
}

}  // namespace

Codebooks::Codebooks(const std::filesystem::path& path) {
  ModelFile file(path);
  data_ = Copy(file.Get("codebooks", {kMaxCodebooks, kCodebookEntries, kLatentDim}));
  norms_.resize(kMaxCodebooks * kCodebookEntries);
  for (int e = 0; e < kMaxCodebooks * kCodebookEntries; ++e) {
    const float* entry = data_.data() + static_cast<std::size_t>(e) * kLatentDim;
    float sum = 0.0f;
    for (int i = 0; i < kLatentDim; ++i) sum += entry[i] * entry[i];
    norms_[e] = sum;
  }
}

EncoderModel::EncoderModel(const std::filesystem::path& path) {
  ModelFile file(path);
  LoadConv(file, "conv_in", kFilters, 1, 7, &conv_in);
  int channels = kFilters;
  for (int s = 0; s < 4; ++s) {
    const std::string p = "res" + std::to_string(s) + ".";
    const auto c = static_cast<std::uint32_t>(channels);
    LoadConv(file, p + "conv1", c / 2, c, 3, &stages[s].conv1);
    LoadConv(file, p + "conv2", c, c / 2, 1, &stages[s].conv2);
    LoadConv(file, p + "skip", c, c, 1, &stages[s].skip);
    // The last downsampling gives one step per frame.
    LoadConv(file, "down" + std::to_string(s), 2 * c, c, 2 * kStrides[s], &stages[s].down,
             kStrides[s], s == 3);
    channels *= 2;
  }
  for (int l = 0; l < 2; ++l) {
    const std::string p = "lstm" + std::to_string(l) + ".";
    lstm[l].w_ih.Assign(Copy(file.Get(p + "w_ih", {4 * kLstmDim, kLstmDim})), UseHalfWeights());
    lstm[l].w_hh.Assign(Copy(file.Get(p + "w_hh", {4 * kLstmDim, kLstmDim})), UseHalfWeights());
    lstm[l].b = Copy(file.Get(p + "b", {4 * kLstmDim}));
  }
  LoadConv(file, "conv_out", kLatentDim, kLstmDim, 7, &conv_out, 1, true);
}

std::size_t EncoderModel::bytes() const {
  std::size_t total = 0;
  const auto add = [&total](const ConvWeights& c) { total += c.w.bytes() + c.b.size() * 4; };
  add(conv_in);
  add(conv_out);
  for (const Stage& s : stages) {
    add(s.conv1);
    add(s.conv2);
    add(s.skip);
    add(s.down);
  }
  for (const Lstm& l : lstm) total += l.w_ih.bytes() + l.w_hh.bytes() + l.b.size() * 4;
  return total;
}

void CausalConv::Init(const ConvWeights* weights, int max_input_steps) {
  weights_ = weights;
  history_ = weights->taps - weights->stride;
  buffer_.assign(static_cast<std::size_t>(history_ + max_input_steps) * weights->in, 0.0f);
}

void CausalConv::Reset() { std::fill(buffer_.begin(), buffer_.end(), 0.0f); }

void CausalConv::Run(const float* input, int steps, float* output, bool elu,
                     bool accumulate) {
  const ConvWeights& c = *weights_;
  const int in = c.in;
  float* fresh = buffer_.data() + static_cast<std::size_t>(history_) * in;
  std::memcpy(fresh, input, sizeof(float) * steps * in);
  if (elu) kernels::Elu(fresh, static_cast<std::size_t>(steps) * in);
  const int depth = c.taps * in;
  const std::size_t step = static_cast<std::size_t>(c.stride) * in;
  if (c.transposed) {
    c.w.DenseT(c.b.data(), c.out, depth, buffer_.data(), step, steps / c.stride, output, c.out,
               accumulate);
  } else {
    c.w.Dense(depth, c.b.data(), c.out, depth, buffer_.data(), step, steps / c.stride, output,
              c.out, accumulate);
  }
  if (history_) {
    std::memmove(buffer_.data(), buffer_.data() + static_cast<std::size_t>(steps) * in,
                 sizeof(float) * history_ * in);
  }
}

StreamingEncoder::StreamingEncoder(std::shared_ptr<const EncoderModel> model,
                                   std::shared_ptr<const Codebooks> codebooks,
                                   int codebooks_used)
    : model_(std::move(model)), books_(std::move(codebooks)), codebooks_(codebooks_used) {
  if (codebooks_ < 1 || codebooks_ > kMaxCodebooks) {
    throw std::invalid_argument("codebook count must be 1.." + std::to_string(kMaxCodebooks));
  }
  int steps = kFrameSamples;
  conv_in_.Init(&model_->conv_in, steps);
  for (int s = 0; s < 4; ++s) {
    stages_[s].conv1.Init(&model_->stages[s].conv1, steps);
    stages_[s].conv2.Init(&model_->stages[s].conv2, steps);
    stages_[s].skip.Init(&model_->stages[s].skip, steps);
    stages_[s].down.Init(&model_->stages[s].down, steps);
    steps /= kStrides[s];
  }
  conv_out_.Init(&model_->conv_out, 1);
  const std::size_t largest = kFrameSamples * kFilters;  // every level holds 10240 values
  x_.resize(largest);
  e_.resize(largest);
  h_.resize(largest);
  r_.resize(largest);
  for (int l = 0; l < 2; ++l) {
    lstm_h_[l].assign(kLstmDim, 0.0f);
    lstm_c_[l].assign(kLstmDim, 0.0f);
  }
  gates_.resize(4 * kLstmDim);
  scores_.resize(kCodebookEntries);
}

void StreamingEncoder::Reset() {
  conv_in_.Reset();
  conv_out_.Reset();
  for (auto& stage : stages_) {
    stage.conv1.Reset();
    stage.conv2.Reset();
    stage.skip.Reset();
    stage.down.Reset();
  }
  for (int l = 0; l < 2; ++l) {
    std::fill(lstm_h_[l].begin(), lstm_h_[l].end(), 0.0f);
    std::fill(lstm_c_[l].begin(), lstm_c_[l].end(), 0.0f);
  }
}

namespace {
inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

void StreamingEncoder::Embed(const float* pcm, float* latent) {
  int steps = kFrameSamples;
  conv_in_.Run(pcm, steps, x_.data(), false);
  for (int s = 0; s < 4; ++s) {
    Stage& stage = stages_[s];
    // x = skip(x) + conv2(elu(conv1(elu(x))))
    stage.conv1.Run(x_.data(), steps, h_.data(), true);
    stage.skip.Run(x_.data(), steps, r_.data(), false);
    stage.conv2.Run(h_.data(), steps, r_.data(), true, true);
    stage.down.Run(r_.data(), steps, x_.data(), true);
    steps /= kStrides[s];
  }
  // Two LSTM layers over the single frame, plus the skip connection.
  const float* input = x_.data();
  for (int l = 0; l < 2; ++l) {
    const auto& lstm = model_->lstm[l];
    lstm.w_ih.Dense(kLstmDim, lstm.b.data(), 4 * kLstmDim, kLstmDim, input, 0, 1,
                    gates_.data(), 0, false);
    lstm.w_hh.Dense(kLstmDim, nullptr, 4 * kLstmDim, kLstmDim, lstm_h_[l].data(), 0, 1,
                    gates_.data(), 0, true);
    float* c = lstm_c_[l].data();
    float* h = lstm_h_[l].data();
    for (int i = 0; i < kLstmDim; ++i) {  // gate order: input, forget, cell, output
      const float gi = Sigmoid(gates_[i]);
      const float gf = Sigmoid(gates_[kLstmDim + i]);
      const float gg = std::tanh(gates_[2 * kLstmDim + i]);
      const float go = Sigmoid(gates_[3 * kLstmDim + i]);
      c[i] = gf * c[i] + gi * gg;
      h[i] = go * std::tanh(c[i]);
    }
    input = h;
  }
  for (int i = 0; i < kLstmDim; ++i) e_[i] = lstm_h_[1][i] + x_[i];
  conv_out_.Run(e_.data(), 1, latent, true);
}

void StreamingEncoder::Quantize(const float* latent, int* indices) const {
  float residual[kLatentDim];
  std::copy(latent, latent + kLatentDim, residual);
  for (int q = 0; q < codebooks_; ++q) {
    kernels::Dense(books_->Book(q), kLatentDim, nullptr, kCodebookEntries, kLatentDim,
                   residual, 0, 1, scores_.data(), 0, false);
    // Nearest entry: argmin |e|^2 - 2 r.e, the first on ties.
    const float* norms = books_->Norms(q);
    int best = 0;
    float best_score = norms[0] - 2.0f * scores_[0];
    for (int e = 1; e < kCodebookEntries; ++e) {
      const float score = norms[e] - 2.0f * scores_[e];
      if (score < best_score) {
        best_score = score;
        best = e;
      }
    }
    indices[q] = best;
    const float* entry = books_->Entry(q, best);
    for (int i = 0; i < kLatentDim; ++i) residual[i] -= entry[i];
  }
}

void StreamingEncoder::Encode(const float* pcm, int* indices) {
  float latent[kLatentDim];
  Embed(pcm, latent);
  Quantize(latent, indices);
}

}  // namespace vocoscodec
