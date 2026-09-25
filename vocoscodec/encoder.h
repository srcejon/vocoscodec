// EnCodec 24 kHz encoder and residual vector quantiser, streaming.

#ifndef VOCOSCODEC_ENCODER_H_
#define VOCOSCODEC_ENCODER_H_

#include <filesystem>
#include <memory>
#include <vector>

#include "vocoscodec/weights.h"

namespace vocoscodec {

inline constexpr int kSampleRate = 24000;
inline constexpr int kFrameSamples = 320;
inline constexpr int kMaxCodebooks = 16;
inline constexpr int kCodebookEntries = 1024;
inline constexpr int kLatentDim = 128;

// The codebooks (encodec_codebooks.bin), shared by encoder and decoder.
class Codebooks {
 public:
  explicit Codebooks(const std::filesystem::path& path);
  // Entry `index` of codebook `book`: kLatentDim floats.
  const float* Entry(int book, int index) const {
    return data_.data() + (static_cast<std::size_t>(book) * kCodebookEntries + index) * kLatentDim;
  }
  const float* Book(int book) const { return Entry(book, 0); }
  const float* Norms(int book) const { return norms_.data() + book * kCodebookEntries; }
  std::size_t bytes() const { return (data_.size() + norms_.size()) * sizeof(float); }

 private:
  std::vector<float> data_;
  std::vector<float> norms_;  // |e|^2 of every entry
};

// A convolution. Those over many time steps per frame are held [tap][in][out]
// for kernels::DenseT; those at the frame rate (one step) [out][tap][in] for
// kernels::Dense, which streams long rows faster. depth = tap * in + i.
struct ConvWeights {
  Weights w;
  std::vector<float> b;
  int in = 0, out = 0, taps = 0, stride = 1;
  bool transposed = true;
};

// The encoder network (encodec_encoder.bin).
class EncoderModel {
 public:
  explicit EncoderModel(const std::filesystem::path& path);

  struct Stage {
    ConvWeights conv1, conv2, skip, down;
  };
  ConvWeights conv_in, conv_out;
  Stage stages[4];
  struct Lstm {
    Weights w_ih, w_hh;  // [2048][512]
    std::vector<float> b;
  } lstm[2];
  std::size_t bytes() const;
};

// One causal convolution with its input history.
class CausalConv {
 public:
  void Init(const ConvWeights* weights, int max_input_steps);
  // `steps` input rows (a multiple of the stride) -> steps / stride output
  // rows. With elu the ELU is applied to the input first.
  void Run(const float* input, int steps, float* output, bool elu, bool accumulate = false);
  void Reset();

 private:
  const ConvWeights* weights_ = nullptr;
  int history_ = 0;
  std::vector<float> buffer_;  // history rows, then this call's input
};

class StreamingEncoder {
 public:
  StreamingEncoder(std::shared_ptr<const EncoderModel> model,
                   std::shared_ptr<const Codebooks> codebooks, int codebooks_used);

  // 320 samples in [-1, 1) -> one index per codebook, coarse first.
  void Encode(const float* pcm, int* indices);
  // The latent before quantisation (tests).
  void Embed(const float* pcm, float* latent);
  void Quantize(const float* latent, int* indices) const;
  void Reset();
  int codebooks() const { return codebooks_; }

 private:
  std::shared_ptr<const EncoderModel> model_;
  std::shared_ptr<const Codebooks> books_;
  int codebooks_;
  CausalConv conv_in_, conv_out_;
  struct Stage {
    CausalConv conv1, conv2, skip, down;
  } stages_[4];
  std::vector<float> x_, e_, h_, r_;
  std::vector<float> lstm_h_[2], lstm_c_[2];
  std::vector<float> gates_;
  mutable std::vector<float> scores_;
};

}  // namespace vocoscodec

#endif  // VOCOSCODEC_ENCODER_H_
