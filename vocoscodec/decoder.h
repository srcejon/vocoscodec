// Vocos decoder (backbone, ISTFT head), streaming with bounded lookahead.
//
// Output lags the newest frame received by L frames (the lookahead). Frame t
// is decoded as the whole-signal decoder would decode the frames received
// so far followed by nothing: every layer sees zero beyond the newest frame,
// as at the end of a signal. One packet per call gives every frame exactly
// L frames of lookahead. Frames early in a group given together see up to
// the group's end, which only helps. Truncating the right context like this
// costs about 0.05 PESQ at L = 3. Masking each layer's future taps instead
// costs about 0.7 (validation/README.md).
//
// Every one of the nine k=7 convolutions widens the receptive field by 3.
// Layer k's value at position p therefore no longer depends on the right
// edge E once p <= E - 3(k + 1). Those values are kept. Each call recomputes
// only the values near the new edge that its output frames need.

#ifndef VOCOSCODEC_DECODER_H_
#define VOCOSCODEC_DECODER_H_

#include <array>
#include <complex>
#include <filesystem>
#include <memory>
#include <vector>

#include "vocoscodec/concealment.h"
#include "vocoscodec/encoder.h"
#include "vocoscodec/fft.h"
#include "vocoscodec/weights.h"

namespace vocoscodec {

inline constexpr int kDim = 384;
inline constexpr int kIntermediate = 1152;
inline constexpr int kBlocks = 8;
inline constexpr int kLayers = kBlocks + 1;  // the k=7 convolutions
inline constexpr int kReceptiveField = 3 * kLayers;  // 27 frames each side
inline constexpr int kBandwidths = 4;
inline constexpr int kFftSize = 1280;
inline constexpr int kBins = kFftSize / 2 + 1;
inline constexpr int kIstftPad = (kFftSize - kFrameSamples) / 2;  // 480

// The decoder network (vocos_decoder.bin).
class DecoderModel {
 public:
  explicit DecoderModel(const std::filesystem::path& path);

  // Linear and embedding weights are transposed at load ([in][out], the
  // layout of kernels::DenseT) and held in half precision if configured.
  Weights embed_w;  // [7][128][384]
  std::vector<float> embed_b;
  std::vector<float> norm_scale, norm_shift;  // [4][384]
  struct Block {
    std::vector<float> dw_w;  // [7][384], transposed at load
    std::vector<float> dw_b;
    std::vector<float> norm_scale, norm_shift;
    Weights pw1_w;  // [384][1152]
    std::vector<float> pw1_b;
    Weights pw2_w;  // [1152][384], layer scale folded in
    std::vector<float> pw2_b;
  } blocks[kBlocks];
  std::vector<float> final_w, final_b;
  Weights head_w;  // [384][1282]
  std::vector<float> head_b;
  std::size_t bytes() const;
};

class StreamingDecoder {
 public:
  // lookahead: frames, 0..27. comfort_noise: what concealment fades to
  // (otherwise silence).
  StreamingDecoder(std::shared_ptr<const DecoderModel> model, int lookahead,
                   bool comfort_noise = true);

  // `count` frames of features ([count][128]) with their bandwidth ids ->
  // count * 320 samples, delay() samples behind. Samples before the first
  // frame's signal time are zero. noise_weights (null: all 0) mixes each
  // frame's spectrum with comfort noise. A frame with a null `concealed`, or
  // with concealed[j] false, came from a packet and updates the noise floor.
  void Run(const float* features, const int* bandwidth_ids, int count, float* out,
           const float* noise_weights = nullptr, const std::uint8_t* concealed = nullptr);
  // The remaining delay() samples, as if the signal ended after the last frame
  // (the whole-signal decoder's zero padding). Then resets.
  void Flush(float* out);
  void Reset();
  int delay() const { return delay_; }
  int lookahead() const { return lookahead_; }
  // True when the last frame synthesized was comfort noise (or silence) alone.
  bool is_comfort_noise() const { return comfort_noise_only_; }
  // ConvNeXt blocks evaluated so far (for cost reports).
  long long block_evaluations() const { return block_evaluations_; }

 private:
  static constexpr int kMaxBatch = 32;
  // Rows kept per layer. A call needs at most batch + lookahead + 38 of them
  // (97): the frames it adds, those still to be synthesized, and the
  // receptive field of the next frame out.
  static constexpr int kCapacity = 128;

  struct Range {
    long long lo, hi;  // inclusive, empty when hi < lo
  };

  float* Row(int layer, long long position);  // layer -1: features
  void Compact();
  void Advance(const float* features, const int* ids, const float* noise_weights,
               const std::uint8_t* concealed, int count);
  void Compute(long long first_out, long long last_out);
  void ComputeEmbed(Range range);
  // Both ranges (either may be empty) in one pass over the block's weights.
  void ComputeBlock(int block, Range first, Range second);
  void Synthesize(long long first, long long last, float* out);
  void Emit(float* out, int samples);

  std::shared_ptr<const DecoderModel> model_;
  int lookahead_;
  int delay_;

  // Rows of positions base_ .. base_ + kCapacity - 1 for the features and each
  // layer, with the bandwidth id of each position.
  long long base_ = 0;
  std::vector<float> features_;
  std::array<std::vector<float>, kLayers> layers_;
  std::vector<int> ids_;
  std::vector<float> noise_weights_;
  std::vector<std::uint8_t> concealed_;
  long long edge_ = -1;  // last frame received (E)
  std::array<long long, kLayers> final_{};  // layer k is final up to here
  long long emitted_ = -1;  // last frame through the ISTFT

  std::vector<float> y_, z_, residual_, hidden_, head_;
  std::vector<long long> positions_;
  InverseRealFft fft_;
  std::vector<float> window_, window_sq_, frame_, ola_, envelope_;
  std::vector<std::complex<float>> spectrum_;
  std::vector<float> magnitudes_;
  ComfortNoise comfort_noise_;
  bool comfort_noise_only_ = false;
  long long samples_out_ = 0;
  long long block_evaluations_ = 0;
};

}  // namespace vocoscodec

#endif  // VOCOSCODEC_DECODER_H_
