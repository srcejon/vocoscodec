// A weight matrix held in float or IEEE half precision.
//
// With VOCOSCODEC_HALF_WEIGHTS (the default), the layers whose weights are
// read once per frame are held in half precision. These are the encoder's
// LSTM and frame-rate convolutions and the decoder's linear layers. That
// halves their memory and memory traffic, which bounds their speed. It
// changes 0.6% of encoder indices and no decoder PESQ measurably
// (tools/reference/fp16_weights.py). The model files stay float32.
//
// That applies only where the kernels convert half to float in hardware
// (AVX2 with F16C, NEON). With the SSE4.2 or portable kernels the conversion
// costs more than the traffic saved (SSE4.2 on an i9-14900K: encoding 3.0
// against 2.1 ms per frame), so there the weights stay float. Every kernel
// set handles both, so the choice at load time affects speed, not results
// beyond the rounding.

#ifndef VOCOSCODEC_WEIGHTS_H_
#define VOCOSCODEC_WEIGHTS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vocoscodec/half.h"
#include "vocoscodec/kernels.h"

namespace vocoscodec {

#if defined(VOCOSCODEC_HALF_WEIGHTS)
inline constexpr bool kHalfWeights = true;
#else
inline constexpr bool kHalfWeights = false;
#endif

// Whether models loaded now hold their per-frame layers in half precision.
inline bool UseHalfWeights() { return kHalfWeights && kernels::HardwareHalf(); }

class Weights {
 public:
  void Assign(std::vector<float> values, bool half) {
    if (half) {
      f16_.resize(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) f16_[i] = FloatToHalf(values[i]);
      f32_.clear();
    } else {
      f32_ = std::move(values);
      f16_.clear();
    }
  }

  // kernels::Dense with these weights ([outputs][w_stride]).
  void Dense(std::size_t w_stride, const float* bias, int outputs, int depth, const float* in,
             std::size_t in_step, int steps, float* out, std::size_t out_stride,
             bool accumulate) const {
    if (f16_.empty()) {
      kernels::Dense(f32_.data(), w_stride, bias, outputs, depth, in, in_step, steps, out,
                     out_stride, accumulate);
    } else {
      kernels::Dense(f16_.data(), w_stride, bias, outputs, depth, in, in_step, steps, out,
                     out_stride, accumulate);
    }
  }

  // kernels::DenseT with these weights ([depth][outputs]).
  void DenseT(const float* bias, int outputs, int depth, const float* in, std::size_t in_step,
              int steps, float* out, std::size_t out_stride, bool accumulate) const {
    if (f16_.empty()) {
      kernels::DenseT(f32_.data(), bias, outputs, depth, in, in_step, steps, out, out_stride,
                      accumulate);
    } else {
      kernels::DenseT(f16_.data(), bias, outputs, depth, in, in_step, steps, out, out_stride,
                      accumulate);
    }
  }

  std::size_t bytes() const { return f32_.size() * 4 + f16_.size() * 2; }

 private:
  std::vector<float> f32_;
  std::vector<std::uint16_t> f16_;
};

}  // namespace vocoscodec

#endif  // VOCOSCODEC_WEIGHTS_H_
