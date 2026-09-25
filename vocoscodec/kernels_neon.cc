// NEON kernels for arm64 (every arm64 CPU has NEON and half conversion).

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>

#include "vocoscodec/half.h"
#include "vocoscodec/kernels_internal.h"

namespace vocoscodec {
namespace kernels {
namespace {

// Four weights as floats, from float or half storage.
inline float32x4_t Load4(const float* w) { return vld1q_f32(w); }
inline float32x4_t Load4(const std::uint16_t* w) {
  return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(w)));
}
inline float Weight(const float* w) { return *w; }
inline float Weight(const std::uint16_t* w) { return HalfToFloat(*w); }

// ---------------------------------------------------------------- Dense
template <typename W, int NO, int NT>
void Block(const W* w, std::size_t w_stride, const float* bias, int depth, const float* in,
           std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  float32x4_t acc[NO][NT];
  for (int o = 0; o < NO; ++o)
    for (int t = 0; t < NT; ++t) acc[o][t] = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= depth; i += 4) {
    float32x4_t x[NT];
    for (int t = 0; t < NT; ++t) x[t] = vld1q_f32(in + t * in_step + i);
    for (int o = 0; o < NO; ++o) {
      const float32x4_t row = Load4(w + o * w_stride + i);
      for (int t = 0; t < NT; ++t) acc[o][t] = vfmaq_f32(acc[o][t], row, x[t]);
    }
  }
  for (int o = 0; o < NO; ++o) {
    for (int t = 0; t < NT; ++t) {
      float sum = vaddvq_f32(acc[o][t]);
      for (int j = i; j < depth; ++j) sum += Weight(w + o * w_stride + j) * in[t * in_step + j];
      if (bias) sum += bias[o];
      float& y = out[t * out_stride + o];
      y = accumulate ? y + sum : sum;
    }
  }
}

template <typename W, int NO>
void Rows(int nt, const W* w, std::size_t w_stride, const float* bias, int depth,
          const float* in, std::size_t in_step, float* out, std::size_t out_stride,
          bool accumulate) {
  switch (nt) {
    case 4: Block<W, NO, 4>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 3: Block<W, NO, 3>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 2: Block<W, NO, 2>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate); break;
    default: Block<W, NO, 1>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate);
  }
}

template <typename W>
void DenseNeon(const W* w, std::size_t w_stride, const float* bias, int outputs, int depth,
               const float* in, std::size_t in_step, int steps, float* out,
               std::size_t out_stride, bool accumulate) {
  for (int o = 0; o < outputs; o += 4) {
    const int no = std::min(4, outputs - o);
    const W* wo = w + o * w_stride;
    const float* bo = bias ? bias + o : nullptr;
    for (int t = 0; t < steps; t += 4) {
      const int nt = std::min(4, steps - t);
      const float* x = in + t * in_step;
      float* y = out + t * out_stride + o;
      switch (no) {
        case 4: Rows<W, 4>(nt, wo, w_stride, bo, depth, x, in_step, y, out_stride, accumulate); break;
        case 3: Rows<W, 3>(nt, wo, w_stride, bo, depth, x, in_step, y, out_stride, accumulate); break;
        case 2: Rows<W, 2>(nt, wo, w_stride, bo, depth, x, in_step, y, out_stride, accumulate); break;
        default: Rows<W, 1>(nt, wo, w_stride, bo, depth, x, in_step, y, out_stride, accumulate);
      }
    }
  }
}

// ---------------------------------------------------------------- DenseT
// NT steps by NV vectors of 4 outputs, as in kernels_avx2.cc.
template <typename W, int NT, int NV>
void TileT(const W* w, int outputs, const float* bias, int depth, const float* in,
           std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  float32x4_t acc[NT][NV];
  for (int t = 0; t < NT; ++t)
    for (int v = 0; v < NV; ++v) acc[t][v] = vdupq_n_f32(0.0f);
  for (int i = 0; i < depth; ++i) {
    float32x4_t row[NV];
    for (int v = 0; v < NV; ++v) row[v] = Load4(w + static_cast<std::size_t>(i) * outputs + 4 * v);
    for (int t = 0; t < NT; ++t) {
      const float32x4_t x = vdupq_n_f32(in[t * in_step + i]);
      for (int v = 0; v < NV; ++v) acc[t][v] = vfmaq_f32(acc[t][v], row[v], x);
    }
  }
  for (int t = 0; t < NT; ++t) {
    for (int v = 0; v < NV; ++v) {
      float32x4_t value = acc[t][v];
      if (bias) value = vaddq_f32(value, vld1q_f32(bias + 4 * v));
      float* y = out + t * out_stride + 4 * v;
      if (accumulate) value = vaddq_f32(value, vld1q_f32(y));
      vst1q_f32(y, value);
    }
  }
}

template <typename W, int NV>
void StepsT(int nt, const W* w, int outputs, const float* bias, int depth, const float* in,
            std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  switch (nt) {
    case 6: TileT<W, 6, NV>(w, outputs, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 5: TileT<W, 5, NV>(w, outputs, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 4: TileT<W, 4, NV>(w, outputs, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 3: TileT<W, 3, NV>(w, outputs, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 2: TileT<W, 2, NV>(w, outputs, bias, depth, in, in_step, out, out_stride, accumulate); break;
    default: TileT<W, 1, NV>(w, outputs, bias, depth, in, in_step, out, out_stride, accumulate);
  }
}

template <typename W>
void DenseTNeon(const W* w, const float* bias, int outputs, int depth, const float* in,
                std::size_t in_step, int steps, float* out, std::size_t out_stride,
                bool accumulate) {
  const int vectors = outputs / 4;
  for (int o = 0; o < vectors * 4; o += 16) {
    const int nv = std::min(4, vectors - o / 4);
    const float* bo = bias ? bias + o : nullptr;
    for (int t = 0; t < steps; t += 6) {
      const int nt = std::min(6, steps - t);
      const float* x = in + t * in_step;
      float* y = out + t * out_stride + o;
      switch (nv) {
        case 4: StepsT<W, 4>(nt, w + o, outputs, bo, depth, x, in_step, y, out_stride, accumulate); break;
        case 3: StepsT<W, 3>(nt, w + o, outputs, bo, depth, x, in_step, y, out_stride, accumulate); break;
        case 2: StepsT<W, 2>(nt, w + o, outputs, bo, depth, x, in_step, y, out_stride, accumulate); break;
        default: StepsT<W, 1>(nt, w + o, outputs, bo, depth, x, in_step, y, out_stride, accumulate);
      }
    }
  }
  for (int o = vectors * 4; o < outputs; ++o) {
    for (int t = 0; t < steps; ++t) {
      const float* x = in + t * in_step;
      float sum = 0.0f;
      for (int i = 0; i < depth; ++i) sum += Weight(w + static_cast<std::size_t>(i) * outputs + o) * x[i];
      if (bias) sum += bias[o];
      float& y = out[t * out_stride + o];
      y = accumulate ? y + sum : sum;
    }
  }
}

// ---------------------------------------------------------------- activations
// exp(x) after Cephes expf, as in kernels_avx2.cc.
inline float32x4_t Exp(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
  const float32x4_t n =
      vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f)));
  x = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
  x = vfmsq_f32(x, n, vdupq_n_f32(-2.12194440e-4f));
  float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
  y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
  y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
  y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
  y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
  y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
  y = vfmaq_f32(vaddq_f32(x, vdupq_n_f32(1.0f)), y, vmulq_f32(x, x));
  const int32x4_t exponent = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(exponent));
}

inline float32x4_t Elu(float32x4_t v) {
  const float32x4_t negative = vsubq_f32(Exp(v), vdupq_n_f32(1.0f));
  return vbslq_f32(vcltq_f32(v, vdupq_n_f32(0.0f)), negative, v);
}

inline float32x4_t Gelu(float32x4_t v) {
  const float32x4_t z = vmulq_f32(v, vdupq_n_f32(0.70710678118654752f));
  const float32x4_t a = vabsq_f32(z);
  const float32x4_t t =
      vdivq_f32(vdupq_n_f32(1.0f), vfmaq_f32(vdupq_n_f32(1.0f), a, vdupq_n_f32(kErfP)));
  float32x4_t p = vdupq_n_f32(kErfA5);
  p = vfmaq_f32(vdupq_n_f32(kErfA4), p, t);
  p = vfmaq_f32(vdupq_n_f32(kErfA3), p, t);
  p = vfmaq_f32(vdupq_n_f32(kErfA2), p, t);
  p = vfmaq_f32(vdupq_n_f32(kErfA1), p, t);
  p = vmulq_f32(p, t);
  const float32x4_t e = Exp(vnegq_f32(vmulq_f32(a, a)));
  float32x4_t erf = vfmsq_f32(vdupq_n_f32(1.0f), p, e);
  erf = vbslq_f32(vcltq_f32(z, vdupq_n_f32(0.0f)), vnegq_f32(erf), erf);
  return vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), v), vaddq_f32(vdupq_n_f32(1.0f), erf));
}

template <float32x4_t (*F)(float32x4_t)>
void Apply(float* x, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(x + i, F(vld1q_f32(x + i)));
  if (i < n) {
    float lane[4] = {};
    std::copy(x + i, x + n, lane);
    vst1q_f32(lane, F(vld1q_f32(lane)));
    std::copy(lane, lane + (n - i), x + i);
  }
}

}  // namespace

const Table kNeon = {"neon",
                     true,
                     DenseNeon<float>,
                     DenseTNeon<float>,
                     DenseNeon<std::uint16_t>,
                     DenseTNeon<std::uint16_t>,
                     Apply<Elu>,
                     Apply<Gelu>};

}  // namespace kernels
}  // namespace vocoscodec
