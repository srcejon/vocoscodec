// SSE4.2 kernels for x86-64 CPUs without AVX2: 4-wide vectors, separate
// multiply and add (no FMA), half weights converted in integer SIMD (no F16C).
// Only called after kernels.cc has checked the CPU.

#include <nmmintrin.h>  // SSE4.2 (and SSE4.1, SSSE3, SSE3, SSE2)

#include <algorithm>
#include <cstdint>

#include "vocoscodec/half.h"
#include "vocoscodec/kernels_internal.h"

namespace vocoscodec {
namespace kernels {
namespace {

// Four finite halves as floats, without F16C: each half moved to the top of
// its lane, the exponent and mantissa shifted into float position and
// rebiased by multiplying by 2^112 (which also normalises subnormals), the
// sign put back. After Fabian Giesen's half_to_float_fast. Weights are
// finite, so infinities and NaNs are not handled.
inline __m128 Load4(const std::uint16_t* w) {
  const __m128i h = _mm_unpacklo_epi16(
      _mm_setzero_si128(), _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w)));
  const __m128i magnitude = _mm_srli_epi32(_mm_slli_epi32(h, 1), 4);
  const __m128 f =
      _mm_mul_ps(_mm_castsi128_ps(magnitude), _mm_castsi128_ps(_mm_set1_epi32(0x77800000)));
  const __m128i sign = _mm_and_si128(h, _mm_set1_epi32(static_cast<int>(0x80000000u)));
  return _mm_or_ps(f, _mm_castsi128_ps(sign));
}

// Four weights as floats, from float storage.
inline __m128 Load4(const float* w) { return _mm_loadu_ps(w); }
inline float Weight(const float* w) { return *w; }
inline float Weight(const std::uint16_t* w) { return HalfToFloat(*w); }

inline float HorizontalSum(__m128 v) {
  __m128 shuffled = _mm_movehdup_ps(v);
  __m128 sums = _mm_add_ps(v, shuffled);
  shuffled = _mm_movehl_ps(shuffled, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, shuffled));
}

// ---------------------------------------------------------------- Dense
// NO output rows by NT time steps: at most 4 x 2, so that accumulators,
// inputs and weights fit the 16 XMM registers.
template <typename W, int NO, int NT>
void Block(const W* w, std::size_t w_stride, const float* bias, int depth, const float* in,
           std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  __m128 acc[NO][NT];
  for (int o = 0; o < NO; ++o)
    for (int t = 0; t < NT; ++t) acc[o][t] = _mm_setzero_ps();
  int i = 0;
  for (; i + 4 <= depth; i += 4) {
    __m128 x[NT];
    for (int t = 0; t < NT; ++t) x[t] = _mm_loadu_ps(in + t * in_step + i);
    for (int o = 0; o < NO; ++o) {
      const __m128 row = Load4(w + o * w_stride + i);
      for (int t = 0; t < NT; ++t) acc[o][t] = _mm_add_ps(acc[o][t], _mm_mul_ps(row, x[t]));
    }
  }
  for (int o = 0; o < NO; ++o) {
    for (int t = 0; t < NT; ++t) {
      float sum = HorizontalSum(acc[o][t]);
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
  if (nt == 2) {
    Block<W, NO, 2>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate);
  } else {
    Block<W, NO, 1>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate);
  }
}

template <typename W>
void DenseSse(const W* w, std::size_t w_stride, const float* bias, int outputs, int depth,
              const float* in, std::size_t in_step, int steps, float* out,
              std::size_t out_stride, bool accumulate) {
  for (int o = 0; o < outputs; o += 4) {
    const int no = std::min(4, outputs - o);
    const W* wo = w + o * w_stride;
    const float* bo = bias ? bias + o : nullptr;
    for (int t = 0; t < steps; t += 2) {
      const int nt = std::min(2, steps - t);
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
// Transposed weights: NT steps by NV vectors of 4 outputs. Tiles span 16
// outputs, a whole 64-byte line of each weight row, over 2 steps (8
// accumulators, 4 weight vectors and a broadcast input in the 16 registers);
// narrower tails take up to 6 steps.
template <typename W, int NT, int NV>
void TileT(const W* w, int outputs, const float* bias, int depth, const float* in,
           std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  __m128 acc[NT][NV];
  for (int t = 0; t < NT; ++t)
    for (int v = 0; v < NV; ++v) acc[t][v] = _mm_setzero_ps();
  for (int i = 0; i < depth; ++i) {
    __m128 row[NV];
    for (int v = 0; v < NV; ++v) row[v] = Load4(w + static_cast<std::size_t>(i) * outputs + 4 * v);
    for (int t = 0; t < NT; ++t) {
      const __m128 x = _mm_set1_ps(in[t * in_step + i]);
      for (int v = 0; v < NV; ++v) acc[t][v] = _mm_add_ps(acc[t][v], _mm_mul_ps(row[v], x));
    }
  }
  for (int t = 0; t < NT; ++t) {
    for (int v = 0; v < NV; ++v) {
      __m128 value = acc[t][v];
      if (bias) value = _mm_add_ps(value, _mm_loadu_ps(bias + 4 * v));
      float* y = out + t * out_stride + 4 * v;
      if (accumulate) value = _mm_add_ps(value, _mm_loadu_ps(y));
      _mm_storeu_ps(y, value);
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
void DenseTSse(const W* w, const float* bias, int outputs, int depth, const float* in,
               std::size_t in_step, int steps, float* out, std::size_t out_stride,
               bool accumulate) {
  const int vectors = outputs / 4;
  for (int o = 0; o < vectors * 4; o += 16) {
    const float* bo = bias ? bias + o : nullptr;
    const int nv = std::min(4, vectors - o / 4);
    const int most = nv >= 3 ? 2 : 6;
    for (int t = 0; t < steps; t += most) {
      const int nt = std::min(most, steps - t);
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
// exp(x) after Cephes expf, as in kernels_avx2.cc, without FMA.
inline __m128 Exp(__m128 x) {
  x = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-87.3f)), _mm_set1_ps(88.3f));
  const __m128 n = _mm_floor_ps(
      _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f)), _mm_set1_ps(0.5f)));
  x = _mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(0.693359375f)));
  x = _mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(-2.12194440e-4f)));
  __m128 y = _mm_set1_ps(1.9875691500e-4f);
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.3981999507e-3f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(8.3334519073e-3f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(4.1665795894e-2f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.6666665459e-1f));
  y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(5.0000001201e-1f));
  y = _mm_add_ps(_mm_mul_ps(y, _mm_mul_ps(x, x)), _mm_add_ps(x, _mm_set1_ps(1.0f)));
  const __m128i exponent =
      _mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(n), _mm_set1_epi32(127)), 23);
  return _mm_mul_ps(y, _mm_castsi128_ps(exponent));
}

inline __m128 Elu(__m128 v) {
  const __m128 negative = _mm_sub_ps(Exp(v), _mm_set1_ps(1.0f));
  return _mm_blendv_ps(v, negative, _mm_cmplt_ps(v, _mm_setzero_ps()));
}

inline __m128 Gelu(__m128 v) {
  const __m128 z = _mm_mul_ps(v, _mm_set1_ps(0.70710678118654752f));
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 a = _mm_andnot_ps(sign_mask, z);
  const __m128 t = _mm_div_ps(_mm_set1_ps(1.0f),
                              _mm_add_ps(_mm_mul_ps(a, _mm_set1_ps(kErfP)), _mm_set1_ps(1.0f)));
  __m128 p = _mm_set1_ps(kErfA5);
  p = _mm_add_ps(_mm_mul_ps(p, t), _mm_set1_ps(kErfA4));
  p = _mm_add_ps(_mm_mul_ps(p, t), _mm_set1_ps(kErfA3));
  p = _mm_add_ps(_mm_mul_ps(p, t), _mm_set1_ps(kErfA2));
  p = _mm_add_ps(_mm_mul_ps(p, t), _mm_set1_ps(kErfA1));
  p = _mm_mul_ps(p, t);
  const __m128 e = Exp(_mm_sub_ps(_mm_setzero_ps(), _mm_mul_ps(a, a)));
  __m128 erf = _mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(p, e));
  erf = _mm_or_ps(erf, _mm_and_ps(z, sign_mask));  // erf is odd
  return _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.5f), v), _mm_add_ps(_mm_set1_ps(1.0f), erf));
}

template <__m128 (*F)(__m128)>
void Apply(float* x, std::size_t n) {
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) _mm_storeu_ps(x + i, F(_mm_loadu_ps(x + i)));
  if (i < n) {
    alignas(16) float lane[4] = {};
    std::copy(x + i, x + n, lane);
    _mm_store_ps(lane, F(_mm_load_ps(lane)));
    std::copy(lane, lane + (n - i), x + i);
  }
}

}  // namespace

const Table kSse42 = {"sse4.2",
                      false,
                      DenseSse<float>,
                      DenseTSse<float>,
                      DenseSse<std::uint16_t>,
                      DenseTSse<std::uint16_t>,
                      Apply<Elu>,
                      Apply<Gelu>};

}  // namespace kernels
}  // namespace vocoscodec
