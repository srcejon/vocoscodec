// AVX2 + FMA + F16C kernels. This file alone is compiled with those
// instructions enabled; it is only called after kernels.cc has checked the CPU.

#include <immintrin.h>

#include <algorithm>
#include <cstdint>

#include "vocoscodec/half.h"
#include "vocoscodec/kernels_internal.h"

namespace vocoscodec {
namespace kernels {
namespace {

// Eight weights as floats, from float or half storage.
inline __m256 Load8(const float* w) { return _mm256_loadu_ps(w); }
inline __m256 Load8(const std::uint16_t* w) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
}
inline float Weight(const float* w) { return *w; }
inline float Weight(const std::uint16_t* w) { return HalfToFloat(*w); }

inline float HorizontalSum(__m256 v) {
  __m128 low = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  __m128 shuffled = _mm_movehdup_ps(low);
  __m128 sums = _mm_add_ps(low, shuffled);
  shuffled = _mm_movehl_ps(shuffled, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, shuffled));
}

// ---------------------------------------------------------------- Dense
// NO output rows by NT time steps, accumulated in NO * NT registers.
template <typename W, int NO, int NT>
void Block(const W* w, std::size_t w_stride, const float* bias, int depth, const float* in,
           std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  __m256 acc[NO][NT];
  for (int o = 0; o < NO; ++o)
    for (int t = 0; t < NT; ++t) acc[o][t] = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= depth; i += 8) {
    __m256 x[NT];
    for (int t = 0; t < NT; ++t) x[t] = _mm256_loadu_ps(in + t * in_step + i);
    for (int o = 0; o < NO; ++o) {
      const __m256 row = Load8(w + o * w_stride + i);
      for (int t = 0; t < NT; ++t) acc[o][t] = _mm256_fmadd_ps(row, x[t], acc[o][t]);
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
  switch (nt) {
    case 4: Block<W, NO, 4>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 3: Block<W, NO, 3>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate); break;
    case 2: Block<W, NO, 2>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate); break;
    default: Block<W, NO, 1>(w, w_stride, bias, depth, in, in_step, out, out_stride, accumulate);
  }
}

template <typename W>
void DenseAvx2(const W* w, std::size_t w_stride, const float* bias, int outputs, int depth,
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
// Transposed weights: NT steps by NV vectors of 8 outputs, accumulated over
// the depth with each input broadcast.
template <typename W, int NT, int NV>
void TileT(const W* w, int outputs, const float* bias, int depth, const float* in,
           std::size_t in_step, float* out, std::size_t out_stride, bool accumulate) {
  __m256 acc[NT][NV];
  for (int t = 0; t < NT; ++t)
    for (int v = 0; v < NV; ++v) acc[t][v] = _mm256_setzero_ps();
  for (int i = 0; i < depth; ++i) {
    __m256 row[NV];
    for (int v = 0; v < NV; ++v) row[v] = Load8(w + static_cast<std::size_t>(i) * outputs + 8 * v);
    for (int t = 0; t < NT; ++t) {
      const __m256 x = _mm256_set1_ps(in[t * in_step + i]);
      for (int v = 0; v < NV; ++v) acc[t][v] = _mm256_fmadd_ps(row[v], x, acc[t][v]);
    }
  }
  for (int t = 0; t < NT; ++t) {
    for (int v = 0; v < NV; ++v) {
      __m256 value = acc[t][v];
      if (bias) value = _mm256_add_ps(value, _mm256_loadu_ps(bias + 8 * v));
      float* y = out + t * out_stride + 8 * v;
      if (accumulate) value = _mm256_add_ps(value, _mm256_loadu_ps(y));
      _mm256_storeu_ps(y, value);
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

// Outputs 16 at a time, six steps at a time; outputs past a multiple of 8
// one at a time.
template <typename W>
void DenseTAvx2(const W* w, const float* bias, int outputs, int depth, const float* in,
                std::size_t in_step, int steps, float* out, std::size_t out_stride,
                bool accumulate) {
  const int vectors = outputs / 8;
  for (int o = 0; o < vectors * 8; o += 16) {
    const float* bo = bias ? bias + o : nullptr;
    const bool two = vectors - o / 8 >= 2;
    for (int t = 0; t < steps; t += 6) {
      const int nt = std::min(6, steps - t);
      const float* x = in + t * in_step;
      float* y = out + t * out_stride + o;
      if (two) {
        StepsT<W, 2>(nt, w + o, outputs, bo, depth, x, in_step, y, out_stride, accumulate);
      } else {
        StepsT<W, 1>(nt, w + o, outputs, bo, depth, x, in_step, y, out_stride, accumulate);
      }
    }
  }
  for (int o = vectors * 8; o < outputs; ++o) {
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
// exp(x) after Cephes expf: range reduction by ln 2, degree-5 polynomial.
inline __m256 Exp(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
  __m256 n = _mm256_floor_ps(
      _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  const __m256i exponent = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(exponent));
}

inline __m256 Elu(__m256 v) {
  const __m256 negative = _mm256_sub_ps(Exp(v), _mm256_set1_ps(1.0f));
  return _mm256_blendv_ps(v, negative, _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ));
}

inline __m256 Gelu(__m256 v) {
  const __m256 z = _mm256_mul_ps(v, _mm256_set1_ps(0.70710678118654752f));
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 a = _mm256_andnot_ps(sign_mask, z);
  const __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                 _mm256_fmadd_ps(a, _mm256_set1_ps(kErfP), _mm256_set1_ps(1.0f)));
  __m256 p = _mm256_set1_ps(kErfA5);
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(kErfA4));
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(kErfA3));
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(kErfA2));
  p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(kErfA1));
  p = _mm256_mul_ps(p, t);
  const __m256 e = Exp(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(a, a)));
  __m256 erf = _mm256_fnmadd_ps(p, e, _mm256_set1_ps(1.0f));
  erf = _mm256_or_ps(erf, _mm256_and_ps(z, sign_mask));  // erf is odd
  return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), v),
                       _mm256_add_ps(_mm256_set1_ps(1.0f), erf));
}

// Applies f to n values, the tail through a zero-padded vector.
template <__m256 (*F)(__m256)>
void Apply(float* x, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, F(_mm256_loadu_ps(x + i)));
  if (i < n) {
    alignas(32) float lane[8] = {};
    std::copy(x + i, x + n, lane);
    _mm256_store_ps(lane, F(_mm256_load_ps(lane)));
    std::copy(lane, lane + (n - i), x + i);
  }
}

}  // namespace

const Table kAvx2 = {"avx2",
                     true,
                     DenseAvx2<float>,
                     DenseTAvx2<float>,
                     DenseAvx2<std::uint16_t>,
                     DenseTAvx2<std::uint16_t>,
                     Apply<Elu>,
                     Apply<Gelu>};

}  // namespace kernels
}  // namespace vocoscodec
