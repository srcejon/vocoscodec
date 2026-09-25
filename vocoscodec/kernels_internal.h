// Per-instruction-set kernel tables, selected by kernels.cc.

#ifndef VOCOSCODEC_KERNELS_INTERNAL_H_
#define VOCOSCODEC_KERNELS_INTERNAL_H_

#include <cstddef>
#include <cstdint>

namespace vocoscodec {
namespace kernels {

template <typename W>
using DenseFn = void (*)(const W* w, std::size_t w_stride, const float* bias, int outputs,
                         int depth, const float* in, std::size_t in_step, int steps,
                         float* out, std::size_t out_stride, bool accumulate);
template <typename W>
using DenseTFn = void (*)(const W* w, const float* bias, int outputs, int depth,
                          const float* in, std::size_t in_step, int steps, float* out,
                          std::size_t out_stride, bool accumulate);

struct Table {
  const char* name;
  bool hardware_half;  // converts half weights in hardware (F16C, NEON)
  DenseFn<float> dense;
  DenseTFn<float> dense_t;
  DenseFn<std::uint16_t> dense_h;
  DenseTFn<std::uint16_t> dense_t_h;
  void (*elu)(float* x, std::size_t n);
  void (*gelu)(float* x, std::size_t n);
};

extern const Table kGeneric;
#if defined(VOCOSCODEC_HAVE_AVX2)
extern const Table kAvx2;
#endif
#if defined(VOCOSCODEC_HAVE_SSE42)
extern const Table kSse42;
#endif
#if defined(VOCOSCODEC_HAVE_NEON)
extern const Table kNeon;
#endif

// erf as in Abramowitz and Stegun 7.1.26 (|error| < 1.5e-7), shared by the
// SIMD GELUs; the generic one calls std::erf.
constexpr float kErfP = 0.3275911f;
constexpr float kErfA1 = 0.254829592f;
constexpr float kErfA2 = -0.284496736f;
constexpr float kErfA3 = 1.421413741f;
constexpr float kErfA4 = -1.453152027f;
constexpr float kErfA5 = 1.061405429f;

}  // namespace kernels
}  // namespace vocoscodec

#endif  // VOCOSCODEC_KERNELS_INTERNAL_H_
