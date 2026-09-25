// Numeric kernels. Activations are time-major: row t holds the channels of
// time step t. The instruction set is chosen once at run time: on x86-64
// AVX2 with FMA and F16C if the CPU has them, else SSE4.2; NEON on arm64;
// portable C++ otherwise.

#ifndef VOCOSCODEC_KERNELS_H_
#define VOCOSCODEC_KERNELS_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace vocoscodec {
namespace kernels {

// For t < steps, o < outputs:
//   out[t * out_stride + o] (+)= bias[o] + sum_{i<depth} w[o * w_stride + i] * in[t * in_step + i]
// `bias` may be null. With accumulate the result is added to `out`.
// A convolution over time-major input with weights laid out [out][tap][in]
// is this with depth = taps * channels and in_step = stride * channels.
void Dense(const float* w, std::size_t w_stride, const float* bias, int outputs, int depth,
           const float* in, std::size_t in_step, int steps, float* out,
           std::size_t out_stride, bool accumulate);

// The same with the weights transposed: w[i * outputs + o]. Faster when
// `depth` is small or `steps` large (vectorised over outputs, no horizontal
// sums); Dense suits a single step over long rows.
void DenseT(const float* w, const float* bias, int outputs, int depth, const float* in,
            std::size_t in_step, int steps, float* out, std::size_t out_stride,
            bool accumulate);

// Dense and DenseT with IEEE half precision weights (half.h), converted to
// float as they are read: half the memory traffic, float arithmetic.
void Dense(const std::uint16_t* w, std::size_t w_stride, const float* bias, int outputs,
           int depth, const float* in, std::size_t in_step, int steps, float* out,
           std::size_t out_stride, bool accumulate);
void DenseT(const std::uint16_t* w, const float* bias, int outputs, int depth, const float* in,
            std::size_t in_step, int steps, float* out, std::size_t out_stride,
            bool accumulate);

// x = x > 0 ? x : exp(x) - 1
void Elu(float* x, std::size_t n);
// x = 0.5 x (1 + erf(x / sqrt 2))
void Gelu(float* x, std::size_t n);

// "avx2", "sse4.2", "neon" or "generic"
const char* InstructionSet();
// True when these kernels convert half-precision weights in hardware (F16C,
// NEON), so that half weights are faster than float ones.
bool HardwareHalf();
// Use the named kernels ("avx2", "sse4.2", "neon", "generic"), or the best
// this CPU runs for "" or "auto". False if they are not built or not supported
// here; the selection is then unchanged. For tests and benchmarks: call
// before any encoder or decoder is used.
bool Select(const std::string& name);
// Force the portable kernels (true), or the best again (false).
void UseGeneric(bool generic);

}  // namespace kernels
}  // namespace vocoscodec

#endif  // VOCOSCODEC_KERNELS_H_
