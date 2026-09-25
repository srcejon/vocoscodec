#include "vocoscodec/kernels.h"

#include <atomic>
#include <cmath>
#include <type_traits>
#include <vector>

#include "vocoscodec/half.h"
#include "vocoscodec/kernels_internal.h"

#if defined(VOCOSCODEC_HAVE_AVX2) || defined(VOCOSCODEC_HAVE_SSE42)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace vocoscodec {
namespace kernels {
namespace {

// Every half value as a float (256 kB), for the portable kernels.
const std::vector<float>& HalfTable() {
  static const std::vector<float> table = [] {
    std::vector<float> t(65536);
    for (std::uint32_t h = 0; h < 65536; ++h) t[h] = HalfToFloat(static_cast<std::uint16_t>(h));
    return t;
  }();
  return table;
}

// Reads weights as floats: float storage directly, half through the table.
struct Reader {
  const float* table;
  float operator()(const float* w) const { return *w; }
  float operator()(const std::uint16_t* w) const { return table[*w]; }
};

template <typename W>
Reader MakeReader() {
  return Reader{std::is_same<W, std::uint16_t>::value ? HalfTable().data() : nullptr};
}

template <typename W>
void DenseGeneric(const W* w, std::size_t w_stride, const float* bias, int outputs, int depth,
                  const float* in, std::size_t in_step, int steps, float* out,
                  std::size_t out_stride, bool accumulate) {
  const Reader weight = MakeReader<W>();
  for (int t = 0; t < steps; ++t) {
    const float* x = in + t * in_step;
    float* y = out + t * out_stride;
    for (int o = 0; o < outputs; ++o) {
      const W* row = w + o * w_stride;
      float sum = 0.0f;
      for (int i = 0; i < depth; ++i) sum += weight(row + i) * x[i];
      if (bias) sum += bias[o];
      y[o] = accumulate ? y[o] + sum : sum;
    }
  }
}

template <typename W>
void DenseTGeneric(const W* w, const float* bias, int outputs, int depth, const float* in,
                   std::size_t in_step, int steps, float* out, std::size_t out_stride,
                   bool accumulate) {
  const Reader weight = MakeReader<W>();
  std::vector<float> sum(outputs);
  for (int t = 0; t < steps; ++t) {
    const float* x = in + t * in_step;
    float* y = out + t * out_stride;
    for (int o = 0; o < outputs; ++o) sum[o] = 0.0f;
    for (int i = 0; i < depth; ++i) {
      const W* row = w + static_cast<std::size_t>(i) * outputs;
      const float v = x[i];
      for (int o = 0; o < outputs; ++o) sum[o] += weight(row + o) * v;
    }
    for (int o = 0; o < outputs; ++o) {
      const float value = bias ? sum[o] + bias[o] : sum[o];
      y[o] = accumulate ? y[o] + value : value;
    }
  }
}

void EluGeneric(float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] < 0.0f) x[i] = std::expm1(x[i]);
  }
}

void GeluGeneric(float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * 0.70710678118654752f));
  }
}

#if defined(VOCOSCODEC_HAVE_AVX2)
bool CpuHasAvx2() {
  int info[4];
#if defined(_MSC_VER)
  __cpuid(info, 0);
  if (info[0] < 7) return false;
  __cpuid(info, 1);
#else
  unsigned a, b, c, d;
  if (__get_cpuid_max(0, nullptr) < 7) return false;
  __cpuid(1, a, b, c, d);
  info[2] = static_cast<int>(c);
#endif
  const bool fma = info[2] & (1 << 12);
  const bool osxsave = info[2] & (1 << 27);
  const bool avx = info[2] & (1 << 28);
  const bool f16c = info[2] & (1 << 29);
  if (!(fma && osxsave && avx && f16c)) return false;
#if defined(_MSC_VER)
  const unsigned long long xcr0 = _xgetbv(0);
  __cpuidex(info, 7, 0);
  const bool avx2 = info[1] & (1 << 5);
#else
  unsigned xlo, xhi;
  __asm__("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
  const unsigned long long xcr0 = xlo | (static_cast<unsigned long long>(xhi) << 32);
  __cpuid_count(7, 0, a, b, c, d);
  const bool avx2 = b & (1 << 5);
#endif
  return avx2 && (xcr0 & 6) == 6;  // the OS saves the YMM state
}
#endif

#if defined(VOCOSCODEC_HAVE_SSE42)
bool CpuHasSse42() {
  int info[4];
#if defined(_MSC_VER)
  __cpuid(info, 1);
#else
  unsigned a, b, c, d;
  __cpuid(1, a, b, c, d);
  info[2] = static_cast<int>(c);
#endif
  const bool sse3 = info[2] & (1 << 0);
  const bool ssse3 = info[2] & (1 << 9);
  const bool sse41 = info[2] & (1 << 19);
  const bool sse42 = info[2] & (1 << 20);
  return sse3 && ssse3 && sse41 && sse42;
}
#endif

// The kernels named, if built and supported by this CPU.
const Table* Find(const std::string& name) {
#if defined(VOCOSCODEC_HAVE_AVX2)
  if (name == "avx2") return CpuHasAvx2() ? &kAvx2 : nullptr;
#endif
#if defined(VOCOSCODEC_HAVE_SSE42)
  if (name == "sse4.2") return CpuHasSse42() ? &kSse42 : nullptr;
#endif
#if defined(VOCOSCODEC_HAVE_NEON)
  if (name == "neon") return &kNeon;
#endif
  if (name == "generic") return &kGeneric;
  return nullptr;
}

const Table* Best() {
  for (const char* name : {"avx2", "sse4.2", "neon"}) {
    if (const Table* table = Find(name)) return table;
  }
  return &kGeneric;
}

std::atomic<const Table*> active{nullptr};

const Table& Active() {
  const Table* table = active.load(std::memory_order_acquire);
  if (!table) {
    table = Best();
    active.store(table, std::memory_order_release);
  }
  return *table;
}

}  // namespace

const Table kGeneric = {"generic",
                        false,
                        DenseGeneric<float>,
                        DenseTGeneric<float>,
                        DenseGeneric<std::uint16_t>,
                        DenseTGeneric<std::uint16_t>,
                        EluGeneric,
                        GeluGeneric};

void Dense(const float* w, std::size_t w_stride, const float* bias, int outputs, int depth,
           const float* in, std::size_t in_step, int steps, float* out,
           std::size_t out_stride, bool accumulate) {
  Active().dense(w, w_stride, bias, outputs, depth, in, in_step, steps, out, out_stride,
                 accumulate);
}

void DenseT(const float* w, const float* bias, int outputs, int depth, const float* in,
            std::size_t in_step, int steps, float* out, std::size_t out_stride,
            bool accumulate) {
  Active().dense_t(w, bias, outputs, depth, in, in_step, steps, out, out_stride, accumulate);
}

void Dense(const std::uint16_t* w, std::size_t w_stride, const float* bias, int outputs,
           int depth, const float* in, std::size_t in_step, int steps, float* out,
           std::size_t out_stride, bool accumulate) {
  Active().dense_h(w, w_stride, bias, outputs, depth, in, in_step, steps, out, out_stride,
                   accumulate);
}

void DenseT(const std::uint16_t* w, const float* bias, int outputs, int depth, const float* in,
            std::size_t in_step, int steps, float* out, std::size_t out_stride,
            bool accumulate) {
  Active().dense_t_h(w, bias, outputs, depth, in, in_step, steps, out, out_stride, accumulate);
}

void Elu(float* x, std::size_t n) { Active().elu(x, n); }
void Gelu(float* x, std::size_t n) { Active().gelu(x, n); }
const char* InstructionSet() { return Active().name; }
bool HardwareHalf() { return Active().hardware_half; }

bool Select(const std::string& name) {
  const Table* table = name.empty() || name == "auto" ? Best() : Find(name);
  if (!table) return false;
  active.store(table, std::memory_order_release);
  return true;
}

void UseGeneric(bool generic) {
  active.store(generic ? &kGeneric : Best(), std::memory_order_release);
}

}  // namespace kernels
}  // namespace vocoscodec
