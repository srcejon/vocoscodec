// SIMD kernels against the portable ones, and the FFT against a direct DFT.

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "vocoscodec/fft.h"
#include "vocoscodec/half.h"
#include "vocoscodec/kernels.h"
#include "vocoscodec/tests/test_util.h"

using vocoscodec::InverseRealFft;
namespace kernels = vocoscodec::kernels;

namespace {

std::vector<float> Random(std::mt19937& rng, std::size_t n, float scale = 1.0f) {
  std::normal_distribution<float> normal(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) x = normal(rng);
  return v;
}

// The SIMD kernel sets this CPU runs.
std::vector<std::string> SimdSets() {
  std::vector<std::string> sets;
  for (const char* name : {"avx2", "sse4.2", "neon"}) {
    if (kernels::Select(name)) sets.push_back(name);
  }
  kernels::Select("auto");
  return sets;
}

std::string current;  // the kernels TestDense and TestDenseT check

void TestDense(std::mt19937& rng) {
  // Every block shape and tail: outputs and steps 1..9, depths around the
  // 8- and 4-wide vector lengths, strided and overlapping inputs.
  for (int outputs = 1; outputs <= 9; ++outputs) {
    for (int steps = 1; steps <= 9; ++steps) {
      for (int depth : {1, 3, 7, 8, 9, 16, 31, 64, 100}) {
        for (int step : {depth, 2}) {
          const auto w = Random(rng, static_cast<std::size_t>(outputs) * (depth + 3));
          const auto b = Random(rng, outputs);
          const auto x = Random(rng, static_cast<std::size_t>(steps) * step + depth);
          for (bool accumulate : {false, true}) {
            std::vector<float> fast(static_cast<std::size_t>(steps) * (outputs + 2), 0.5f);
            std::vector<float> slow = fast;
            kernels::Select(current);
            kernels::Dense(w.data(), depth + 3, b.data(), outputs, depth, x.data(), step, steps,
                           fast.data(), outputs + 2, accumulate);
            kernels::UseGeneric(true);
            kernels::Dense(w.data(), depth + 3, b.data(), outputs, depth, x.data(), step, steps,
                           slow.data(), outputs + 2, accumulate);
            const float diff = test::MaxAbsDiff(fast.data(), slow.data(), fast.size());
            CHECK_MSG(diff < 1e-4f, "Dense " + std::to_string(outputs) + "x" +
                                        std::to_string(steps) + "x" + std::to_string(depth) +
                                        " differs by " + std::to_string(diff));
          }
        }
      }
    }
  }
  kernels::UseGeneric(false);
}

// DenseT with transposed weights against the portable Dense.
void TestDenseT(std::mt19937& rng) {
  for (int outputs : {1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 40, 1282}) {
    for (int steps : {1, 2, 5, 6, 7, 13}) {
      for (int depth : {1, 7, 16, 33}) {
        const auto w = Random(rng, static_cast<std::size_t>(outputs) * depth);
        std::vector<float> wt(w.size());
        for (int o = 0; o < outputs; ++o)
          for (int i = 0; i < depth; ++i) wt[static_cast<std::size_t>(i) * outputs + o] = w[o * depth + i];
        const auto b = Random(rng, outputs);
        const auto x = Random(rng, static_cast<std::size_t>(steps) * 3 + depth);
        for (bool accumulate : {false, true}) {
          std::vector<float> fast(static_cast<std::size_t>(steps) * (outputs + 1), 0.25f);
          std::vector<float> slow = fast, generic = fast;
          kernels::Select(current);
          kernels::DenseT(wt.data(), b.data(), outputs, depth, x.data(), 3, steps, fast.data(),
                          outputs + 1, accumulate);
          kernels::UseGeneric(true);
          kernels::DenseT(wt.data(), b.data(), outputs, depth, x.data(), 3, steps,
                          generic.data(), outputs + 1, accumulate);
          kernels::Dense(w.data(), depth, b.data(), outputs, depth, x.data(), 3, steps,
                         slow.data(), outputs + 1, accumulate);
          const float diff = std::fmax(test::MaxAbsDiff(fast.data(), slow.data(), fast.size()),
                                       test::MaxAbsDiff(generic.data(), slow.data(), fast.size()));
          CHECK_MSG(diff < 1e-4f, "DenseT " + std::to_string(outputs) + "x" +
                                      std::to_string(steps) + "x" + std::to_string(depth) +
                                      " differs by " + std::to_string(diff));
        }
      }
    }
  }
  kernels::UseGeneric(false);
}

// Half-precision weights: the chosen kernels against the portable ones.
void TestHalf(std::mt19937& rng) {
  for (int outputs : {1, 5, 8, 16, 17, 40}) {
    for (int steps : {1, 2, 7}) {
      for (int depth : {3, 8, 33, 128}) {
        const auto values = Random(rng, static_cast<std::size_t>(outputs) * depth);
        std::vector<std::uint16_t> w(values.size());
        for (std::size_t i = 0; i < w.size(); ++i) w[i] = vocoscodec::FloatToHalf(values[i]);
        const auto b = Random(rng, outputs);
        const auto x = Random(rng, static_cast<std::size_t>(steps) * depth);
        for (bool transposed : {false, true}) {
          std::vector<float> fast(static_cast<std::size_t>(steps) * outputs), slow(fast.size());
          for (int pass = 0; pass < 2; ++pass) {
            kernels::Select(pass == 0 ? current : "generic");
            float* y = pass == 0 ? fast.data() : slow.data();
            if (transposed) {
              kernels::DenseT(w.data(), b.data(), outputs, depth, x.data(), depth, steps, y,
                              outputs, false);
            } else {
              kernels::Dense(w.data(), depth, b.data(), outputs, depth, x.data(), depth, steps,
                             y, outputs, false);
            }
          }
          const float diff = test::MaxAbsDiff(fast.data(), slow.data(), fast.size());
          CHECK_MSG(diff < 1e-4f, "half weights (" + current + ") differ by " +
                                      std::to_string(diff));
        }
      }
    }
  }
}

void TestActivations(std::mt19937& rng) {
  auto x = Random(rng, 10007, 4.0f);
  x.push_back(-100.0f);
  x.push_back(100.0f);
  x.push_back(0.0f);
  x.push_back(-1e-7f);
  auto elu = x, gelu = x;
  kernels::Elu(elu.data(), elu.size());
  kernels::Gelu(gelu.data(), gelu.size());
  float elu_error = 0.0f, gelu_error = 0.0f;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double v = x[i];
    const double elu_ref = v > 0 ? v : std::expm1(v);
    const double gelu_ref = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
    elu_error = std::fmax(elu_error, static_cast<float>(std::fabs(elu[i] - elu_ref)));
    gelu_error = std::fmax(gelu_error, static_cast<float>(std::fabs(gelu[i] - gelu_ref) /
                                                          std::fmax(1.0, std::fabs(v))));
  }
  std::printf("kernels (%s): ELU max error %.2e, GELU max relative error %.2e\n",
              kernels::InstructionSet(), elu_error, gelu_error);
  CHECK(elu_error < 2e-7f);
  CHECK(gelu_error < 2e-7f);
}

void TestFft(std::mt19937& rng) {
  for (int n : {8, 20, 1280}) {
    InverseRealFft fft(n);
    const int bins = n / 2 + 1;
    const auto re = Random(rng, bins), im = Random(rng, bins);
    std::vector<std::complex<float>> spectrum(bins);
    for (int k = 0; k < bins; ++k) spectrum[k] = {re[k], im[k]};
    std::vector<float> out(n);
    fft.Run(spectrum.data(), out.data());
    double worst = 0.0;
    const double pi = 3.14159265358979323846;
    for (int t = 0; t < n; ++t) {
      // irfft: DC and Nyquist real parts, other bins twice their real projection.
      double sum = re[0] + re[n / 2] * ((t % 2) ? -1.0 : 1.0);
      for (int k = 1; k < n / 2; ++k) {
        const double angle = 2.0 * pi * k * t / n;
        sum += 2.0 * (re[k] * std::cos(angle) - im[k] * std::sin(angle));
      }
      worst = std::fmax(worst, std::fabs(sum / n - out[t]));
    }
    CHECK_MSG(worst < 1e-5, "inverse FFT " + std::to_string(n) + " error " +
                                std::to_string(worst));
  }
}

}  // namespace

int main() {
  std::mt19937 rng(1234);
  for (const std::string& set : SimdSets()) {
    current = set;
    TestDense(rng);
    TestDenseT(rng);
    TestHalf(rng);
    kernels::Select(set);
    TestActivations(rng);
  }
  kernels::Select("auto");
  TestFft(rng);
  return test::Finish("vocoscodec_kernels_test");
}
