#include "vocoscodec/fft.h"

#include <cmath>
#include <stdexcept>

namespace vocoscodec {

InverseRealFft::InverseRealFft(int n) : n_(n) {
  if (n < 2 || n % 2) throw std::invalid_argument("inverse real FFT length must be even");
  int remaining = n;
  for (int radix : {4, 2, 3, 5}) {
    while (remaining % radix == 0) {
      remaining /= radix;
      factors_.push_back(radix);
      factors_.push_back(remaining);
    }
  }
  if (remaining != 1) throw std::invalid_argument("FFT length has a factor above 5");
  twiddles_.resize(n);
  const double pi = 3.14159265358979323846;
  for (int k = 0; k < n; ++k) {  // inverse transform: e^{+2 pi i k / n}
    const double angle = 2.0 * pi * k / n;
    twiddles_[k] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
  }
  full_.resize(n);
  result_.resize(n);
  scratch_.resize(5);
}

// Decimation in time, one radix per level (after KISS FFT's kf_work).
void InverseRealFft::Work(std::complex<float>* out, const std::complex<float>* in,
                          int fstride, const int* factors) {
  const int radix = factors[0];
  const int m = factors[1];
  if (m == 1) {
    for (int j = 0; j < radix; ++j) out[j] = in[j * fstride];
  } else {
    for (int j = 0; j < radix; ++j) {
      Work(out + j * m, in + j * fstride, fstride * radix, factors + 2);
    }
  }
  for (int u = 0; u < m; ++u) {
    for (int q = 0; q < radix; ++q) scratch_[q] = out[u + q * m];
    for (int q1 = 0; q1 < radix; ++q1) {
      const int k = u + q1 * m;
      std::complex<float> sum = scratch_[0];
      for (int q = 1; q < radix; ++q) {
        const long index = (static_cast<long>(fstride) * q * k) % n_;
        sum += scratch_[q] * twiddles_[index];
      }
      out[k] = sum;
    }
  }
}

void InverseRealFft::Run(const std::complex<float>* spectrum, float* samples) {
  const int half = n_ / 2;
  full_[0] = {spectrum[0].real(), 0.0f};
  full_[half] = {spectrum[half].real(), 0.0f};
  for (int k = 1; k < half; ++k) {
    full_[k] = spectrum[k];
    full_[n_ - k] = std::conj(spectrum[k]);
  }
  Work(result_.data(), full_.data(), 1, factors_.data());
  const float scale = 1.0f / n_;
  for (int i = 0; i < n_; ++i) samples[i] = result_[i].real() * scale;
}

}  // namespace vocoscodec
