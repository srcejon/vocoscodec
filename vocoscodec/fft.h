// Inverse real FFT of any length whose factors are 2, 3, 4 and 5.

#ifndef VOCOSCODEC_FFT_H_
#define VOCOSCODEC_FFT_H_

#include <complex>
#include <vector>

namespace vocoscodec {

class InverseRealFft {
 public:
  explicit InverseRealFft(int n);

  // n / 2 + 1 bins -> n samples, scaled by 1 / n (numpy / torch irfft). The
  // imaginary parts of the DC and Nyquist bins are ignored, as irfft does.
  void Run(const std::complex<float>* spectrum, float* samples);
  int size() const { return n_; }

 private:
  void Work(std::complex<float>* out, const std::complex<float>* in, int fstride,
            const int* factors);

  int n_;
  std::vector<int> factors_;  // pairs (radix, remaining length)
  std::vector<std::complex<float>> twiddles_;
  std::vector<std::complex<float>> full_, result_, scratch_;
};

}  // namespace vocoscodec

#endif  // VOCOSCODEC_FFT_H_
