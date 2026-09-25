// Minimal checks for the vocoscodec tests (no test framework dependency).

#ifndef VOCOSCODEC_TESTS_TEST_UTIL_H_
#define VOCOSCODEC_TESTS_TEST_UTIL_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace test {

inline int& Failures() {
  static int failures = 0;
  return failures;
}

inline void Check(bool ok, const std::string& what, const char* file, int line) {
  if (!ok) {
    std::fprintf(stderr, "%s:%d: FAILED: %s\n", file, line, what.c_str());
    ++Failures();
  }
}

inline int Finish(const char* name) {
  if (Failures()) {
    std::fprintf(stderr, "%s: %d failure(s)\n", name, Failures());
    return EXIT_FAILURE;
  }
  std::printf("%s: passed\n", name);
  return EXIT_SUCCESS;
}

inline float MaxAbsDiff(const float* a, const float* b, std::size_t n) {
  float worst = 0.0f;
  for (std::size_t i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs(a[i] - b[i]));
  return worst;
}

// Signal-to-difference ratio of b against a, in dB.
inline double SnrDb(const float* a, const float* b, std::size_t n) {
  double signal = 0.0, noise = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    signal += double(a[i]) * a[i];
    noise += double(a[i] - b[i]) * (a[i] - b[i]);
  }
  return 10.0 * std::log10((signal + 1e-30) / (noise + 1e-30));
}

}  // namespace test

#define CHECK(condition) test::Check((condition), #condition, __FILE__, __LINE__)
#define CHECK_MSG(condition, message) test::Check((condition), (message), __FILE__, __LINE__)

#endif  // VOCOSCODEC_TESTS_TEST_UTIL_H_
