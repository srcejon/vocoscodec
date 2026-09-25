// IEEE 754 binary16 conversion in software (portable and exact).

#ifndef VOCOSCODEC_HALF_H_
#define VOCOSCODEC_HALF_H_

#include <cstdint>
#include <cstring>

namespace vocoscodec {

// Round to nearest, ties to even (as numpy's astype(float16)).
inline std::uint16_t FloatToHalf(float value) {
  std::uint32_t f;
  std::memcpy(&f, &value, 4);
  const std::uint32_t sign = (f >> 16) & 0x8000u;
  const std::uint32_t magnitude = f & 0x7FFFFFFFu;
  if (magnitude >= 0x7F800000u) {  // inf or NaN
    return static_cast<std::uint16_t>(sign | 0x7C00u | (magnitude > 0x7F800000u ? 0x200u : 0u));
  }
  if (magnitude >= 0x477FF000u) return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow
  if (magnitude < 0x38800000u) {  // subnormal or zero in half precision
    if (magnitude < 0x33000000u) return static_cast<std::uint16_t>(sign);
    const std::uint32_t exponent = magnitude >> 23;
    const std::uint32_t mantissa = (magnitude & 0x7FFFFFu) | 0x800000u;
    const std::uint32_t shift = 126 - exponent;  // 14..24
    std::uint32_t half = mantissa >> shift;
    const std::uint32_t rest = mantissa & ((1u << shift) - 1);
    const std::uint32_t midpoint = 1u << (shift - 1);
    if (rest > midpoint || (rest == midpoint && (half & 1u))) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = ((magnitude >> 13) - (112u << 10));
  const std::uint32_t rest = magnitude & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;
  return static_cast<std::uint16_t>(sign | half);
}

inline float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exponent = (h >> 10) & 0x1Fu;
  std::uint32_t mantissa = h & 0x3FFu;
  std::uint32_t f;
  if (exponent == 0x1F) {
    f = sign | 0x7F800000u | (mantissa << 13);
  } else if (exponent == 0) {
    if (mantissa == 0) {
      f = sign;
    } else {  // subnormal: normalise
      exponent = 113;
      while (!(mantissa & 0x400u)) {
        mantissa <<= 1;
        --exponent;
      }
      f = sign | (exponent << 23) | ((mantissa & 0x3FFu) << 13);
    }
  } else {
    f = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &f, 4);
  return value;
}

}  // namespace vocoscodec

#endif  // VOCOSCODEC_HALF_H_
