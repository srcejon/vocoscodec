#include "vocoscodec/packet.h"

#include "vocoscodec/encoder.h"

namespace vocoscodec {

int PacketBytes(int codebooks) {
  if (codebooks < 1 || codebooks > kMaxCodebooks) return 0;
  return (kBitsPerCodebook * codebooks + 7) / 8;
}

int PacketCodebooks(int bytes) {
  if (bytes <= 0) return 0;
  const int codebooks = bytes * 8 / kBitsPerCodebook;
  return PacketBytes(codebooks) == bytes ? codebooks : 0;
}

void Pack(const int* indices, int codebooks, std::uint8_t* packet) {
  std::uint32_t value = 0;
  int bits = 0;
  int out = 0;
  for (int q = 0; q < codebooks; ++q) {
    value = (value << kBitsPerCodebook) | (static_cast<std::uint32_t>(indices[q]) & 0x3FF);
    bits += kBitsPerCodebook;
    while (bits >= 8) {
      bits -= 8;
      packet[out++] = static_cast<std::uint8_t>(value >> bits);
    }
    value &= (1u << bits) - 1;
  }
  if (bits) packet[out] = static_cast<std::uint8_t>(value << (8 - bits));
}

int Unpack(const std::uint8_t* packet, int bytes, int* indices) {
  const int codebooks = PacketCodebooks(bytes);
  std::uint32_t value = 0;
  int bits = 0;
  int in = 0;
  for (int q = 0; q < codebooks; ++q) {
    while (bits < kBitsPerCodebook) {
      value = (value << 8) | packet[in++];
      bits += 8;
    }
    bits -= kBitsPerCodebook;
    indices[q] = static_cast<int>((value >> bits) & 0x3FF);
    value &= (1u << bits) - 1;
  }
  return codebooks;
}

int Truncate(std::uint8_t* packet, int bytes, int codebooks) {
  const int have = PacketCodebooks(bytes);
  if (have == 0 || codebooks < 1 || codebooks > have) return 0;
  const int size = PacketBytes(codebooks);
  const int spare = size * 8 - kBitsPerCodebook * codebooks;
  if (spare) packet[size - 1] &= static_cast<std::uint8_t>(0xFF << spare);
  return size;
}

}  // namespace vocoscodec
