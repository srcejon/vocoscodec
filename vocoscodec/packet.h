// Packet layout (normative, README.md "Packets"): the 10-bit codebook
// indices, coarse codebook first, each most significant bit first, packed
// without gaps and padded with zero bits to a whole byte. A packet of n
// codebooks is ceil(10 n / 8) bytes; its first 10 m bits are the packet of
// m < n codebooks, so a relay truncates by keeping ceil(10 m / 8) bytes and
// clearing the bits after 10 m.

#ifndef VOCOSCODEC_PACKET_H_
#define VOCOSCODEC_PACKET_H_

#include <cstddef>
#include <cstdint>

namespace vocoscodec {

inline constexpr int kBitsPerCodebook = 10;

// Bytes in a packet of `codebooks` (1..16), 0 otherwise.
int PacketBytes(int codebooks);
// Codebooks in a packet of `bytes`, 0 if no count gives that size.
int PacketCodebooks(int bytes);
// Writes PacketBytes(codebooks) bytes.
void Pack(const int* indices, int codebooks, std::uint8_t* packet);
// Reads PacketCodebooks(bytes) indices; returns that count (0: invalid size).
int Unpack(const std::uint8_t* packet, int bytes, int* indices);
// Truncates a packet in place to fewer codebooks; returns the new size.
int Truncate(std::uint8_t* packet, int bytes, int codebooks);

}  // namespace vocoscodec

#endif  // VOCOSCODEC_PACKET_H_
