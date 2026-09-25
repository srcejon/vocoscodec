// Mono 16-bit PCM WAV files, and the tools' packet files.
//
// A packet file is a sequence of records: one byte giving the packet's size,
// then the packet. Size 0 marks a lost packet.

#ifndef VOCOSCODEC_TOOLS_WAV_IO_H_
#define VOCOSCODEC_TOOLS_WAV_IO_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vocoscodec::tools {

bool ReadMonoWav(const std::filesystem::path& path, std::vector<std::int16_t>* samples,
                 int* sample_rate, std::string* error);
bool WriteMonoWav(const std::filesystem::path& path, const std::vector<std::int16_t>& samples,
                  int sample_rate, std::string* error);

bool ReadPackets(const std::filesystem::path& path,
                 std::vector<std::vector<std::uint8_t>>* packets, std::string* error);
bool WritePackets(const std::filesystem::path& path,
                  const std::vector<std::vector<std::uint8_t>>& packets, std::string* error);

}  // namespace vocoscodec::tools

#endif  // VOCOSCODEC_TOOLS_WAV_IO_H_
